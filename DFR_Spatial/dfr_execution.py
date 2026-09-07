"""Runtime bridge from strict official DFR token-state to one real Comfy/LTX denoise step.

Phase B-05 adds the first actual transformer execution:
- materialize the official token-state into the exact 5-D latent/mask/keyframe inputs
  expected by ComfyUI's native LTX model;
- run one real denoise pass through the loaded MODEL + CONDITIONING;
- repack the returned 5-D x0 prediction back into the strict official token layout;
- perform the deterministic official Euler update in token space.

This keeps the authoritative state in the strict official token layout while using
ComfyUI's native LTX runtime only as the model execution backend.
"""

from __future__ import annotations

from typing import Any

import torch

from .dfr_model_bridge import DFRComfyModelInput, materialize_comfy_model_input
from .latent_state import (
    OFFICIAL_STATE_KEY,
    clone_or_create_official_state,
    ensure_token_state,
)


def _normalize_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(sigmas):
        raise ValueError(f"Expected SIGMAS as a torch.Tensor, got {type(sigmas).__name__}.")
    if sigmas.ndim != 1:
        raise ValueError(f"Expected a rank-1 sigma tensor, got shape={tuple(sigmas.shape)}.")
    if sigmas.numel() < 2:
        raise ValueError(f"Sigma tensor must contain at least 2 entries, got {sigmas.numel()}.")
    return sigmas.detach().to(dtype=torch.float32)


def _require_sampling_runtime():
    try:
        import comfy.samplers  # type: ignore
        import comfy.sampler_helpers  # type: ignore
        import comfy.model_patcher  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime-only dependency inside ComfyUI
        raise RuntimeError(
            "This node must run inside a ComfyUI runtime with sampler/model helpers available."
        ) from exc
    return comfy.samplers, comfy.sampler_helpers, comfy.model_patcher


def _inject_ltx_runtime_metadata(conditioning: Any, model_input: DFRComfyModelInput) -> Any:
    """Attach bridge-generated LTX model metadata to a normal Comfy CONDITIONING.

    Comfy's public CONDITIONING socket is a list of ``[cross_attn, metadata]`` pairs.
    ``sampling_function()`` cannot consume that public representation directly; it first
    has to be converted and run through ``process_conds()``.  The LTX-specific runtime
    fields also live in the conditioning metadata, so inject the bridge values before
    conversion exactly as stock LTX conditioning nodes do.
    """
    if conditioning is None:
        return None
    values = {
        "frame_rate": float(model_input.frame_rate),
        "denoise_mask": model_input.denoise_mask,
    }
    if model_input.keyframe_idxs is not None:
        values["keyframe_idxs"] = model_input.keyframe_idxs
    if model_input.generated_keyframes is not None:
        values["generated_keyframes"] = model_input.generated_keyframes

    out = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "Expected a normal Comfy CONDITIONING entry [cross_attn, metadata], "
                f"got {type(item).__name__}: {item!r}."
            )
        cross_attn, metadata = item
        metadata = dict(metadata)
        metadata.update(values)
        out.append([cross_attn, metadata])
    return out


def _unpatchify_base_tokens(tokens: torch.Tensor, base_shape: tuple[int, int, int, int, int]) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,C], got {tuple(tokens.shape)}.")
    b, c, t, h, w = base_shape
    needed = t * h * w
    if tokens.shape[0] != b or tokens.shape[2] != c:
        raise ValueError(
            f"Token tensor {tuple(tokens.shape)} is incompatible with base shape {base_shape}."
        )
    if tokens.shape[1] < needed:
        raise ValueError(
            f"Token tensor has only {tokens.shape[1]} tokens, but base shape {base_shape} needs at least {needed}."
        )
    base = tokens[:, :needed].reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
    return base


def _pack_materialized_frames(frames_5d: torch.Tensor) -> torch.Tensor:
    if frames_5d.ndim != 5:
        raise ValueError(f"Expected materialized latent [B,C,T,H,W], got {tuple(frames_5d.shape)}.")
    b, c, t, h, w = frames_5d.shape
    return frames_5d.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)


def _extract_official_tokens_from_materialized_output(
    materialized_output: torch.Tensor,
    model_input: DFRComfyModelInput,
) -> torch.Tensor:
    """Invert the materialization packing for a returned Comfy 5-D denoised latent.

    The returned tensor matches the strict official token ordering:
    base canvas tokens first, then each appended keyframe/reference group in the
    same order used during materialization.
    """
    if materialized_output.ndim != 5:
        raise ValueError(
            f"Expected model output shaped [B,C,T,H,W], got {tuple(materialized_output.shape)}."
        )
    if materialized_output.shape[0] != model_input.x.shape[0] or materialized_output.shape[1] != model_input.x.shape[1]:
        raise ValueError(
            "Model output batch/channels do not match the materialized model input: "
            f"output={tuple(materialized_output.shape)}, input={tuple(model_input.x.shape)}."
        )
    packed_all = _pack_materialized_frames(materialized_output)
    base_token_count = int(model_input.base_frames) * int(materialized_output.shape[3]) * int(materialized_output.shape[4])
    base_tokens = packed_all[:, : base_token_count]

    extras: list[torch.Tensor] = []
    cursor = int(model_input.base_frames)
    latent_h = int(model_input.x.shape[3])
    latent_w = int(model_input.x.shape[4])

    for op in model_input.operation_frames:
        frames = int(op["materialized_frames"])
        frame_block = materialized_output[:, :, cursor : cursor + frames]
        cursor += frames
        if frame_block.shape[2] != frames:
            raise ValueError(
                f"Could not slice {frames} materialized frames for operation {op.get('type')} from model output."
            )

        if bool(op.get("sparse", False)):
            source_h = int(op["source_height"])
            source_w = int(op["source_width"])
            h_factor = max(1, latent_h // source_h)
            w_factor = max(1, latent_w // source_w)
            if source_h * h_factor != latent_h or source_w * w_factor != latent_w:
                raise ValueError(
                    "Sparse operation cannot be repacked cleanly because source shape does not divide the materialized grid: "
                    f"source=({source_h},{source_w}), materialized=({latent_h},{latent_w})."
                )
            frame_block = frame_block[:, :, :, ::h_factor, ::w_factor]
        packed = _pack_materialized_frames(frame_block)
        extras.append(packed)

    if cursor != int(materialized_output.shape[2]):
        raise ValueError(
            "Materialized output temporal length was not fully consumed during repack: "
            f"consumed={cursor}, total={int(materialized_output.shape[2])}."
        )

    if extras:
        return torch.cat([base_tokens] + extras, dim=1)
    return base_tokens


def _latent_with_replaced_token_latent(source_latent: dict[str, Any], token_latent: torch.Tensor) -> dict[str, Any]:
    state = clone_or_create_official_state(source_latent)
    token_state = ensure_token_state(
        state,
        target_samples=source_latent["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    if token_latent.shape != token_state["latent"].shape:
        raise ValueError(
            "Replacement token latent shape mismatch: "
            f"new={tuple(token_latent.shape)}, existing={tuple(token_state['latent'].shape)}."
        )
    token_state["latent"] = token_latent.clone()

    base_samples = _unpatchify_base_tokens(token_latent, tuple(state["base_shape"]))
    out = source_latent.copy()
    out["samples"] = base_samples.to(device=source_latent["samples"].device, dtype=source_latent["samples"].dtype)
    out[OFFICIAL_STATE_KEY] = state
    return out


def repack_comfy_denoised_to_official_state(
    source_official_state: dict[str, Any],
    model_input: DFRComfyModelInput,
    denoised_materialized: torch.Tensor,
) -> dict[str, Any]:
    tokens = _extract_official_tokens_from_materialized_output(denoised_materialized, model_input)
    return _latent_with_replaced_token_latent(source_official_state, tokens)


def execute_comfy_model_denoise(
    model_patcher: Any,
    positive: Any,
    negative: Any,
    noised_official_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
    cfg_scale: float = 1.0,
    seed: int = 0,
) -> tuple[dict[str, Any], DFRComfyModelInput, str]:
    comfy_samplers, sampler_helpers, comfy_model_patcher = _require_sampling_runtime()
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    step_index = int(step_index)
    if step_index < 0 or step_index >= sigma_schedule.numel() - 1:
        raise ValueError(
            f"step_index must be in [0, {sigma_schedule.numel() - 2}], got {step_index}."
        )

    model_input = materialize_comfy_model_input(noised_official_state)
    sigma_value = float(sigma_schedule[step_index].item())

    base_model = getattr(model_patcher, "model", None)
    original_model_options = getattr(model_patcher, "model_options", None)
    if base_model is None or original_model_options is None:
        raise ValueError(
            "Input MODEL does not look like a ComfyUI ModelPatcher: expected .model and .model_options."
        )

    # Public Comfy CONDITIONING is NOT the internal structure expected by
    # sampling_function/calc_cond_batch.  Reproduce CFGGuider's preparation path:
    # inject our LTX bridge metadata -> convert_cond -> prepare model/hooks ->
    # prepare_sampling -> process_conds -> one sampling_function call.
    positive_runtime = _inject_ltx_runtime_metadata(positive, model_input)
    negative_runtime = _inject_ltx_runtime_metadata(negative, model_input)
    conds = {
        "positive": sampler_helpers.convert_cond(positive_runtime),
        "negative": sampler_helpers.convert_cond(negative_runtime),
    }

    model_options = comfy_model_patcher.create_model_options_clone(original_model_options)
    comfy_samplers.preprocess_conds_hooks(conds)
    sampler_helpers.prepare_model_patcher(model_patcher, conds, model_options)
    comfy_samplers.filter_registered_hooks_on_conds(conds, model_options)

    loaded_models = []
    previous_patcher = getattr(base_model, "current_patcher", None)
    orig_hook_mode = getattr(model_patcher, "hook_mode", None)
    try:
        real_model, conds, loaded_models = sampler_helpers.prepare_sampling(
            model_patcher,
            model_input.x.shape,
            conds,
            model_options=model_options,
        )
        device = model_patcher.load_device
        x = model_input.x.to(device=device)
        sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
        denoise_mask = model_input.denoise_mask.to(device=device, dtype=torch.float32)

        conds = comfy_samplers.process_conds(
            real_model,
            x,
            conds,
            device,
            latent_image=None,
            denoise_mask=denoise_mask,
            seed=int(seed),
            latent_shapes=[tuple(x.shape)],
        )

        runtime_model_options = comfy_model_patcher.create_model_options_clone(model_options)
        runtime_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_schedule.to(device)
        comfy_samplers.cast_to_load_options(
            runtime_model_options,
            device=device,
            dtype=model_patcher.model_dtype(),
        )

        real_model.current_patcher = model_patcher
        model_patcher.pre_run()
        denoised_materialized = comfy_samplers.sampling_function(
            real_model,
            x,
            sigma,
            conds.get("negative", None),
            conds.get("positive", None),
            float(cfg_scale),
            model_options=runtime_model_options,
            seed=int(seed),
        )
    finally:
        base_model.current_patcher = previous_patcher
        try:
            model_patcher.cleanup()
        finally:
            if loaded_models:
                sampler_helpers.cleanup_models(conds, loaded_models)
            try:
                comfy_samplers.cast_to_load_options(model_options, device=model_patcher.offload_device)
            except Exception:
                pass
            if orig_hook_mode is not None:
                model_patcher.hook_mode = orig_hook_mode
            try:
                model_patcher.restore_hook_patches()
            except Exception:
                pass

    denoised_official = repack_comfy_denoised_to_official_state(
        source_official_state=noised_official_state,
        model_input=model_input,
        denoised_materialized=denoised_materialized,
    )

    report = (
        f"sigma={sigma_value:.9g}; step_index={step_index}; cfg_scale={float(cfg_scale):.9g}; "
        f"materialized_shape={tuple(int(x) for x in model_input.x.shape)}; "
        f"denoised_shape={tuple(int(x) for x in denoised_materialized.shape)}; "
        f"official_tokens={int(denoised_official[OFFICIAL_STATE_KEY]['token_state']['latent'].shape[1])}; "
        f"operations={len(model_input.operation_frames)}; seed={int(seed)}; "
        f"conditioning_prepared=True; ltx_runtime_metadata_injected=True."
    )
    return denoised_official, model_input, report


def post_process_official_denoised(
    noised_official_state: dict[str, Any],
    raw_denoised_official_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    noised_state = clone_or_create_official_state(noised_official_state)
    denoised_state = clone_or_create_official_state(raw_denoised_official_state)
    noised_tokens = ensure_token_state(
        noised_state,
        target_samples=noised_official_state["samples"],
        fps=float(noised_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in noised_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(noised_state.get("causal_fix", True)),
    )
    denoised_tokens = ensure_token_state(
        denoised_state,
        target_samples=raw_denoised_official_state["samples"],
        fps=float(denoised_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in denoised_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(denoised_state.get("causal_fix", True)),
    )
    if denoised_tokens["latent"].shape != noised_tokens["latent"].shape:
        raise ValueError(
            "Raw denoised token shape does not match the noised official state: "
            f"denoised={tuple(denoised_tokens['latent'].shape)}, noised={tuple(noised_tokens['latent'].shape)}."
        )
    # Comfy may execute/offload the transformer on CUDA while the strict official
    # state metadata (clean_latent / denoise_mask) remains on CPU.  Lightricks'
    # post_process_latent() is device-local tensor math, so explicitly move the
    # preserved operands to the model-output device before blending.
    execution_device = denoised_tokens["latent"].device
    denoised_f32 = denoised_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    mask_f32 = noised_tokens["denoise_mask"].to(device=execution_device, dtype=torch.float32)
    clean_f32 = noised_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
    blended = (
        denoised_f32 * mask_f32
        + clean_f32 * (1.0 - mask_f32)
    ).to(device=execution_device, dtype=denoised_tokens["latent"].dtype)
    out = _latent_with_replaced_token_latent(noised_official_state, blended)
    report = (
        f"token_shape={tuple(int(x) for x in blended.shape)}; "
        f"base_token_count={int(noised_tokens['base_token_count'])}; "
        f"execution_device={execution_device}; "
        f"raw_denoised_device={denoised_tokens['latent'].device}; "
        f"clean_source_device={noised_tokens['clean_latent'].device}; "
        f"mask_source_device={noised_tokens['denoise_mask'].device}; "
        f"mask_min={float(noised_tokens['denoise_mask'].min().item()):.9g}; "
        f"mask_max={float(noised_tokens['denoise_mask'].max().item()):.9g}."
    )
    return out, report


def euler_step_from_official_denoised(
    noised_official_state: dict[str, Any],
    raw_denoised_official_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    step_index = int(step_index)
    if step_index < 0 or step_index >= sigma_schedule.numel() - 1:
        raise ValueError(
            f"step_index must be in [0, {sigma_schedule.numel() - 2}], got {step_index}."
        )

    noised_state = clone_or_create_official_state(noised_official_state)
    noised_tokens = ensure_token_state(
        noised_state,
        target_samples=noised_official_state["samples"],
        fps=float(noised_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in noised_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(noised_state.get("causal_fix", True)),
    )
    post_processed_state, pp_report = post_process_official_denoised(noised_official_state, raw_denoised_official_state)
    pp_meta = clone_or_create_official_state(post_processed_state)
    pp_tokens = ensure_token_state(
        pp_meta,
        target_samples=post_processed_state["samples"],
        fps=float(pp_meta.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in pp_meta.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(pp_meta.get("causal_fix", True)),
    )

    # The post-processed x0 lives on the actual model execution device.  The
    # preserved pre-step latent may still be CPU-resident because Comfy moved
    # only the materialized model input.  Euler math must therefore follow x0.
    execution_device = pp_tokens["latent"].device
    x = noised_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    denoised = pp_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    sigma = float(sigma_schedule[step_index].item())
    sigma_next = float(sigma_schedule[step_index + 1].item())

    if sigma_next == 0.0:
        x_next = denoised
    else:
        if sigma == 0.0:
            raise ValueError("Current sigma is zero before the terminal step; cannot perform an Euler update.")
        sigma_ratio = sigma_next / sigma
        x_next = sigma_ratio * x + (1.0 - sigma_ratio) * denoised

    next_state = _latent_with_replaced_token_latent(
        noised_official_state,
        x_next.to(device=execution_device, dtype=noised_tokens["latent"].dtype),
    )
    report = (
        f"sigma={sigma:.9g}; sigma_next={sigma_next:.9g}; step_index={step_index}; "
        f"execution_device={execution_device}; "
        f"token_shape={tuple(int(x) for x in x_next.shape)}; base_token_count={int(noised_tokens['base_token_count'])}; "
        f"post_process=({pp_report})."
    )
    return next_state, post_processed_state, report


def _assert_stage1_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    """Require the exact 8-transition distilled Stage-1 schedule."""
    from .dfr_sigmas import stage_1_sigmas

    actual = _normalize_sigma_schedule(sigmas)
    expected = stage_1_sigmas()
    if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected):
        raise ValueError(
            "LTX Official Stage-1 Loop requires the exact DFR Stage-1 sigma schedule. "
            f"Got {actual.detach().cpu().tolist()}, expected {expected.tolist()}."
        )
    return actual


def _assert_static_model_input_compatible(reference: DFRComfyModelInput, current: DFRComfyModelInput) -> None:
    """The diffusion trajectory may change x only; DFR geometry/conditioning metadata must stay fixed."""
    scalar_fields = (
        "frame_rate",
        "base_frames",
        "appended_frames",
        "appended_tokens_before_filter",
        "appended_tokens_after_filter",
        "total_tokens_after_filter",
    )
    for field in scalar_fields:
        if getattr(reference, field) != getattr(current, field):
            raise RuntimeError(
                f"DFR model-input geometry changed during Stage 1: {field} "
                f"{getattr(reference, field)!r} -> {getattr(current, field)!r}."
            )
    if tuple(reference.x.shape) != tuple(current.x.shape):
        raise RuntimeError(
            f"DFR materialized x shape changed during Stage 1: {tuple(reference.x.shape)} -> {tuple(current.x.shape)}."
        )
    if reference.generated_keyframes != current.generated_keyframes:
        raise RuntimeError("generated_keyframes metadata changed during Stage-1 denoising.")
    if reference.operation_frames != current.operation_frames:
        raise RuntimeError("operation-frame layout changed during Stage-1 denoising.")


def extract_official_base_latent(final_official_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the authoritative base video tokens as a normal Comfy LATENT."""
    state = clone_or_create_official_state(final_official_state)
    token_state = ensure_token_state(
        state,
        target_samples=final_official_state["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    base = _unpatchify_base_tokens(token_state["latent"], tuple(state["base_shape"]))
    return {"samples": base.contiguous()}


def extract_official_generated_keyframes(final_official_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the authoritative generated slot tokens as [B,C,K,H,W] LATENT."""
    from .latent_state import extract_generated_keyframes

    return {"samples": extract_generated_keyframes(final_official_state).contiguous()}


def execute_stage1_denoising_loop(
    model_patcher: Any,
    positive: Any,
    negative: Any,
    noised_official_state: dict[str, Any],
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    """Run the complete official deterministic 8-step Spatial-DFR Stage-1 video loop.

    The loaded Comfy model is prepared once and remains active across all eight
    transformer evaluations.  The authoritative trajectory remains in our strict
    official token layout; only each current x is materialized into Comfy's 5-D
    representation for the native LTX model call.
    """
    import time

    comfy_samplers, sampler_helpers, comfy_model_patcher = _require_sampling_runtime()
    sigma_schedule = _assert_stage1_sigma_schedule(sigmas)

    # Stage 1 must begin from the Gaussian-noised official state and must carry
    # generated keyframe slots so they can be returned after the loop.
    initial_state = clone_or_create_official_state(noised_official_state)
    initial_tokens = ensure_token_state(
        initial_state,
        target_samples=noised_official_state["samples"],
        fps=float(initial_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in initial_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(initial_state.get("causal_fix", True)),
    )
    noiser_meta = initial_tokens.get("gaussian_noiser")
    if noiser_meta is None:
        raise ValueError("Stage-1 loop requires a state produced by LTX Official: Gaussian Noiser (DFR).")
    if abs(float(noiser_meta.get("noise_scale", -1.0)) - 1.0) > 1e-9:
        raise ValueError(
            "Official Spatial-DFR Stage 1 requires Gaussian noiser noise_scale=1.0; "
            f"state reports {noiser_meta.get('noise_scale')}."
        )
    if initial_tokens.get("generated_keyframe_layout") is None:
        raise ValueError("Stage-1 loop requires VideoGeneratedKeyframeSlots before noising.")

    reference_input = materialize_comfy_model_input(noised_official_state)
    base_model = getattr(model_patcher, "model", None)
    original_model_options = getattr(model_patcher, "model_options", None)
    if base_model is None or original_model_options is None:
        raise ValueError("Input MODEL does not look like a ComfyUI ModelPatcher: expected .model and .model_options.")

    positive_runtime = _inject_ltx_runtime_metadata(positive, reference_input)
    negative_runtime = _inject_ltx_runtime_metadata(negative, reference_input)
    conds = {
        "positive": sampler_helpers.convert_cond(positive_runtime),
        "negative": sampler_helpers.convert_cond(negative_runtime),
    }

    model_options = comfy_model_patcher.create_model_options_clone(original_model_options)
    comfy_samplers.preprocess_conds_hooks(conds)
    sampler_helpers.prepare_model_patcher(model_patcher, conds, model_options)
    comfy_samplers.filter_registered_hooks_on_conds(conds, model_options)

    loaded_models = []
    previous_patcher = getattr(base_model, "current_patcher", None)
    orig_hook_mode = getattr(model_patcher, "hook_mode", None)
    current_state = noised_official_state
    step_reports: list[str] = []
    total_started = time.perf_counter()

    try:
        real_model, conds, loaded_models = sampler_helpers.prepare_sampling(
            model_patcher,
            reference_input.x.shape,
            conds,
            model_options=model_options,
        )
        device = model_patcher.load_device
        initial_x = reference_input.x.to(device=device)
        denoise_mask = reference_input.denoise_mask.to(device=device, dtype=torch.float32)

        # Stock Comfy prepares public CONDITIONING once before the sampler loop;
        # follow the same lifecycle because geometry and conditionings do not
        # change between DFR Stage-1 sigma steps.
        conds = comfy_samplers.process_conds(
            real_model,
            initial_x,
            conds,
            device,
            latent_image=None,
            denoise_mask=denoise_mask,
            seed=int(seed),
            latent_shapes=[tuple(initial_x.shape)],
        )

        runtime_model_options = comfy_model_patcher.create_model_options_clone(model_options)
        runtime_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_schedule.to(device)
        comfy_samplers.cast_to_load_options(
            runtime_model_options,
            device=device,
            dtype=model_patcher.model_dtype(),
        )

        real_model.current_patcher = model_patcher
        model_patcher.pre_run()

        # Optional UI progress without making the module import Comfy at load time.
        try:
            import comfy.utils  # type: ignore

            progress = comfy.utils.ProgressBar(int(sigma_schedule.numel() - 1))
        except Exception:  # pragma: no cover - cosmetic only
            progress = None

        for step_index in range(int(sigma_schedule.numel() - 1)):
            step_started = time.perf_counter()
            model_input = materialize_comfy_model_input(current_state)
            _assert_static_model_input_compatible(reference_input, model_input)
            x = model_input.x.to(device=device)
            sigma_value = float(sigma_schedule[step_index].item())
            sigma_next = float(sigma_schedule[step_index + 1].item())
            sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)

            denoised_materialized = comfy_samplers.sampling_function(
                real_model,
                x,
                sigma,
                conds.get("negative", None),
                conds.get("positive", None),
                float(cfg_scale),
                model_options=runtime_model_options,
                seed=int(seed),
            )
            raw_denoised = repack_comfy_denoised_to_official_state(
                source_official_state=current_state,
                model_input=model_input,
                denoised_materialized=denoised_materialized,
            )
            current_state, _post_processed, _step_report = euler_step_from_official_denoised(
                noised_official_state=current_state,
                raw_denoised_official_state=raw_denoised,
                sigmas=sigma_schedule,
                step_index=step_index,
            )
            elapsed = time.perf_counter() - step_started
            step_reports.append(
                f"{step_index}:{sigma_value:.9g}->{sigma_next:.9g}@{elapsed:.2f}s"
            )
            if progress is not None:
                try:
                    progress.update(1)
                except Exception:
                    pass
    finally:
        base_model.current_patcher = previous_patcher
        try:
            model_patcher.cleanup()
        finally:
            if loaded_models:
                sampler_helpers.cleanup_models(conds, loaded_models)
            try:
                comfy_samplers.cast_to_load_options(model_options, device=model_patcher.offload_device)
            except Exception:
                pass
            if orig_hook_mode is not None:
                model_patcher.hook_mode = orig_hook_mode
            try:
                model_patcher.restore_hook_patches()
            except Exception:
                pass

    base_latent = extract_official_base_latent(current_state)
    generated_keyframes = extract_official_generated_keyframes(current_state)
    final_state_meta = clone_or_create_official_state(current_state)
    final_tokens = ensure_token_state(
        final_state_meta,
        target_samples=current_state["samples"],
        fps=float(final_state_meta.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in final_state_meta.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(final_state_meta.get("causal_fix", True)),
    )
    total_elapsed = time.perf_counter() - total_started
    layout = final_tokens["generated_keyframe_layout"]
    report = (
        f"PASS=True; stage=1; steps={int(sigma_schedule.numel()-1)}; cfg_scale={float(cfg_scale):.9g}; "
        f"seed={int(seed)}; execution_device={final_tokens['latent'].device}; "
        f"token_shape={tuple(int(x) for x in final_tokens['latent'].shape)}; "
        f"base_latent_shape={tuple(int(x) for x in base_latent['samples'].shape)}; "
        f"generated_keyframes_shape={tuple(int(x) for x in generated_keyframes['samples'].shape)}; "
        f"num_generated_keyframes={int(layout['num_keyframes'])}; total_time={total_elapsed:.2f}s; "
        f"trajectory=[{' | '.join(step_reports)}]."
    )
    return current_state, base_latent, generated_keyframes, report


def validate_stage1_loop_outputs(
    final_official_state: dict[str, Any],
    base_latent: dict[str, torch.Tensor],
    generated_keyframes: dict[str, torch.Tensor],
) -> tuple[bool, float, float, str]:
    """Validate that both exported Stage-1 latents are exact views of the final official token state."""
    expected_base = extract_official_base_latent(final_official_state)["samples"]
    expected_kf = extract_official_generated_keyframes(final_official_state)["samples"]
    actual_base = base_latent.get("samples")
    actual_kf = generated_keyframes.get("samples")
    if not torch.is_tensor(actual_base) or not torch.is_tensor(actual_kf):
        raise ValueError("base_latent and generated_keyframes must each contain a 'samples' tensor.")
    if tuple(actual_base.shape) != tuple(expected_base.shape):
        raise ValueError(f"base_latent shape {tuple(actual_base.shape)} != expected {tuple(expected_base.shape)}.")
    if tuple(actual_kf.shape) != tuple(expected_kf.shape):
        raise ValueError(f"generated_keyframes shape {tuple(actual_kf.shape)} != expected {tuple(expected_kf.shape)}.")
    base_err = float(
        (actual_base.to(device=expected_base.device, dtype=torch.float32) - expected_base.to(torch.float32)).abs().max().item()
    ) if expected_base.numel() else 0.0
    kf_err = float(
        (actual_kf.to(device=expected_kf.device, dtype=torch.float32) - expected_kf.to(torch.float32)).abs().max().item()
    ) if expected_kf.numel() else 0.0
    passed = base_err == 0.0 and kf_err == 0.0
    state = clone_or_create_official_state(final_official_state)
    token_state = ensure_token_state(
        state,
        target_samples=final_official_state["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    report = (
        f"PASS={passed}; base_max_abs_error={base_err:.9g}; generated_keyframes_max_abs_error={kf_err:.9g}; "
        f"base_shape={tuple(int(x) for x in actual_base.shape)}; "
        f"generated_keyframes_shape={tuple(int(x) for x in actual_kf.shape)}; "
        f"token_device={token_state['latent'].device}; token_dtype={token_state['latent'].dtype}; "
        f"num_generated_keyframes={int(token_state['generated_keyframe_layout']['num_keyframes'])}."
    )
    return passed, base_err, kf_err, report
