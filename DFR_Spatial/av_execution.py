"""Joint Stage-1 AV execution bridge for one real LTXAV transformer step.

Phase B-11 rewrites the execution half of B2 around ComfyUI's native LTXAV
sampling bridge instead of hunting for a raw ``forward(video, audio, ...)``
module on the MODEL socket.  This mirrors the earlier video-only fix:

- VIDEO remains authoritative in the strict official token-state format;
- AUDIO remains authoritative in its own strict official token-state format;
- for execution, the VIDEO branch is materialized into Comfy's native LTXV
  latent + denoise-mask + keyframe metadata geometry;
- the AUDIO branch is materialized into Comfy's native LTXAV audio latent
  geometry;
- both modalities are packed exactly the way Comfy's native AV bridge expects;
- one real x0 prediction is produced through ``comfy.samplers.sampling_function``;
- the packed multimodal output is unpacked and written back into the strict
  official VIDEO/AUDIO states.

A direct ``forward(video, audio, ...)`` fallback is still kept for pure-torch
unit tests and non-Comfy runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch

from .audio_state import (
    AUDIO_OFFICIAL_STATE_KEY,
    clone_audio_official_state,
    get_audio_official_state,
    _unpatchify_audio,
)
from .av_model_bridge import DFRStage1AVModelInput, materialize_stage1_av_model_input
from .dfr_model_bridge import DFRComfyModelInput, materialize_comfy_model_input, materialize_video_tokens_with_layout
from .memory_profile import SamplerMemoryProfile
from .dfr_noiser import NOISER_METADATA_KEY
from .dfr_sigmas import validate_custom_sigma_schedule
from .dfr_execution import (
    _assert_stage1_sigma_schedule,
    _assert_static_model_input_compatible,
    _extract_official_tokens_from_materialized_output,
    extract_official_base_latent,
    extract_official_generated_keyframes,
)
from .latent_state import (
    OFFICIAL_STATE_KEY,
    clone_or_create_official_state,
    ensure_token_state,
)
from .stage1_profile import Stage1RuntimeProfiler, profile_section


try:  # pragma: no cover - only available in real LTX/Comfy runtimes
    from ltx_core.model.transformer.modality import Modality as OfficialModality  # type: ignore
except Exception:  # pragma: no cover - exercised by pure-torch tests via fallback
    @dataclass(frozen=True)
    class OfficialModality:  # type: ignore[no-redef]
        latent: torch.Tensor
        sigma: torch.Tensor
        timesteps: torch.Tensor
        positions: torch.Tensor
        context: torch.Tensor
        enabled: bool = True
        context_mask: torch.Tensor | None = None
        attention_mask: torch.Tensor | None = None
        keyframes_mask: torch.Tensor | None = None


def _normalize_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(sigmas):
        raise ValueError(f"Expected SIGMAS as a torch.Tensor, got {type(sigmas).__name__}.")
    if sigmas.ndim != 1:
        raise ValueError(f"Expected a rank-1 sigma tensor, got shape={tuple(sigmas.shape)}.")
    if sigmas.numel() < 2:
        raise ValueError(f"Sigma tensor must contain at least 2 entries, got {sigmas.numel()}.")
    return sigmas.detach().to(dtype=torch.float32)


def _validate_custom_stage1_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    """Validate an advanced Stage-1 override without requiring official values."""
    return validate_custom_sigma_schedule(sigmas, "Custom Stage-1 sigmas")


def _unpatchify_video_base_tokens(tokens: torch.Tensor, base_shape: tuple[int, int, int, int, int]) -> torch.Tensor:
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
    return tokens[:, :needed].reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


def _unpatchify_audio_mask(mask_tokens: torch.Tensor, *, frames: int) -> torch.Tensor:
    if not torch.is_tensor(mask_tokens) or mask_tokens.ndim != 3 or mask_tokens.shape[-1] != 1:
        raise ValueError(f"audio mask tokens must have shape [B,T,1], got {getattr(mask_tokens, 'shape', None)}.")
    b, t, _ = mask_tokens.shape
    if t != int(frames):
        raise ValueError(f"audio mask token count {t} does not match audio latent time {frames}.")
    return mask_tokens.reshape(b, t, 1, 1).permute(0, 2, 1, 3).contiguous()


def _expand_mask_to_match_latent(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask) or not torch.is_tensor(latent):
        raise ValueError("mask and latent must be tensors.")
    if mask.shape[0] != latent.shape[0]:
        raise ValueError(f"Mask batch {mask.shape[0]} does not match latent batch {latent.shape[0]}.")
    if mask.ndim != latent.ndim:
        raise ValueError(
            f"Mask rank {mask.ndim} does not match latent rank {latent.ndim}; shapes={tuple(mask.shape)} vs {tuple(latent.shape)}."
        )
    expand_sizes = []
    for idx, (m, x) in enumerate(zip(mask.shape, latent.shape)):
        if idx == 0:
            expand_sizes.append(x)
            continue
        if m == x:
            expand_sizes.append(x)
            continue
        if m == 1:
            expand_sizes.append(x)
            continue
        raise ValueError(
            f"Mask dimension {idx} has size {m}, incompatible with latent size {x}; "
            f"mask={tuple(mask.shape)}, latent={tuple(latent.shape)}."
        )
    return mask.expand(*expand_sizes).contiguous().to(dtype=torch.float32)



def _video_latent_with_replaced_token_latent(source_latent: dict[str, Any], token_latent: torch.Tensor) -> dict[str, Any]:
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
            "Replacement video token latent shape mismatch: "
            f"new={tuple(token_latent.shape)}, existing={tuple(token_state['latent'].shape)}."
        )
    token_state["latent"] = token_latent.clone()
    base_samples = _unpatchify_video_base_tokens(token_latent, tuple(state["base_shape"]))
    out = source_latent.copy()
    out["samples"] = base_samples.to(device=source_latent["samples"].device, dtype=source_latent["samples"].dtype)
    out[OFFICIAL_STATE_KEY] = state
    return out



def _audio_latent_with_replaced_token_latent(source_latent: dict[str, Any], token_latent: torch.Tensor) -> dict[str, Any]:
    state = clone_audio_official_state(source_latent)
    token_state = state["token_state"]
    if token_latent.shape != token_state["latent"].shape:
        raise ValueError(
            "Replacement audio token latent shape mismatch: "
            f"new={tuple(token_latent.shape)}, existing={tuple(token_state['latent'].shape)}."
        )
    token_state["latent"] = token_latent.clone()
    rebuilt = _unpatchify_audio(
        token_latent,
        channels=int(token_state["channels"]),
        mel_bins=int(token_state["mel_bins"]),
    )
    out = source_latent.copy()
    out["samples"] = rebuilt.to(device=source_latent["samples"].device, dtype=source_latent["samples"].dtype)
    out[AUDIO_OFFICIAL_STATE_KEY] = state
    return out



def _extract_primary_conditioning_item(conditioning: Any, label: str) -> tuple[torch.Tensor, dict[str, Any], int]:
    if conditioning is None:
        raise ValueError(f"{label} conditioning is required.")
    if torch.is_tensor(conditioning):
        return conditioning, {}, 1
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        raise ValueError(
            f"Expected {label} CONDITIONING to be a non-empty list of [cross_attn, metadata] pairs."
        )
    item = conditioning[0]
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise ValueError(
            f"Expected {label} CONDITIONING entry [cross_attn, metadata], got {type(item).__name__}: {item!r}."
        )
    cross_attn, metadata = item
    if not torch.is_tensor(cross_attn):
        raise ValueError(f"{label} cross_attn must be a torch.Tensor, got {type(cross_attn).__name__}.")
    return cross_attn, dict(metadata or {}), len(conditioning)



def _resolve_context_fields(conditioning: Any, label: str) -> dict[str, Any]:
    cross_attn, metadata, item_count = _extract_primary_conditioning_item(conditioning, label)
    video_context = metadata.get("video_context")
    audio_context = metadata.get("audio_context")
    if video_context is None:
        video_context = cross_attn
    if audio_context is None:
        audio_context = cross_attn

    video_context_mask = metadata.get("video_context_mask")
    audio_context_mask = metadata.get("audio_context_mask")
    shared_mask = metadata.get("context_mask")
    if video_context_mask is None:
        video_context_mask = shared_mask
    if audio_context_mask is None:
        audio_context_mask = shared_mask

    return {
        "video_context": video_context,
        "audio_context": audio_context,
        "video_context_mask": video_context_mask,
        "audio_context_mask": audio_context_mask,
        "items": int(item_count),
    }



def _nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur



def _signature_like_av_forward(obj: Any) -> bool:
    forward = getattr(obj, "forward", None)
    if forward is None or not callable(forward):
        return False
    try:
        params = list(inspect.signature(forward).parameters.values())
    except Exception:
        return False
    names = [p.name for p in params]
    return len(names) >= 2 and names[0] == "video" and names[1] == "audio"



def _find_av_forward_module(model_or_patcher: Any) -> tuple[Any, str]:
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path in (
        "",
        "diffusion_model",
        "model",
        "model.diffusion_model",
        "model.model",
        "model.model.diffusion_model",
        "inner_model",
    ):
        obj = model_or_patcher if path == "" else _nested_attr(model_or_patcher, path)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        candidates.append((path or "input", obj))

    for path, obj in candidates:
        if _signature_like_av_forward(obj):
            return obj, path

    if hasattr(model_or_patcher, "model") and _signature_like_av_forward(getattr(model_or_patcher, "model")):
        return getattr(model_or_patcher, "model"), "model"

    tried = ", ".join(path for path, _ in candidates) or "<none>"
    raise RuntimeError(
        "Could not locate an AV-capable forward module with signature forward(video, audio, ...). "
        f"Tried: {tried}."
    )



def _module_device_and_dtype(module: Any, fallback_device: torch.device) -> tuple[torch.device, torch.dtype]:
    try:
        first = next(module.parameters())
        device = first.device
        dtype = first.dtype
        if device.type != "meta":
            return device, dtype
    except Exception:
        pass

    load_device = getattr(module, "load_device", None)
    if isinstance(load_device, torch.device):
        return load_device, torch.float32
    return fallback_device, torch.float32



def _build_modality(
    latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
    sigma_value: float,
    *,
    keyframes_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> OfficialModality:
    if not torch.is_tensor(context):
        raise ValueError(f"Conditioning context must be a torch.Tensor, got {type(context).__name__}.")
    batch = int(latent.shape[0])
    sigma = torch.full((batch,), float(sigma_value), device=latent.device, dtype=torch.float32)
    timestep_scale = denoise_mask.squeeze(-1).to(device=latent.device, dtype=torch.float32)
    timesteps = sigma.unsqueeze(1) * timestep_scale
    return OfficialModality(
        latent=latent,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=context_mask,
        attention_mask=attention_mask,
        keyframes_mask=keyframes_mask,
    )



def _call_av_forward(module: Any, video: OfficialModality, audio: OfficialModality, sigma_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    forward = getattr(module, "forward")
    try:
        param_names = list(inspect.signature(forward).parameters.keys())
    except Exception:
        param_names = ["video", "audio", "perturbations"]

    kwargs: dict[str, Any] = {}
    if "sigma" in param_names:
        kwargs["sigma"] = float(sigma_value)
    if "perturbations" in param_names:
        kwargs["perturbations"] = None

    result = module(video=video, audio=audio, **kwargs)
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise RuntimeError(
            "AV forward module returned an unexpected result. Expected (video_tokens, audio_tokens), "
            f"got {type(result).__name__}: {result!r}."
        )
    video_out, audio_out = result
    if not torch.is_tensor(video_out) or not torch.is_tensor(audio_out):
        raise RuntimeError(
            "AV forward module must return two torch.Tensor outputs (video, audio). "
            f"Got types {type(video_out).__name__} and {type(audio_out).__name__}."
        )
    return video_out, audio_out



def _model_looks_like_comfy_patcher(model: Any) -> bool:
    return (
        hasattr(model, "model")
        and hasattr(model, "model_options")
        and hasattr(model, "load_device")
        and hasattr(model, "cleanup")
        and hasattr(model, "pre_run")
    )



def _require_sampling_runtime():
    try:
        import comfy.samplers  # type: ignore
        import comfy.sampler_helpers  # type: ignore
        import comfy.model_patcher  # type: ignore
        import comfy.utils  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime-only dependency inside ComfyUI
        raise RuntimeError(
            "This AV node must run inside a ComfyUI runtime with sampler/model helpers available."
        ) from exc
    return comfy.samplers, comfy.sampler_helpers, comfy.model_patcher, comfy.utils



def _inject_ltxav_runtime_metadata(
    conditioning: Any,
    *,
    frame_rate: float,
    denoise_mask: torch.Tensor,
    latent_shapes: list[tuple[int, ...]],
    keyframe_idxs: torch.Tensor | None,
    generated_keyframes: dict[str, int] | None,
) -> Any:
    if conditioning is None:
        return None
    values = {
        "frame_rate": float(frame_rate),
        "denoise_mask": denoise_mask,
        "latent_shapes": latent_shapes,
    }
    if keyframe_idxs is not None:
        values["keyframe_idxs"] = keyframe_idxs
    if generated_keyframes is not None:
        values["generated_keyframes"] = generated_keyframes

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



def _materialize_native_comfy_av_inputs(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
) -> tuple[DFRComfyModelInput, torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
    samplers, sampler_helpers, model_patcher_mod, comfy_utils = _require_sampling_runtime()
    del samplers, sampler_helpers, model_patcher_mod

    video_model_input = materialize_comfy_model_input(noised_video_state)

    audio_state = get_audio_official_state(noised_audio_state)
    audio_tokens = audio_state["token_state"]
    audio_latent = noised_audio_state["samples"]
    if not torch.is_tensor(audio_latent) or audio_latent.ndim != 4:
        raise ValueError(
            f"Noised audio preview latent must be [B,C,T,F], got {getattr(audio_latent, 'shape', None)}."
        )

    audio_mask = _unpatchify_audio_mask(audio_tokens["denoise_mask"], frames=int(audio_latent.shape[2]))

    video_mask_full = _expand_mask_to_match_latent(video_model_input.denoise_mask, video_model_input.x)
    audio_mask_full = _expand_mask_to_match_latent(audio_mask, audio_latent)

    packed_latent, latent_shapes = comfy_utils.pack_latents([video_model_input.x, audio_latent])
    packed_mask, _ = comfy_utils.pack_latents([video_mask_full, audio_mask_full])

    latent_shapes = [tuple(int(v) for v in shape) for shape in latent_shapes]
    return video_model_input, packed_latent, packed_mask.to(dtype=torch.float32), latent_shapes



def _execute_stage1_av_via_comfy_native_bridge(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    step_index: int,
    cfg_scale: float,
    seed: int,
    av_model_input: DFRStage1AVModelInput,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    samplers, sampler_helpers, model_patcher_mod, comfy_utils = _require_sampling_runtime()

    video_model_input, packed_latent, packed_mask, latent_shapes = _materialize_native_comfy_av_inputs(
        noised_video_state,
        noised_audio_state,
    )

    positive_rt = _inject_ltxav_runtime_metadata(
        positive,
        frame_rate=video_model_input.frame_rate,
        denoise_mask=packed_mask,
        latent_shapes=latent_shapes,
        keyframe_idxs=video_model_input.keyframe_idxs,
        generated_keyframes=video_model_input.generated_keyframes,
    )
    negative_rt = _inject_ltxav_runtime_metadata(
        negative,
        frame_rate=video_model_input.frame_rate,
        denoise_mask=packed_mask,
        latent_shapes=latent_shapes,
        keyframe_idxs=video_model_input.keyframe_idxs,
        generated_keyframes=video_model_input.generated_keyframes,
    ) if negative is not None else None

    positive_items = len(positive_rt) if isinstance(positive_rt, (list, tuple)) else 0
    negative_items = len(negative_rt) if isinstance(negative_rt, (list, tuple)) else 0

    converted_positive = sampler_helpers.convert_cond(positive_rt)
    converted_negative = sampler_helpers.convert_cond(negative_rt) if negative_rt is not None else None
    conds_for_loading = {"positive": converted_positive}
    if converted_negative is not None:
        conds_for_loading["negative"] = converted_negative

    real_model = None
    loaded_models = None
    processed_conds = None
    output = None

    try:
        real_model, _, loaded_models = sampler_helpers.prepare_sampling(
            model,
            packed_latent.shape,
            conds_for_loading,
            model.model_options,
        )
        load_device = getattr(real_model, "load_device", getattr(model, "load_device", packed_latent.device))
        packed_latent = packed_latent.to(device=load_device)
        packed_mask = packed_mask.to(device=load_device, dtype=torch.float32)
        sigma_batch = torch.full(
            (packed_latent.shape[0],),
            float(sigma_schedule[step_index].item()),
            device=load_device,
            dtype=torch.float32,
        )
        sigma_history = sigma_schedule.to(device=load_device)

        processed_conds = samplers.process_conds(
            real_model,
            packed_latent,
            {
                "positive": converted_positive,
                "negative": converted_negative,
            },
            load_device,
            None,
            None,
            seed,
            latent_shapes=latent_shapes,
        )

        extra_model_options = model_patcher_mod.create_model_options_clone(model.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_history
        real_model.latent_shapes = latent_shapes

        model.pre_run()
        output = samplers.sampling_function(
            real_model,
            packed_latent,
            sigma_batch,
            processed_conds.get("negative", None),
            processed_conds["positive"],
            cfg_scale,
            model_options=extra_model_options,
            seed=seed,
        )
        output = real_model.process_latent_out(output.to(torch.float32))
    finally:
        if loaded_models is not None and processed_conds is not None:
            sampler_helpers.cleanup_models(processed_conds, loaded_models)
        try:
            model.cleanup()
        except Exception:
            pass

    if output is None:
        raise RuntimeError("Comfy native AV bridge produced no output tensor.")

    unpacked = comfy_utils.unpack_latents(output, latent_shapes)
    if len(unpacked) < 2:
        raise RuntimeError(
            f"Expected unpacked multimodal output to contain [video, audio], got {len(unpacked)} modalities."
        )
    video_output, audio_output = unpacked[0], unpacked[1]

    if tuple(video_output.shape) != tuple(video_model_input.x.shape):
        raise RuntimeError(
            "Comfy native AV bridge returned an unexpected materialized video latent shape: "
            f"got {tuple(video_output.shape)}, expected {tuple(video_model_input.x.shape)}."
        )
    if tuple(audio_output.shape) != tuple(noised_audio_state["samples"].shape):
        raise RuntimeError(
            "Comfy native AV bridge returned an unexpected audio latent shape: "
            f"got {tuple(audio_output.shape)}, expected {tuple(noised_audio_state['samples'].shape)}."
        )

    video_token_output = _extract_official_tokens_from_materialized_output(video_output, video_model_input)
    audio_tokens = audio_output.permute(0, 2, 1, 3).reshape(audio_output.shape[0], audio_output.shape[2], -1).contiguous()

    if tuple(video_token_output.shape) != tuple(av_model_input.video_latent.shape):
        raise RuntimeError(
            "Recovered official video token output shape does not match the official AV model-input bookkeeping: "
            f"got {tuple(video_token_output.shape)}, expected {tuple(av_model_input.video_latent.shape)}."
        )
    if tuple(audio_tokens.shape) != tuple(av_model_input.audio_latent.shape):
        raise RuntimeError(
            "Recovered official audio token output shape does not match the official AV model-input bookkeeping: "
            f"got {tuple(audio_tokens.shape)}, expected {tuple(av_model_input.audio_latent.shape)}."
        )

    denoised_video_state = _video_latent_with_replaced_token_latent(
        noised_video_state,
        video_token_output.to(dtype=av_model_input.video_latent.dtype),
    )
    denoised_audio_state = _audio_latent_with_replaced_token_latent(
        noised_audio_state,
        audio_tokens.to(dtype=av_model_input.audio_latent.dtype),
    )

    report = (
        f"execution_mode=comfy_native_bridge; sigma={float(sigma_schedule[step_index].item()):.9g}; "
        f"step_index={step_index}; cfg_scale={cfg_scale:.9g}; seed={int(seed)}; seed_source=shared_av_noiser; "
        f"cfg_mode={'single_pass' if abs(cfg_scale - 1.0) <= 1e-12 else 'sampling_function_cfg'}; "
        f"load_device={load_device}; latent_shapes={latent_shapes}; "
        f"packed_latent_shape={tuple(int(x) for x in packed_latent.shape)}; "
        f"packed_mask_shape={tuple(int(x) for x in packed_mask.shape)}; "
        f"materialized_video_shape={tuple(int(x) for x in video_output.shape)}; "
        f"audio_latent_shape={tuple(int(x) for x in audio_output.shape)}; "
        f"official_video_token_shape={tuple(int(x) for x in video_token_output.shape)}; "
        f"official_audio_token_shape={tuple(int(x) for x in audio_tokens.shape)}; "
        f"positive_items={positive_items}; negative_items={negative_items}; "
        f"video_suffix_pre_filter={int(video_model_input.appended_tokens_before_filter)}; "
        f"video_suffix_post_filter={int(video_model_input.appended_tokens_after_filter)}."
    )
    return denoised_video_state, denoised_audio_state, report



def _execute_stage1_av_via_direct_forward(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    step_index: int,
    cfg_scale: float,
    seed: int,
    av_model_input: DFRStage1AVModelInput,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    sigma_value = float(sigma_schedule[step_index].item())
    pos = _resolve_context_fields(positive, "positive")
    neg = _resolve_context_fields(negative, "negative") if negative is not None else None

    cleanup_needed = False
    if hasattr(model, "pre_run") and callable(model.pre_run):
        model.pre_run()
        cleanup_needed = hasattr(model, "cleanup") and callable(model.cleanup)

    try:
        forward_module, forward_path = _find_av_forward_module(model)
        default_device = av_model_input.video_latent.device
        execution_device, model_dtype = _module_device_and_dtype(forward_module, default_device)

        def _cast_optional(t: torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            if t is None:
                return None
            return t.to(device=execution_device, dtype=dtype or t.dtype)

        positive_video = _build_modality(
            latent=av_model_input.video_latent.to(device=execution_device, dtype=model_dtype),
            denoise_mask=av_model_input.video_denoise_mask.to(device=execution_device, dtype=torch.float32),
            positions=av_model_input.video_positions.to(device=execution_device, dtype=torch.float32),
            context=pos["video_context"].to(device=execution_device, dtype=model_dtype),
            context_mask=_cast_optional(pos["video_context_mask"], dtype=torch.float32),
            sigma_value=sigma_value,
            keyframes_mask=_cast_optional(av_model_input.video_keyframes_mask, dtype=torch.float32),
        )
        positive_audio = _build_modality(
            latent=av_model_input.audio_latent.to(device=execution_device, dtype=model_dtype),
            denoise_mask=av_model_input.audio_denoise_mask.to(device=execution_device, dtype=torch.float32),
            positions=av_model_input.audio_positions.to(device=execution_device, dtype=torch.float32),
            context=pos["audio_context"].to(device=execution_device, dtype=model_dtype),
            context_mask=_cast_optional(pos["audio_context_mask"], dtype=torch.float32),
            sigma_value=sigma_value,
        )

        pos_video_out, pos_audio_out = _call_av_forward(forward_module, positive_video, positive_audio, sigma_value)

        if neg is not None and abs(cfg_scale - 1.0) > 1e-12:
            negative_video = _build_modality(
                latent=positive_video.latent,
                denoise_mask=av_model_input.video_denoise_mask.to(device=execution_device, dtype=torch.float32),
                positions=positive_video.positions,
                context=neg["video_context"].to(device=execution_device, dtype=model_dtype),
                context_mask=_cast_optional(neg["video_context_mask"], dtype=torch.float32),
                sigma_value=sigma_value,
                keyframes_mask=_cast_optional(av_model_input.video_keyframes_mask, dtype=torch.float32),
            )
            negative_audio = _build_modality(
                latent=positive_audio.latent,
                denoise_mask=av_model_input.audio_denoise_mask.to(device=execution_device, dtype=torch.float32),
                positions=positive_audio.positions,
                context=neg["audio_context"].to(device=execution_device, dtype=model_dtype),
                context_mask=_cast_optional(neg["audio_context_mask"], dtype=torch.float32),
                sigma_value=sigma_value,
            )
            neg_video_out, neg_audio_out = _call_av_forward(forward_module, negative_video, negative_audio, sigma_value)
            video_out = neg_video_out + (pos_video_out - neg_video_out) * cfg_scale
            audio_out = neg_audio_out + (pos_audio_out - neg_audio_out) * cfg_scale
            cfg_mode = "explicit_two_pass"
        else:
            video_out = pos_video_out
            audio_out = pos_audio_out
            cfg_mode = "single_pass"

        if tuple(video_out.shape) != tuple(av_model_input.video_latent.shape):
            raise RuntimeError(
                "AV forward returned an unexpected video token shape: "
                f"got {tuple(video_out.shape)}, expected {tuple(av_model_input.video_latent.shape)}."
            )
        if tuple(audio_out.shape) != tuple(av_model_input.audio_latent.shape):
            raise RuntimeError(
                "AV forward returned an unexpected audio token shape: "
                f"got {tuple(audio_out.shape)}, expected {tuple(av_model_input.audio_latent.shape)}."
            )

        denoised_video_state = _video_latent_with_replaced_token_latent(
            noised_video_state,
            video_out.to(dtype=av_model_input.video_latent.dtype),
        )
        denoised_audio_state = _audio_latent_with_replaced_token_latent(
            noised_audio_state,
            audio_out.to(dtype=av_model_input.audio_latent.dtype),
        )
    finally:
        if cleanup_needed:
            try:
                model.cleanup()
            except Exception:
                pass

    report = (
        f"execution_mode=direct_forward_fallback; sigma={sigma_value:.9g}; step_index={step_index}; "
        f"cfg_scale={cfg_scale:.9g}; seed={int(seed)}; seed_source=shared_av_noiser; cfg_mode={cfg_mode}; forward_module={type(forward_module).__name__}; "
        f"forward_path={forward_path}; execution_device={execution_device}; model_dtype={model_dtype}; "
        f"positive_items={pos['items']}; negative_items={0 if neg is None else neg['items']}; "
        f"video_token_shape={tuple(int(x) for x in video_out.shape)}; "
        f"audio_token_shape={tuple(int(x) for x in audio_out.shape)}; "
        f"video_mask_shape={tuple(int(x) for x in av_model_input.video_denoise_mask.shape)}; "
        f"audio_mask_shape={tuple(int(x) for x in av_model_input.audio_denoise_mask.shape)}; "
        f"joint_mask_shape={tuple(int(x) for x in av_model_input.joint_denoise_mask.shape)}."
    )
    return denoised_video_state, denoised_audio_state, report



def resolve_shared_av_seed(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
) -> int:
    """Recover the single authoritative Stage-1 seed from the shared AV noiser metadata."""
    v_state = clone_or_create_official_state(noised_video_state)
    v_tokens = ensure_token_state(
        v_state,
        target_samples=noised_video_state["samples"],
        fps=float(v_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in v_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(v_state.get("causal_fix", True)),
    )
    a_state = get_audio_official_state(noised_audio_state)
    a_tokens = a_state["token_state"]
    vm = v_tokens.get(NOISER_METADATA_KEY) or {}
    am = a_tokens.get(NOISER_METADATA_KEY) or {}

    if vm.get("shared_av_rng") is not True or am.get("shared_av_rng") is not True:
        raise ValueError(
            "Stage-1 AV denoise requires states produced by 'LTX Official: Stage 1 AV Gaussian Noiser' so the "
            "single authoritative seed can be recovered from shared-RNG metadata."
        )
    if vm.get("rng_order") != "video_then_audio" or am.get("rng_order") != "video_then_audio":
        raise ValueError("Shared AV noiser metadata does not declare the required video_then_audio RNG order.")
    if "seed" not in vm or "seed" not in am:
        raise ValueError("Shared AV noiser metadata is missing the Stage-1 seed.")
    v_seed = int(vm["seed"])
    a_seed = int(am["seed"])
    if v_seed != a_seed:
        raise ValueError(f"Video/audio noiser seed mismatch: video={v_seed}, audio={a_seed}.")
    return v_seed


def execute_stage1_av_denoise(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
    cfg_scale: float = 1.0,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], DFRStage1AVModelInput, str]:
    """Run one real joint AV transformer/x0 evaluation.

    The strict official VIDEO/AUDIO states remain authoritative; the real model is
    used only as the execution backend.
    """
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    step_index = int(step_index)
    if step_index < 0 or step_index >= sigma_schedule.numel() - 1:
        raise ValueError(f"step_index must be in [0, {sigma_schedule.numel() - 2}], got {step_index}.")
    cfg_scale = float(cfg_scale)
    shared_seed = resolve_shared_av_seed(noised_video_state, noised_audio_state)
    if seed is not None and int(seed) != shared_seed:
        raise ValueError(
            f"Explicit execution seed {int(seed)} does not match authoritative shared AV noiser seed {shared_seed}. "
            "The parity path must use one Stage-1 seed."
        )
    seed = shared_seed

    av_model_input = materialize_stage1_av_model_input(noised_video_state, noised_audio_state)

    if _model_looks_like_comfy_patcher(model):
        denoised_video_state, denoised_audio_state, report = _execute_stage1_av_via_comfy_native_bridge(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_video_state,
            noised_audio_state=noised_audio_state,
            sigma_schedule=sigma_schedule,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=seed,
            av_model_input=av_model_input,
        )
    else:
        denoised_video_state, denoised_audio_state, report = _execute_stage1_av_via_direct_forward(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_video_state,
            noised_audio_state=noised_audio_state,
            sigma_schedule=sigma_schedule,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=seed,
            av_model_input=av_model_input,
        )

    return denoised_video_state, denoised_audio_state, av_model_input, report



def validate_stage1_av_denoise(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    raw_denoised_video_state: dict[str, Any],
    raw_denoised_audio_state: dict[str, Any],
) -> dict[str, Any]:
    """Structural validator for the first real Stage-1 AV transformer step."""
    v_in = clone_or_create_official_state(noised_video_state)
    v_out = clone_or_create_official_state(raw_denoised_video_state)
    a_in = get_audio_official_state(noised_audio_state)
    a_out = get_audio_official_state(raw_denoised_audio_state)

    v_in_tokens = ensure_token_state(
        v_in,
        target_samples=noised_video_state["samples"],
        fps=float(v_in.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in v_in.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(v_in.get("causal_fix", True)),
    )
    v_out_tokens = ensure_token_state(
        v_out,
        target_samples=raw_denoised_video_state["samples"],
        fps=float(v_out.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in v_out.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(v_out.get("causal_fix", True)),
    )
    a_in_tokens = a_in["token_state"]
    a_out_tokens = a_out["token_state"]

    video_shape_error = 0.0 if tuple(v_in_tokens["latent"].shape) == tuple(v_out_tokens["latent"].shape) else 1.0
    audio_shape_error = 0.0 if tuple(a_in_tokens["latent"].shape) == tuple(a_out_tokens["latent"].shape) else 1.0

    video_preview = _unpatchify_video_base_tokens(v_out_tokens["latent"], tuple(v_out["base_shape"]))
    audio_preview = _unpatchify_audio(
        a_out_tokens["latent"],
        channels=int(a_out_tokens["channels"]),
        mel_bins=int(a_out_tokens["mel_bins"]),
    )
    video_preview_error = float(
        (video_preview.to(torch.float32) - raw_denoised_video_state["samples"].to(video_preview.device, dtype=torch.float32))
        .abs()
        .max()
        .item()
    )
    audio_preview_error = float(
        (audio_preview.to(torch.float32) - raw_denoised_audio_state["samples"].to(audio_preview.device, dtype=torch.float32))
        .abs()
        .max()
        .item()
    )

    video_finite_error = 0.0 if torch.isfinite(v_out_tokens["latent"]).all().item() else 1.0
    audio_finite_error = 0.0 if torch.isfinite(a_out_tokens["latent"]).all().item() else 1.0
    video_delta = float((v_out_tokens["latent"].to(torch.float32) - v_in_tokens["latent"].to(v_out_tokens["latent"].device, dtype=torch.float32)).abs().max().item())
    audio_delta = float((a_out_tokens["latent"].to(torch.float32) - a_in_tokens["latent"].to(a_out_tokens["latent"].device, dtype=torch.float32)).abs().max().item())

    passed = (
        video_shape_error == 0.0
        and audio_shape_error == 0.0
        and video_preview_error <= 1e-6
        and audio_preview_error <= 1e-6
        and video_finite_error == 0.0
        and audio_finite_error == 0.0
    )

    return {
        "passed": bool(passed),
        "video_shape_error": float(video_shape_error),
        "audio_shape_error": float(audio_shape_error),
        "video_preview_error": float(video_preview_error),
        "audio_preview_error": float(audio_preview_error),
        "video_finite_error": float(video_finite_error),
        "audio_finite_error": float(audio_finite_error),
        "video_delta": float(video_delta),
        "audio_delta": float(audio_delta),
        "video_token_shape": tuple(int(x) for x in v_out_tokens["latent"].shape),
        "audio_token_shape": tuple(int(x) for x in a_out_tokens["latent"].shape),
        "video_device": str(v_out_tokens["latent"].device),
        "audio_device": str(a_out_tokens["latent"].device),
    }


def _stage1_av_token_euler_update(
    current: torch.Tensor,
    raw_denoised: torch.Tensor,
    denoise_mask: torch.Tensor,
    clean_latent: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """Apply official post-process + Euler math without materializing LATENT wrappers."""
    if tuple(raw_denoised.shape) != tuple(current.shape):
        raise RuntimeError(
            "Stage-1 AV denoised token shape changed during the trajectory: "
            f"got {tuple(raw_denoised.shape)}, expected {tuple(current.shape)}."
        )

    execution_device = raw_denoised.device
    state_dtype = current.dtype
    raw_state_dtype = raw_denoised.to(device=execution_device, dtype=state_dtype)
    mask_f32 = denoise_mask.to(device=execution_device, dtype=torch.float32)
    clean_f32 = clean_latent.to(device=execution_device, dtype=torch.float32)
    post_processed = (
        raw_state_dtype.to(torch.float32) * mask_f32
        + clean_f32 * (1.0 - mask_f32)
    ).to(dtype=state_dtype)

    if sigma_next == 0.0:
        next_latent = post_processed
    else:
        if sigma == 0.0:
            raise ValueError("Current sigma is zero before the terminal step; cannot perform an Euler update.")
        ratio = sigma_next / sigma
        next_latent = (
            ratio * current.to(device=execution_device, dtype=torch.float32)
            + (1.0 - ratio) * post_processed.to(torch.float32)
        ).to(dtype=state_dtype)
    return next_latent.contiguous()


def _working_stage1_av_states(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Clone trajectory metadata once; only the two token latents change in the loop."""
    video_meta = clone_or_create_official_state(noised_video_state)
    video_tokens = ensure_token_state(
        video_meta,
        target_samples=noised_video_state["samples"],
        fps=float(video_meta.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in video_meta.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(video_meta.get("causal_fix", True)),
    )
    audio_meta = clone_audio_official_state(noised_audio_state)
    audio_tokens = audio_meta["token_state"]

    working_video = noised_video_state.copy()
    working_video[OFFICIAL_STATE_KEY] = video_meta
    working_audio = noised_audio_state.copy()
    working_audio[AUDIO_OFFICIAL_STATE_KEY] = audio_meta
    return working_video, working_audio, video_tokens, audio_tokens


def _set_working_stage1_av_latents(
    working_video: dict[str, Any],
    working_audio: dict[str, Any],
    video_tokens: dict[str, Any],
    audio_tokens: dict[str, Any],
    current_video: torch.Tensor,
    current_audio: torch.Tensor,
) -> None:
    video_tokens["latent"] = current_video
    audio_tokens["latent"] = current_audio
    # The video materializer reads the authoritative token state directly.  The
    # native AV packer reads the audio preview, so refresh only that compact view.
    working_audio["samples"] = _unpatchify_audio(
        current_audio,
        channels=int(audio_tokens["channels"]),
        mel_bins=int(audio_tokens["mel_bins"]),
    )


def _execute_stage1_av_whole_schedule_comfy(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    seed: int,
    collect_diagnostics: bool,
    profiler: Stage1RuntimeProfiler | None,
    memory_profile: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    """Run all Stage-1 AV transitions inside one Comfy sampling lifecycle."""
    import time

    samplers, sampler_helpers, model_patcher_mod, comfy_utils = _require_sampling_runtime()
    with profile_section(profiler, "initial bridge + conditioning"):
        av_model_input = materialize_stage1_av_model_input(noised_video_state, noised_audio_state)
        working_video, working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
            noised_video_state,
            noised_audio_state,
        )

        reference_video_input, initial_packed, packed_mask, latent_shapes = _materialize_native_comfy_av_inputs(
            working_video,
            working_audio,
        )
        positive_rt = _inject_ltxav_runtime_metadata(
            positive,
            frame_rate=reference_video_input.frame_rate,
            denoise_mask=packed_mask,
            latent_shapes=latent_shapes,
            keyframe_idxs=reference_video_input.keyframe_idxs,
            generated_keyframes=reference_video_input.generated_keyframes,
        )
        negative_rt = _inject_ltxav_runtime_metadata(
            negative,
            frame_rate=reference_video_input.frame_rate,
            denoise_mask=packed_mask,
            latent_shapes=latent_shapes,
            keyframe_idxs=reference_video_input.keyframe_idxs,
            generated_keyframes=reference_video_input.generated_keyframes,
        ) if negative is not None else None

        converted_positive = sampler_helpers.convert_cond(positive_rt)
        converted_negative = sampler_helpers.convert_cond(negative_rt) if negative_rt is not None else None
        conds_for_loading = {"positive": converted_positive}
        if converted_negative is not None:
            conds_for_loading["negative"] = converted_negative

        current_video = av_model_input.video_latent
        current_audio = av_model_input.audio_latent
        video_mask = av_model_input.video_denoise_mask
        audio_mask = av_model_input.audio_denoise_mask
        video_clean = video_tokens["clean_latent"]
        audio_clean = audio_tokens["clean_latent"]
        # The native backend does not consume the direct bridge's position copies.
        del av_model_input
        initial_packed_shape = tuple(initial_packed.shape)
        # Only shape/operation metadata is needed after initial packing. Release
        # the initial dense video payload while retaining its geometry contract.
        reference_video_input.x = torch.empty(reference_video_input.x.shape, device="meta")
        reference_video_input.denoise_mask = torch.empty(reference_video_input.denoise_mask.shape, device="meta")
        _set_working_stage1_av_latents(
            working_video, working_audio, video_tokens, audio_tokens, current_video, current_audio,
        )
    loaded_models = None
    processed_conds = None
    real_model = None
    step_reports: list[str] = []

    memory = SamplerMemoryProfile(memory_profile, getattr(model, "load_device", initial_packed.device), model)
    memory.snapshot("before model load")

    try:
        with profile_section(profiler, "model + conditioning preparation"):
            real_model, _, loaded_models = sampler_helpers.prepare_sampling(
                model,
                initial_packed.shape,
                conds_for_loading,
                model.model_options,
            )
            load_device = getattr(real_model, "load_device", getattr(model, "load_device", initial_packed.device))
            initial_packed = initial_packed.to(device=load_device)
            packed_mask = packed_mask.to(device=load_device, dtype=torch.float32)
            processed_conds = samplers.process_conds(
                real_model,
                initial_packed,
                {"positive": converted_positive, "negative": converted_negative},
                load_device,
                None,
                None,
                seed,
                latent_shapes=latent_shapes,
            )

            runtime_model_options = model_patcher_mod.create_model_options_clone(model.model_options)
            from .stage2_temporal_tiling import install_temporal_model_wrapper
            install_temporal_model_wrapper(runtime_model_options, reference_video_input, comfy_utils, noised_video_state)
            runtime_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_schedule.to(
                device=load_device
            )
            real_model.latent_shapes = latent_shapes
            model.pre_run()

        del initial_packed, packed_mask
        memory.snapshot("after model load")

        try:
            progress = comfy_utils.ProgressBar(int(sigma_schedule.numel() - 1))
        except Exception:  # pragma: no cover - cosmetic only
            progress = None

        for step_index in range(int(sigma_schedule.numel() - 1)):
            memory.start_step()
            step_started = time.perf_counter() if collect_diagnostics else None
            with profile_section(profiler, "AV materialization + packing"):
                _set_working_stage1_av_latents(
                    working_video,
                    working_audio,
                    video_tokens,
                    audio_tokens,
                    current_video,
                    current_audio,
                )
                video_payload = materialize_video_tokens_with_layout(current_video, reference_video_input)
                packed_latent, current_shapes = comfy_utils.pack_latents([video_payload, working_audio["samples"]])
                current_shapes = [tuple(int(v) for v in shape) for shape in current_shapes]
                del video_payload
                if current_shapes != latent_shapes:
                    raise RuntimeError(
                        f"Packed AV latent shapes changed during Stage 1: {latent_shapes} -> {current_shapes}."
                    )

                packed_latent = packed_latent.to(device=load_device)
                sigma_value = float(sigma_schedule[step_index].item())
                sigma_next = float(sigma_schedule[step_index + 1].item())
                sigma_batch = torch.full(
                    (packed_latent.shape[0],),
                    sigma_value,
                    device=load_device,
                    dtype=torch.float32,
                )
            with profile_section(profiler, "transformer forward"):
                packed_denoised = samplers.sampling_function(
                    real_model,
                    packed_latent,
                    sigma_batch,
                    processed_conds.get("negative", None),
                    processed_conds["positive"],
                    cfg_scale,
                    model_options=runtime_model_options,
                    seed=seed,
                )
                # Keep the established AV bridge semantics: each x0 prediction is
                # converted back to latent space before official post-process/Euler.
                packed_denoised = real_model.process_latent_out(packed_denoised.to(torch.float32))
            with profile_section(profiler, "unpack + Euler update"):
                unpacked = comfy_utils.unpack_latents(packed_denoised, latent_shapes)
                if len(unpacked) < 2:
                    raise RuntimeError(
                        f"Expected unpacked multimodal output to contain [video, audio], got {len(unpacked)} modalities."
                    )
                video_output, audio_output = unpacked[0], unpacked[1]
                raw_video = _extract_official_tokens_from_materialized_output(video_output, reference_video_input)
                raw_audio = audio_output.permute(0, 2, 1, 3).reshape(
                    audio_output.shape[0], audio_output.shape[2], -1
                ).contiguous()
                current_video = _stage1_av_token_euler_update(
                    current_video,
                    raw_video,
                    video_mask,
                    video_clean,
                    sigma=sigma_value,
                    sigma_next=sigma_next,
                )
                current_audio = _stage1_av_token_euler_update(
                    current_audio,
                    raw_audio,
                    audio_mask,
                    audio_clean,
                    sigma=sigma_value,
                    sigma_next=sigma_next,
                )
                # Refresh the working references immediately so the preceding
                # trajectory tensors cannot survive into the next forward call.
                _set_working_stage1_av_latents(
                    working_video, working_audio, video_tokens, audio_tokens, current_video, current_audio,
                )
                del packed_latent, packed_denoised, unpacked, video_output, audio_output, raw_video, raw_audio
            memory.end_step(step_index)
            if collect_diagnostics:
                assert step_started is not None
                step_reports.append(
                    f"{step_index}:{sigma_value:.9g}->{sigma_next:.9g}@{time.perf_counter()-step_started:.2f}s"
                )
            if progress is not None:
                try:
                    progress.update(1)
                except Exception:
                    pass
    finally:
        with profile_section(profiler, "model cleanup"):
            cleanup_conds = processed_conds if processed_conds is not None else conds_for_loading
            if loaded_models is not None:
                try:
                    sampler_helpers.cleanup_models(cleanup_conds, loaded_models)
                except Exception:
                    pass
            try:
                model.cleanup()
            except Exception:
                pass

    with profile_section(profiler, "final extraction + handoff"):
        final_video = _video_latent_with_replaced_token_latent(noised_video_state, current_video)
        final_audio = _audio_latent_with_replaced_token_latent(noised_audio_state, current_audio)
    details = ""
    if collect_diagnostics:
        details = (
            f"execution_mode=whole_schedule_comfy_native_bridge; prepare_sampling_calls=1; "
            f"process_conds_calls=1; pre_run_calls=1; cleanup_calls=1; "
            f"packed_shape={initial_packed_shape}; latent_shapes={latent_shapes}"
        )
    return final_video, final_audio, step_reports, details


def _execute_stage1_av_whole_schedule_direct(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    seed: int,
    collect_diagnostics: bool,
    profiler: Stage1RuntimeProfiler | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    """Pure-torch whole-schedule fallback with one model lifecycle."""
    import time

    with profile_section(profiler, "initial bridge + conditioning"):
        av_input = materialize_stage1_av_model_input(noised_video_state, noised_audio_state)
        _working_video, _working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
            noised_video_state,
            noised_audio_state,
        )
        pos = _resolve_context_fields(positive, "positive")
        neg = _resolve_context_fields(negative, "negative") if negative is not None else None
        current_video = av_input.video_latent
        current_audio = av_input.audio_latent
    step_reports: list[str] = []
    pre_run_called = False
    cleanup_needed = False
    model_calls = 0

    with profile_section(profiler, "model + conditioning preparation"):
        if hasattr(model, "pre_run") and callable(model.pre_run):
            model.pre_run()
            pre_run_called = True
            cleanup_needed = hasattr(model, "cleanup") and callable(model.cleanup)

    try:
        forward_module, forward_path = _find_av_forward_module(model)
        execution_device, model_dtype = _module_device_and_dtype(forward_module, current_video.device)

        def cast_optional(tensor: torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device=execution_device, dtype=dtype or tensor.dtype)

        video_mask = av_input.video_denoise_mask.to(device=execution_device, dtype=torch.float32)
        audio_mask = av_input.audio_denoise_mask.to(device=execution_device, dtype=torch.float32)
        video_positions = av_input.video_positions.to(device=execution_device, dtype=torch.float32)
        audio_positions = av_input.audio_positions.to(device=execution_device, dtype=torch.float32)
        video_keyframes_mask = cast_optional(av_input.video_keyframes_mask, dtype=torch.float32)
        video_clean = video_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        audio_clean = audio_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        positive_video_context = pos["video_context"].to(device=execution_device, dtype=model_dtype)
        positive_audio_context = pos["audio_context"].to(device=execution_device, dtype=model_dtype)
        positive_video_mask = cast_optional(pos["video_context_mask"], dtype=torch.float32)
        positive_audio_mask = cast_optional(pos["audio_context_mask"], dtype=torch.float32)
        if neg is not None and abs(cfg_scale - 1.0) > 1e-12:
            negative_video_context = neg["video_context"].to(device=execution_device, dtype=model_dtype)
            negative_audio_context = neg["audio_context"].to(device=execution_device, dtype=model_dtype)
            negative_video_mask = cast_optional(neg["video_context_mask"], dtype=torch.float32)
            negative_audio_mask = cast_optional(neg["audio_context_mask"], dtype=torch.float32)
            cfg_mode = "explicit_two_pass"
        else:
            negative_video_context = negative_audio_context = None
            negative_video_mask = negative_audio_mask = None
            cfg_mode = "single_pass"

        for step_index in range(int(sigma_schedule.numel() - 1)):
            step_started = time.perf_counter() if collect_diagnostics else None
            sigma_value = float(sigma_schedule[step_index].item())
            sigma_next = float(sigma_schedule[step_index + 1].item())
            with profile_section(profiler, "AV materialization + packing"):
                positive_video = _build_modality(
                    current_video.to(device=execution_device, dtype=model_dtype),
                    video_mask,
                    video_positions,
                    positive_video_context,
                    positive_video_mask,
                    sigma_value,
                    keyframes_mask=video_keyframes_mask,
                )
                positive_audio = _build_modality(
                    current_audio.to(device=execution_device, dtype=model_dtype),
                    audio_mask,
                    audio_positions,
                    positive_audio_context,
                    positive_audio_mask,
                    sigma_value,
                )
            with profile_section(profiler, "transformer forward"):
                pos_video_out, pos_audio_out = _call_av_forward(
                    forward_module, positive_video, positive_audio, sigma_value
                )
                model_calls += 1

                if negative_video_context is not None and negative_audio_context is not None:
                    negative_video = _build_modality(
                        positive_video.latent,
                        video_mask,
                        video_positions,
                        negative_video_context,
                        negative_video_mask,
                        sigma_value,
                        keyframes_mask=video_keyframes_mask,
                    )
                    negative_audio = _build_modality(
                        positive_audio.latent,
                        audio_mask,
                        audio_positions,
                        negative_audio_context,
                        negative_audio_mask,
                        sigma_value,
                    )
                    neg_video_out, neg_audio_out = _call_av_forward(
                        forward_module, negative_video, negative_audio, sigma_value
                    )
                    model_calls += 1
                    raw_video = neg_video_out + (pos_video_out - neg_video_out) * cfg_scale
                    raw_audio = neg_audio_out + (pos_audio_out - neg_audio_out) * cfg_scale
                else:
                    raw_video = pos_video_out
                    raw_audio = pos_audio_out

            with profile_section(profiler, "unpack + Euler update"):
                current_video = _stage1_av_token_euler_update(
                    current_video,
                    raw_video,
                    video_mask,
                    video_clean,
                    sigma=sigma_value,
                    sigma_next=sigma_next,
                )
                current_audio = _stage1_av_token_euler_update(
                    current_audio,
                    raw_audio,
                    audio_mask,
                    audio_clean,
                    sigma=sigma_value,
                    sigma_next=sigma_next,
                )
            if collect_diagnostics:
                assert step_started is not None
                step_reports.append(
                    f"{step_index}:{sigma_value:.9g}->{sigma_next:.9g}@{time.perf_counter()-step_started:.2f}s"
                )
    finally:
        with profile_section(profiler, "model cleanup"):
            if cleanup_needed:
                try:
                    model.cleanup()
                except Exception:
                    pass

    with profile_section(profiler, "final extraction + handoff"):
        final_video = _video_latent_with_replaced_token_latent(noised_video_state, current_video)
        final_audio = _audio_latent_with_replaced_token_latent(noised_audio_state, current_audio)
    details = ""
    if collect_diagnostics:
        details = (
            f"execution_mode=whole_schedule_direct_forward; pre_run_calls={1 if pre_run_called else 0}; "
            f"cleanup_calls={1 if cleanup_needed else 0}; model_calls={model_calls}; cfg_mode={cfg_mode}; "
            f"forward_module={type(forward_module).__name__}; forward_path={forward_path}; "
            f"execution_device={execution_device}; model_dtype={model_dtype}; seed={seed}"
        )
    return final_video, final_audio, step_reports, details


def execute_stage1_av_denoising_loop(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    *,
    require_official_sigmas: bool = True,
    collect_diagnostics: bool = True,
    profiler: Stage1RuntimeProfiler | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    """Run the complete official deterministic Stage-1 AV denoising trajectory.

    Model/conditioning preparation and cleanup span the whole sigma schedule.  The
    validated single-step bridge remains available as an atomic regression oracle,
    but the production loop no longer reloads and cleans up the model eight times.
    """
    import time

    sigma_schedule = (
        _assert_stage1_sigma_schedule(sigmas)
        if require_official_sigmas
        else _validate_custom_stage1_sigma_schedule(sigmas)
    )
    total_started = time.perf_counter()
    shared_seed = resolve_shared_av_seed(noised_video_state, noised_audio_state)
    cfg_scale = float(cfg_scale)

    if _model_looks_like_comfy_patcher(model):
        current_video_state, current_audio_state, step_reports, backend_report = (
            _execute_stage1_av_whole_schedule_comfy(
                model=model,
                positive=positive,
                negative=negative,
                noised_video_state=noised_video_state,
                noised_audio_state=noised_audio_state,
                sigma_schedule=sigma_schedule,
                cfg_scale=cfg_scale,
                seed=shared_seed,
                collect_diagnostics=collect_diagnostics,
                profiler=profiler,
            )
        )
    else:
        current_video_state, current_audio_state, step_reports, backend_report = (
            _execute_stage1_av_whole_schedule_direct(
                model=model,
                positive=positive,
                negative=negative,
                noised_video_state=noised_video_state,
                noised_audio_state=noised_audio_state,
                sigma_schedule=sigma_schedule,
                cfg_scale=cfg_scale,
                seed=shared_seed,
                collect_diagnostics=collect_diagnostics,
                profiler=profiler,
            )
        )

    with profile_section(profiler, "final extraction + handoff"):
        base_latent = extract_official_base_latent(current_video_state)
        generated_keyframes = extract_official_generated_keyframes(current_video_state)

    resolved_final_seed = resolve_shared_av_seed(current_video_state, current_audio_state)
    if resolved_final_seed != shared_seed:
        raise RuntimeError(
            f"Stage-1 AV seed metadata changed during the trajectory: {shared_seed} -> {resolved_final_seed}."
        )

    if not collect_diagnostics:
        return current_video_state, current_audio_state, base_latent, generated_keyframes, ""

    final_video_state = clone_or_create_official_state(current_video_state)
    final_video_tokens = ensure_token_state(
        final_video_state,
        target_samples=current_video_state["samples"],
        fps=float(final_video_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in final_video_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(final_video_state.get("causal_fix", True)),
    )
    final_audio_state = get_audio_official_state(current_audio_state)
    final_audio_tokens = final_audio_state["token_state"]
    total_elapsed = time.perf_counter() - total_started
    layout = final_video_tokens.get("generated_keyframe_layout") or {}

    report = (
        f"PASS=True; stage=1_av; steps={int(sigma_schedule.numel()-1)}; cfg_scale={float(cfg_scale):.9g}; "
        f"seed={int(shared_seed)}; video_device={final_video_tokens['latent'].device}; "
        f"audio_device={final_audio_tokens['latent'].device}; "
        f"video_token_shape={tuple(int(x) for x in final_video_tokens['latent'].shape)}; "
        f"audio_token_shape={tuple(int(x) for x in final_audio_tokens['latent'].shape)}; "
        f"base_latent_shape={tuple(int(x) for x in base_latent['samples'].shape)}; "
        f"generated_keyframes_shape={tuple(int(x) for x in generated_keyframes['samples'].shape)}; "
        f"num_generated_keyframes={int(layout.get('num_keyframes', 0))}; "
        f"duration_seconds={float(final_audio_state.get('duration_seconds', 0.0)):.9g}; total_time={total_elapsed:.2f}s; "
        f"backend=({backend_report}); "
        f"trajectory=[{' | '.join(step_reports)}]."
    )
    return current_video_state, current_audio_state, base_latent, generated_keyframes, report


def validate_stage1_av_loop_outputs(
    final_video_state: dict[str, Any],
    final_audio_state: dict[str, Any],
    stage_1_base_latent: dict[str, torch.Tensor],
    generated_keyframes: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate the AV Stage-1 loop exports and final audio preview consistency."""
    expected_base = extract_official_base_latent(final_video_state)["samples"]
    actual_base = stage_1_base_latent.get("samples")
    if actual_base is None:
        raise ValueError("stage_1_base_latent must contain a 'samples' tensor.")
    if tuple(expected_base.shape) != tuple(actual_base.shape):
        base_max_abs_error = 1.0
    elif expected_base.numel() == 0:
        base_max_abs_error = 0.0
    else:
        base_max_abs_error = float(
            (expected_base.to(device=actual_base.device, dtype=torch.float32) - actual_base.to(dtype=torch.float32))
            .abs()
            .max()
            .item()
        )

    expected_kf = extract_official_generated_keyframes(final_video_state)["samples"]
    actual_kf = generated_keyframes.get("samples")
    if actual_kf is None:
        raise ValueError("generated_keyframes must contain a 'samples' tensor.")
    if tuple(expected_kf.shape) != tuple(actual_kf.shape):
        generated_keyframes_max_abs_error = 1.0
    elif expected_kf.numel() == 0:
        generated_keyframes_max_abs_error = 0.0
    else:
        generated_keyframes_max_abs_error = float(
            (expected_kf.to(device=actual_kf.device, dtype=torch.float32) - actual_kf.to(dtype=torch.float32))
            .abs()
            .max()
            .item()
        )

    audio_state = get_audio_official_state(final_audio_state)
    audio_tokens = audio_state["token_state"]
    rebuilt_audio = _unpatchify_audio(
        audio_tokens["latent"],
        channels=int(audio_tokens["channels"]),
        mel_bins=int(audio_tokens["mel_bins"]),
    )
    actual_audio = final_audio_state.get("samples")
    if actual_audio is None or tuple(rebuilt_audio.shape) != tuple(actual_audio.shape):
        audio_preview_error = 1.0
    elif rebuilt_audio.numel() == 0:
        audio_preview_error = 0.0
    else:
        audio_preview_error = float(
            (rebuilt_audio.to(device=actual_audio.device, dtype=torch.float32) - actual_audio.to(dtype=torch.float32))
            .abs()
            .max()
            .item()
        )

    metadata_error = 0.0
    try:
        shared_seed = resolve_shared_av_seed(final_video_state, final_audio_state)
    except Exception:
        shared_seed = None
        metadata_error = 1.0

    passed = (
        base_max_abs_error == 0.0
        and generated_keyframes_max_abs_error == 0.0
        and audio_preview_error <= 1e-6
        and metadata_error == 0.0
    )

    return {
        "passed": bool(passed),
        "base_max_abs_error": float(base_max_abs_error),
        "generated_keyframes_max_abs_error": float(generated_keyframes_max_abs_error),
        "audio_preview_error": float(audio_preview_error),
        "metadata_error": float(metadata_error),
        "shared_seed": shared_seed,
        "video_base_shape": tuple(int(x) for x in expected_base.shape),
        "generated_keyframes_shape": tuple(int(x) for x in expected_kf.shape),
        "audio_shape": tuple(int(x) for x in rebuilt_audio.shape),
        "audio_token_shape": tuple(int(x) for x in audio_tokens["latent"].shape),
        "duration_seconds": float(audio_state.get("duration_seconds", 0.0)),
    }
