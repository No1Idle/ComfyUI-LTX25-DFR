"""One complete official Euler-Ancestral Temporal-DFR refinement round.

This keeps the official round structure: x2 latent upsample, seam windows, local
conditioning, continued Gaussian RNG, frozen audio, four eta=0.5 ancestral
transitions, stitching, and carry-forward keyframes.
"""

from __future__ import annotations

from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.av_execution import (
        _audio_latent_with_replaced_token_latent,
        _build_modality,
        _call_av_forward,
        _find_av_forward_module,
        _inject_ltxav_runtime_metadata,
        _materialize_native_comfy_av_inputs,
        _model_looks_like_comfy_patcher,
        _module_device_and_dtype,
        _require_sampling_runtime,
        _resolve_context_fields,
        _set_working_stage1_av_latents,
        _video_latent_with_replaced_token_latent,
        _working_stage1_av_states,
    )
    from ..DFR_Spatial.dfr_execution import (
        _assert_static_model_input_compatible,
        _extract_official_tokens_from_materialized_output,
        extract_official_base_latent,
        extract_official_generated_keyframes,
    )
    from ..DFR_Spatial.latent_state import get_official_state
else:  # Standalone temporal regression tests.
    from DFR_Spatial.av_execution import (
        _audio_latent_with_replaced_token_latent,
        _build_modality,
        _call_av_forward,
        _find_av_forward_module,
        _inject_ltxav_runtime_metadata,
        _materialize_native_comfy_av_inputs,
        _model_looks_like_comfy_patcher,
        _module_device_and_dtype,
        _require_sampling_runtime,
        _resolve_context_fields,
        _set_working_stage1_av_latents,
        _video_latent_with_replaced_token_latent,
        _working_stage1_av_states,
    )
    from DFR_Spatial.dfr_execution import (
        _assert_static_model_input_compatible,
        _extract_official_tokens_from_materialized_output,
        extract_official_base_latent,
        extract_official_generated_keyframes,
    )
    from DFR_Spatial.latent_state import get_official_state

from .temporal_ancestral import (
    draw_temporal_ancestral_noise,
    make_temporal_ancestral_generator,
    temporal_ancestral_noise_seed,
    temporal_ancestral_token_update,
)
from .temporal_conditioning import (
    ANCHOR_KEYFRAME_STRENGTH,
    DFRTemporalPreparedTile,
    prepare_temporal_tile_conditioning,
)
from .temporal_handoff import DFRTemporalHandoff, TEMPORAL_HANDOFF_VERSION, require_temporal_handoff
from .temporal_noiser import (
    generator_from_temporal_handoff,
    materialize_temporal_tile_gaussian_noised_states,
    temporal_execution_device,
)
from .temporal_sigmas import temporal_sigmas, validate_temporal_sigma_schedule
from .temporal_tiles import prepare_temporal_tile_plan
from .temporal_upscale import prepare_temporal_latent_upscale


MAX_OFFICIAL_TEMPORAL_ROUNDS = 2


def _execute_temporal_tile_comfy(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    frozen_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    model_seed: int,
    ancestral_noise_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one tile in one Comfy model lifecycle with frozen audio masks."""
    samplers, sampler_helpers, model_patcher_mod, _comfy_utils = _require_sampling_runtime()
    working_video, working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
        noised_video_state,
        frozen_audio_state,
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
    negative_rt = (
        _inject_ltxav_runtime_metadata(
            negative,
            frame_rate=reference_video_input.frame_rate,
            denoise_mask=packed_mask,
            latent_shapes=latent_shapes,
            keyframe_idxs=reference_video_input.keyframe_idxs,
            generated_keyframes=reference_video_input.generated_keyframes,
        )
        if negative is not None
        else None
    )
    converted_positive = sampler_helpers.convert_cond(positive_rt)
    converted_negative = sampler_helpers.convert_cond(negative_rt) if negative_rt is not None else None
    conds_for_loading = {"positive": converted_positive}
    if converted_negative is not None:
        conds_for_loading["negative"] = converted_negative

    current_video = video_tokens["latent"]
    current_audio = audio_tokens["latent"]
    video_mask = video_tokens["denoise_mask"]
    audio_mask = audio_tokens["denoise_mask"]
    video_clean = video_tokens["clean_latent"]
    audio_clean = audio_tokens["clean_latent"]
    loaded_models = None
    processed_conds = None

    try:
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
            int(model_seed),
            latent_shapes=latent_shapes,
        )
        runtime_model_options = model_patcher_mod.create_model_options_clone(model.model_options)
        runtime_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_schedule.to(
            device=load_device
        )
        real_model.latent_shapes = latent_shapes
        model.pre_run()
        step_noise_generator = make_temporal_ancestral_generator(
            int(ancestral_noise_seed),
            torch.device(load_device),
        )

        for step_index in range(int(sigma_schedule.numel() - 1)):
            _set_working_stage1_av_latents(
                working_video,
                working_audio,
                video_tokens,
                audio_tokens,
                current_video,
                current_audio,
            )
            video_model_input, packed_latent, _current_mask, current_shapes = _materialize_native_comfy_av_inputs(
                working_video,
                working_audio,
            )
            _assert_static_model_input_compatible(reference_video_input, video_model_input)
            if current_shapes != latent_shapes:
                raise RuntimeError(
                    f"Packed Temporal-DFR AV shapes changed inside a tile: {latent_shapes} -> {current_shapes}."
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
            packed_denoised = samplers.sampling_function(
                real_model,
                packed_latent,
                sigma_batch,
                processed_conds.get("negative", None),
                processed_conds["positive"],
                float(cfg_scale),
                model_options=runtime_model_options,
                seed=int(model_seed),
            )
            packed_denoised = real_model.process_latent_out(packed_denoised.to(torch.float32))
            unpacked = _comfy_utils.unpack_latents(packed_denoised, latent_shapes)
            if len(unpacked) < 2:
                raise RuntimeError(
                    f"Expected Temporal-DFR model output [video,audio], got {len(unpacked)} modalities."
                )
            video_output, audio_output = unpacked[0], unpacked[1]
            raw_video = _extract_official_tokens_from_materialized_output(video_output, video_model_input)
            raw_audio = audio_output.permute(0, 2, 1, 3).reshape(
                audio_output.shape[0], audio_output.shape[2], -1
            ).contiguous()
            terminal_step = sigma_next == 0.0
            video_noise = None if terminal_step else draw_temporal_ancestral_noise(
                current_video,
                step_noise_generator,
                device=torch.device(load_device),
            )
            audio_noise = None if terminal_step else draw_temporal_ancestral_noise(
                current_audio,
                step_noise_generator,
                device=torch.device(load_device),
            )
            current_video = temporal_ancestral_token_update(
                current_video,
                raw_video,
                video_mask,
                video_clean,
                sigma=sigma_value,
                sigma_next=sigma_next,
                noise=video_noise,
            )
            # Frozen audio still consumes the official second RNG draw on every
            # nonterminal step, then the zero mask restores the clean latent.
            current_audio = temporal_ancestral_token_update(
                current_audio,
                raw_audio,
                audio_mask,
                audio_clean,
                sigma=sigma_value,
                sigma_next=sigma_next,
                noise=audio_noise,
            )
    finally:
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

    return (
        _video_latent_with_replaced_token_latent(noised_video_state, current_video),
        _audio_latent_with_replaced_token_latent(frozen_audio_state, current_audio),
    )


def _execute_temporal_tile_direct(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    frozen_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    ancestral_noise_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure-torch regression backend; audio receives scalar sigma zero."""
    _working_video, _working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
        noised_video_state,
        frozen_audio_state,
    )
    pos = _resolve_context_fields(positive, "positive")
    neg = _resolve_context_fields(negative, "negative") if negative is not None else None
    current_video = video_tokens["latent"]
    current_audio = audio_tokens["latent"]
    cleanup_needed = False
    if hasattr(model, "pre_run") and callable(model.pre_run):
        model.pre_run()
        cleanup_needed = hasattr(model, "cleanup") and callable(model.cleanup)

    try:
        forward_module, _forward_path = _find_av_forward_module(model)
        execution_device, model_dtype = _module_device_and_dtype(forward_module, current_video.device)

        def cast_optional(value: torch.Tensor | None, dtype: torch.dtype | None = None):
            if value is None:
                return None
            return value.to(device=execution_device, dtype=dtype or value.dtype)

        video_mask = video_tokens["denoise_mask"].to(device=execution_device, dtype=torch.float32)
        audio_mask = audio_tokens["denoise_mask"].to(device=execution_device, dtype=torch.float32)
        video_positions = video_tokens["positions"].to(device=execution_device, dtype=torch.float32)
        audio_positions = audio_tokens["positions"].to(device=execution_device, dtype=torch.float32)
        keyframes_mask = cast_optional(video_tokens.get("keyframes_mask"), torch.float32)
        video_clean = video_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        audio_clean = audio_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        pos_video_context = pos["video_context"].to(device=execution_device, dtype=model_dtype)
        pos_audio_context = pos["audio_context"].to(device=execution_device, dtype=model_dtype)
        pos_video_mask = cast_optional(pos["video_context_mask"], torch.float32)
        pos_audio_mask = cast_optional(pos["audio_context_mask"], torch.float32)

        use_cfg = neg is not None and abs(float(cfg_scale) - 1.0) > 1e-12
        if use_cfg:
            neg_video_context = neg["video_context"].to(device=execution_device, dtype=model_dtype)
            neg_audio_context = neg["audio_context"].to(device=execution_device, dtype=model_dtype)
            neg_video_mask = cast_optional(neg["video_context_mask"], torch.float32)
            neg_audio_mask = cast_optional(neg["audio_context_mask"], torch.float32)

        step_noise_generator = make_temporal_ancestral_generator(
            int(ancestral_noise_seed),
            execution_device,
        )
        for step_index in range(int(sigma_schedule.numel() - 1)):
            sigma_value = float(sigma_schedule[step_index].item())
            sigma_next = float(sigma_schedule[step_index + 1].item())
            positive_video = _build_modality(
                current_video.to(device=execution_device, dtype=model_dtype),
                video_mask,
                video_positions,
                pos_video_context,
                pos_video_mask,
                sigma_value,
                keyframes_mask=keyframes_mask,
            )
            positive_audio = _build_modality(
                current_audio.to(device=execution_device, dtype=model_dtype),
                audio_mask,
                audio_positions,
                pos_audio_context,
                pos_audio_mask,
                0.0,
            )
            pos_video_out, pos_audio_out = _call_av_forward(
                forward_module,
                positive_video,
                positive_audio,
                sigma_value,
            )

            if use_cfg:
                negative_video = _build_modality(
                    positive_video.latent,
                    video_mask,
                    video_positions,
                    neg_video_context,
                    neg_video_mask,
                    sigma_value,
                    keyframes_mask=keyframes_mask,
                )
                negative_audio = _build_modality(
                    positive_audio.latent,
                    audio_mask,
                    audio_positions,
                    neg_audio_context,
                    neg_audio_mask,
                    0.0,
                )
                neg_video_out, neg_audio_out = _call_av_forward(
                    forward_module,
                    negative_video,
                    negative_audio,
                    sigma_value,
                )
                raw_video = neg_video_out + (pos_video_out - neg_video_out) * float(cfg_scale)
                raw_audio = neg_audio_out + (pos_audio_out - neg_audio_out) * float(cfg_scale)
            else:
                raw_video = pos_video_out
                raw_audio = pos_audio_out

            terminal_step = sigma_next == 0.0
            video_noise = None if terminal_step else draw_temporal_ancestral_noise(
                current_video,
                step_noise_generator,
                device=execution_device,
            )
            audio_noise = None if terminal_step else draw_temporal_ancestral_noise(
                current_audio,
                step_noise_generator,
                device=execution_device,
            )
            current_video = temporal_ancestral_token_update(
                current_video,
                raw_video,
                video_mask,
                video_clean,
                sigma=sigma_value,
                sigma_next=sigma_next,
                noise=video_noise,
            )
            current_audio = temporal_ancestral_token_update(
                current_audio,
                raw_audio,
                audio_mask,
                audio_clean,
                sigma=sigma_value,
                sigma_next=sigma_next,
                noise=audio_noise,
            )
    finally:
        if cleanup_needed:
            try:
                model.cleanup()
            except Exception:
                pass

    return (
        _video_latent_with_replaced_token_latent(noised_video_state, current_video),
        _audio_latent_with_replaced_token_latent(frozen_audio_state, current_audio),
    )


def _execute_temporal_tile(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    frozen_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    model_seed: int,
    ancestral_noise_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _model_looks_like_comfy_patcher(model):
        return _execute_temporal_tile_comfy(
            model,
            positive,
            negative,
            noised_video_state,
            frozen_audio_state,
            sigma_schedule,
            cfg_scale,
            model_seed,
            ancestral_noise_seed,
        )
    return _execute_temporal_tile_direct(
        model,
        positive,
        negative,
        noised_video_state,
        frozen_audio_state,
        sigma_schedule,
        cfg_scale,
        ancestral_noise_seed,
    )


def _deduplicate_slots(
    positions: list[int],
    latent_slices: list[torch.Tensor],
) -> tuple[list[int], torch.Tensor | None]:
    if not positions:
        return [], None
    if not latent_slices:
        raise RuntimeError("Temporal slots have positions but no generated latent tensors.")
    combined = torch.cat(latent_slices, dim=2)
    if int(combined.shape[2]) != len(positions):
        raise RuntimeError(
            f"Temporal generated-slot latent count {combined.shape[2]} != position count {len(positions)}."
        )
    first_index: dict[int, int] = {}
    for index, position in enumerate(positions):
        first_index.setdefault(int(position), index)
    ordered = sorted(first_index)
    deduplicated = torch.cat(
        [combined[:, :, first_index[position] : first_index[position] + 1] for position in ordered],
        dim=2,
    )
    return ordered, deduplicated


def _extract_refined_anchor_keyframes(
    final_video_state: dict[str, Any],
    prepared: DFRTemporalPreparedTile,
) -> torch.Tensor:
    """Extract this tile's denoised ordinary anchor tokens as standalone frames.

    Official mode deliberately discards these token outputs and carries the
    original anchors.  The optional experimental mode keeps them.  Ranges were
    captured when the anchor conditioning operations were appended, so user
    conditions and generated slots cannot be mistaken for anchors.
    """
    if not prepared.anchor_positions_global:
        raise ValueError("Cannot extract refined anchors from a tile with no anchors.")
    if len(prepared.anchor_token_ranges) != len(prepared.anchor_positions_global):
        raise RuntimeError("Temporal anchor positions and recorded token ranges do not match.")

    state = get_official_state(final_video_state)
    token_state = state.get("token_state")
    if not isinstance(token_state, dict):
        raise ValueError("Final temporal tile has no official token state.")
    tokens = token_state.get("latent")
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        raise ValueError("Final temporal tile has no [B,N,C] token latent.")

    batch, channels, _frames, height, width = (int(value) for value in state["base_shape"])
    tokens_per_anchor = height * width
    anchors: list[torch.Tensor] = []
    for token_start, token_length in prepared.anchor_token_ranges:
        if int(token_length) != tokens_per_anchor:
            raise RuntimeError(
                f"Temporal anchor token count {token_length} != expected spatial plane {tokens_per_anchor}."
            )
        anchor_tokens = tokens[:, int(token_start) : int(token_start) + int(token_length)]
        expected_shape = (batch, tokens_per_anchor, channels)
        if tuple(anchor_tokens.shape) != expected_shape:
            raise RuntimeError(
                f"Refined temporal anchor tokens have shape {tuple(anchor_tokens.shape)}, expected {expected_shape}."
            )
        anchors.append(
            anchor_tokens.reshape(batch, 1, height, width, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
    return torch.cat(anchors, dim=2)


def _merge_carry_forward(
    anchor_positions: tuple[int, ...],
    anchor_latents: torch.Tensor,
    slot_positions: list[int],
    slot_latents: torch.Tensor | None,
) -> tuple[tuple[int, ...], torch.Tensor]:
    by_position: dict[int, torch.Tensor] = {}
    if int(anchor_latents.shape[2]) != len(anchor_positions):
        raise RuntimeError("Temporal anchor position/keyframe counts do not match.")
    for index, position in enumerate(anchor_positions):
        by_position[int(position)] = anchor_latents[:, :, index : index + 1]
    if slot_positions:
        if slot_latents is None or int(slot_latents.shape[2]) != len(slot_positions):
            raise RuntimeError("Temporal slot position/keyframe counts do not match.")
        for index, position in enumerate(slot_positions):
            by_position[int(position)] = slot_latents[:, :, index : index + 1]
    if not by_position:
        raise RuntimeError("Temporal carry-forward keyframe bag is empty.")
    ordered = tuple(sorted(by_position))
    return ordered, torch.cat([by_position[position] for position in ordered], dim=2)


def run_temporal_dfr_round(
    model: Any,
    positive: Any,
    temporal_handoff: DFRTemporalHandoff,
    upscale_model: Any,
    vae: Any,
    *,
    negative: Any = None,
    sigmas: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    anchor_strength: float = ANCHOR_KEYFRAME_STRENGTH,
    carry_refined_anchors: bool = False,
    device: torch.device | None = None,
) -> DFRTemporalHandoff:
    """Execute one complete x2 temporal round and return the next handoff."""
    source = require_temporal_handoff(temporal_handoff)
    if int(source.completed_rounds) >= MAX_OFFICIAL_TEMPORAL_ROUNDS:
        raise ValueError(
            f"Official DFR supports at most {MAX_OFFICIAL_TEMPORAL_ROUNDS} temporal x2 rounds."
        )
    anchor_strength = float(anchor_strength)
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError(f"anchor_strength must be in [0,1], got {anchor_strength}.")
    carry_refined_anchors = bool(carry_refined_anchors)
    sigma_schedule = temporal_sigmas() if sigmas is None else validate_temporal_sigma_schedule(sigmas)
    upscaled, _upscaled_latent, _upscale_report = prepare_temporal_latent_upscale(
        source,
        upscale_model,
        vae,
    )
    plan, _plan_report = prepare_temporal_tile_plan(upscaled)

    tile_latents: list[torch.Tensor] = []
    slot_positions_all: list[int] = []
    slot_latent_slices: list[torch.Tensor] = []
    refined_anchor_positions_all: list[int] = []
    refined_anchor_latent_slices: list[torch.Tensor] = []
    generator: torch.Generator | None = None
    rng_after: torch.Tensor | None = None

    for tile_index, window in enumerate(plan.tiles):
        prepared = prepare_temporal_tile_conditioning(
            plan,
            tile_index,
            anchor_strength=anchor_strength,
        )
        target_device = temporal_execution_device(prepared, device)
        if generator is None:
            generator = generator_from_temporal_handoff(prepared, target_device)
        noised_video, frozen_audio, rng_after = materialize_temporal_tile_gaussian_noised_states(
            prepared,
            float(sigma_schedule[0].item()),
            generator,
            device=target_device,
        )
        final_video, _final_audio = _execute_temporal_tile(
            model,
            positive,
            negative,
            noised_video,
            frozen_audio,
            sigma_schedule,
            float(cfg_scale),
            int(source.seed),
            temporal_ancestral_noise_seed(
                int(source.seed),
                int(upscaled.round_index),
                int(tile_index),
            ),
        )
        base = extract_official_base_latent(final_video)["samples"]
        tile_latents.append(base[:, :, int(window.interval.left_ramp) :])
        if carry_refined_anchors and window.anchor_global:
            refined_anchors = _extract_refined_anchor_keyframes(final_video, prepared)
            if int(refined_anchors.shape[2]) != len(window.anchor_global):
                raise RuntimeError(
                    f"Temporal tile {tile_index} refined K={refined_anchors.shape[2]} anchors, "
                    f"expected {len(window.anchor_global)}."
                )
            refined_anchor_positions_all.extend(int(x) for x in window.anchor_global)
            refined_anchor_latent_slices.append(refined_anchors)
        if window.slot_global:
            generated = extract_official_generated_keyframes(final_video)["samples"]
            if int(generated.shape[2]) != len(window.slot_global):
                raise RuntimeError(
                    f"Temporal tile {tile_index} generated K={generated.shape[2]}, expected {len(window.slot_global)}."
                )
            slot_positions_all.extend(int(x) for x in window.slot_global)
            slot_latent_slices.append(generated)

    if generator is None or rng_after is None:
        raise RuntimeError("Temporal tile plan produced no executable windows.")
    stitched = torch.cat(tile_latents, dim=2)
    if int(stitched.shape[2]) != int(plan.latent_length):
        raise RuntimeError(
            f"Stitched temporal latent T={stitched.shape[2]} != expected {plan.latent_length}."
        )
    slot_positions, slot_latents = _deduplicate_slots(slot_positions_all, slot_latent_slices)
    carry_anchor_latents = upscaled.anchor_keyframes
    if carry_refined_anchors:
        refined_positions, refined_latents = _deduplicate_slots(
            refined_anchor_positions_all,
            refined_anchor_latent_slices,
        )
        expected_anchor_positions = tuple(int(x) for x in upscaled.seam_positions)
        if tuple(refined_positions) != expected_anchor_positions or refined_latents is None:
            raise RuntimeError(
                "Refined temporal anchors do not cover the complete carry-forward seam set: "
                f"got {tuple(refined_positions)}, expected {expected_anchor_positions}."
            )
        carry_anchor_latents = refined_latents
    carry_positions, carry_keyframes = _merge_carry_forward(
        tuple(int(x) for x in upscaled.seam_positions),
        carry_anchor_latents,
        slot_positions,
        slot_latents,
    )

    return require_temporal_handoff(
        DFRTemporalHandoff(
            version=TEMPORAL_HANDOFF_VERSION,
            source_stage=source.source_stage,
            completed_rounds=int(upscaled.round_index),
            video_latent=stitched,
            carry_keyframes=carry_keyframes,
            carry_positions=carry_positions,
            # Preserve the official window-boundary lineage before it is merged
            # with this round's newly generated midpoint keyframes.
            last_window_seams=tuple(int(x) for x in upscaled.seam_positions),
            user_keyframes=source.user_keyframes,
            stage_1_audio_latent=source.stage_1_audio_latent,
            # This field is the continuation point before the *next* round.
            rng_state_before_temporal=rng_after.detach().cpu().clone(),
            rng_device_type=torch.device(generator.device).type,
            seed=int(source.seed),
            fps=float(upscaled.target_fps),
            duration_seconds=float(source.duration_seconds),
            temporal_scale=int(source.temporal_scale),
            requested_frames=int(upscaled.target_requested_frames),
            padded_frames=int(upscaled.target_padded_frames),
            post_temporal_spatial_completed=False,
        )
    )
