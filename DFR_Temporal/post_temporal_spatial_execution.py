"""Whole-schedule post-temporal Spatial x2 DFR executor.

Oracle reference:
  packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py
    :: _TiledModelWrapper
    :: DFRPipeline._run_spatial_epilogue
  packages/ltx-core/src/ltx_core/modality_tiling.py

The epilogue is one full-canvas diffusion trajectory.  Only the transformer
prediction is tiled: every model tile sees the complete frozen audio stream,
the video predictions are blended back into one full token sequence, and then
one Euler update is applied to that full sequence.  This is intentionally not
equivalent to sampling independent tiles and stitching their final latents.

This module remains an internal implementation.  No Comfy node is registered
until the complete boundary has been exercised from a real workflow.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Callable

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.audio_state import (
        AUDIO_OFFICIAL_STATE_KEY,
        DEFAULT_AUDIO_CAUSAL,
        DEFAULT_AUDIO_HOP_LENGTH,
        DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        DEFAULT_AUDIO_SAMPLE_RATE,
        DEFAULT_AUDIO_SHIFT,
        _audio_positions,
        _new_audio_official_state,
        _unpatchify_audio,
    )
    from ..DFR_Spatial.av_execution import (
        _build_modality,
        _call_av_forward,
        _find_av_forward_module,
        _inject_ltxav_runtime_metadata,
        _materialize_native_comfy_av_inputs,
        _model_looks_like_comfy_patcher,
        _module_device_and_dtype,
        _require_sampling_runtime,
        _resolve_context_fields,
        _stage1_av_token_euler_update,
        _video_latent_with_replaced_token_latent,
        _working_stage1_av_states,
    )
    from ..DFR_Spatial.decoder_official_tiling import compute_trapezoidal_mask_1d
    from ..DFR_Spatial.dfr_execution import (
        _extract_official_tokens_from_materialized_output,
        extract_official_base_latent,
    )
    from ..DFR_Spatial.dfr_model_bridge import DFRComfyModelInput
    from ..DFR_Spatial.dfr_noiser import (
        NOISER_METADATA_KEY,
        OFFICIAL_DFR_STATE_DTYPE,
        official_gaussian_noiser_formula,
    )
    from ..DFR_Spatial.dfr_sigmas import stage_2_sigmas, validate_custom_sigma_schedule
    from ..DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, ensure_token_state
    from ..DFR_Spatial.stage2_execution import _require_stage2_detailing_model
    from ..DFR_Spatial.stage2_noiser import _preferred_comfy_device, _randn_like_with_generator
else:  # Standalone temporal regression tests.
    from DFR_Spatial.audio_state import (
        AUDIO_OFFICIAL_STATE_KEY,
        DEFAULT_AUDIO_CAUSAL,
        DEFAULT_AUDIO_HOP_LENGTH,
        DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        DEFAULT_AUDIO_SAMPLE_RATE,
        DEFAULT_AUDIO_SHIFT,
        _audio_positions,
        _new_audio_official_state,
        _unpatchify_audio,
    )
    from DFR_Spatial.av_execution import (
        _build_modality,
        _call_av_forward,
        _find_av_forward_module,
        _inject_ltxav_runtime_metadata,
        _materialize_native_comfy_av_inputs,
        _model_looks_like_comfy_patcher,
        _module_device_and_dtype,
        _require_sampling_runtime,
        _resolve_context_fields,
        _stage1_av_token_euler_update,
        _video_latent_with_replaced_token_latent,
        _working_stage1_av_states,
    )
    from DFR_Spatial.decoder_official_tiling import compute_trapezoidal_mask_1d
    from DFR_Spatial.dfr_execution import (
        _extract_official_tokens_from_materialized_output,
        extract_official_base_latent,
    )
    from DFR_Spatial.dfr_model_bridge import DFRComfyModelInput
    from DFR_Spatial.dfr_noiser import (
        NOISER_METADATA_KEY,
        OFFICIAL_DFR_STATE_DTYPE,
        official_gaussian_noiser_formula,
    )
    from DFR_Spatial.dfr_sigmas import stage_2_sigmas, validate_custom_sigma_schedule
    from DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, ensure_token_state
    from DFR_Spatial.stage2_execution import _require_stage2_detailing_model
    from DFR_Spatial.stage2_noiser import _preferred_comfy_device, _randn_like_with_generator

from .post_temporal_spatial_conditioning import (
    DFRPostTemporalSpatialModelTile,
    DFRPostTemporalSpatialPreparedHandoff,
    assemble_post_temporal_spatial_conditioning,
    require_post_temporal_spatial_prepared_handoff,
)
from .post_temporal_spatial_upscale import prepare_post_temporal_spatial_upscale
from .temporal_conditioning import audio_latent_for_temporal_tile
from .temporal_handoff import DFRTemporalHandoff, require_temporal_handoff


POST_TEMPORAL_SPATIAL_NOISER_STAGE = "post_temporal_spatial_seed_plus_2000"
POST_TEMPORAL_SPATIAL_SEED_OFFSET = 2000


logger = logging.getLogger(__name__)


def _normalize_spatial_epilogue_sigmas(sigmas: torch.Tensor | None) -> torch.Tensor:
    if sigmas is None:
        return stage_2_sigmas()
    return validate_custom_sigma_schedule(sigmas, "post_temporal_spatial_sigmas")


def _rectangular_mask(length: int, left: int, right: int) -> torch.Tensor:
    mask = torch.ones(int(length), dtype=torch.float32)
    if int(left) > 0:
        mask[: int(left)] = 0.0
    if int(right) > 0:
        mask[-int(right) :] = 0.0
    return mask


def _tile_blend_mask(
    tile: DFRPostTemporalSpatialModelTile,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    temporal_length = int(tile.temporal.end) - int(tile.temporal.start)
    height_length = int(tile.height.end) - int(tile.height.start)
    width_length = int(tile.width.end) - int(tile.width.start)
    if tile.temporal_rectangular:
        temporal = _rectangular_mask(
            temporal_length,
            int(tile.temporal.left_ramp),
            int(tile.temporal.right_ramp),
        )
    else:
        temporal = compute_trapezoidal_mask_1d(
            temporal_length,
            int(tile.temporal.left_ramp),
            int(tile.temporal.right_ramp),
        )
    height = compute_trapezoidal_mask_1d(
        height_length,
        int(tile.height.left_ramp),
        int(tile.height.right_ramp),
    )
    width = compute_trapezoidal_mask_1d(
        width_length,
        int(tile.width.left_ramp),
        int(tile.width.right_ramp),
    )
    return (
        temporal[:, None, None] * height[None, :, None] * width[None, None, :]
    ).to(device=device, dtype=dtype)


def _base_token_indices(
    tile: DFRPostTemporalSpatialModelTile,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    frames = torch.arange(int(tile.temporal.start), int(tile.temporal.end), device=device)
    rows = torch.arange(int(tile.height.start), int(tile.height.end), device=device)
    columns = torch.arange(int(tile.width.start), int(tile.width.end), device=device)
    return (
        frames[:, None, None] * int(height) * int(width)
        + rows[None, :, None] * int(width)
        + columns[None, None, :]
    ).reshape(-1)


def _tile_token_selection(
    positions: torch.Tensor,
    *,
    base_count: int,
    tile: DFRPostTemporalSpatialModelTile,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return generated indices, kept condition indices, and tile offset.

    This is the narrow token-level port of
    ``VideoModalityTilingHelper.tile_modality``.  Condition tokens are retained
    when their three position intervals overlap the tile's generated bounds;
    negative-time references are retained by every tile.
    """
    generated = _base_token_indices(tile, height=height, width=width, device=positions.device)
    generated_positions = positions[:, :, generated, :]
    starts = generated_positions[..., 0].amin(dim=2)
    ends = generated_positions[..., 1].amax(dim=2)
    offset = starts.unsqueeze(-1).unsqueeze(-1)

    if int(positions.shape[2]) == int(base_count):
        return generated, torch.empty(0, dtype=torch.long, device=positions.device), offset

    condition_positions = positions[:, :, int(base_count) :, :]
    overlaps = (
        (condition_positions[..., 0] < ends[..., None])
        & (condition_positions[..., 1] > starts[..., None])
    ).all(dim=1)
    negative_time = condition_positions[:, 0, :, 0] < 0
    keep = (overlaps | negative_time).any(dim=0)
    condition = int(base_count) + keep.nonzero(as_tuple=False).squeeze(1)
    return generated, condition, offset


def _slice_attention_mask(mask: torch.Tensor | None, keep: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask[:, keep, :][:, :, keep]


def _slice_keyframes_mask(mask: torch.Tensor | None, keep: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask[:, keep]


def _direct_tiled_forward(
    forward_module: Any,
    video: Any,
    audio: Any,
    prepared: DFRPostTemporalSpatialPreparedHandoff,
    sigma_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official token-level tiled model wrapper for one prediction."""
    plan = prepared.tile_plan
    frames, height, width = (int(value) for value in plan.latent_shape)
    base_count = frames * height * width
    selections: list[tuple[DFRPostTemporalSpatialModelTile, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    condition_counts = torch.zeros(
        max(0, int(video.latent.shape[1]) - base_count),
        device=video.latent.device,
        dtype=torch.float32,
    )
    for tile in plan.tiles:
        generated, condition, offset = _tile_token_selection(
            video.positions,
            base_count=base_count,
            tile=tile,
            height=height,
            width=width,
        )
        selections.append((tile, generated, condition, offset))
        if condition.numel():
            condition_counts[condition - base_count] += 1.0
    video_accumulator = torch.zeros_like(video.latent)
    audio_accumulator = torch.zeros_like(audio.latent)
    for tile, generated, condition, offset in selections:
        keep = torch.cat([generated, condition])
        tiled_video = replace(
            video,
            latent=video.latent[:, keep],
            timesteps=video.timesteps[:, keep],
            positions=video.positions[:, :, keep, :] - offset,
            attention_mask=_slice_attention_mask(video.attention_mask, keep),
            keyframes_mask=_slice_keyframes_mask(video.keyframes_mask, keep),
        )
        tile_video, tile_audio = _call_av_forward(
            forward_module,
            tiled_video,
            audio,
            sigma_value,
        )
        num_generated = int(generated.numel())
        weights = _tile_blend_mask(
            tile,
            device=tile_video.device,
            dtype=tile_video.dtype,
        ).reshape(-1)
        video_accumulator[:, generated] += tile_video[:, :num_generated] * weights[None, :, None]
        if condition.numel():
            condition_weights = 1.0 / condition_counts[condition - base_count]
            video_accumulator[:, condition] += (
                tile_video[:, num_generated:] * condition_weights[None, :, None].to(tile_video.dtype)
            )
        audio_accumulator += tile_audio
    return video_accumulator, audio_accumulator / float(len(plan.tiles))


def _whole_canvas_frozen_audio(
    prepared: DFRPostTemporalSpatialPreparedHandoff,
) -> torch.Tensor:
    source = prepared.source_upscaled_handoff.source_handoff
    return audio_latent_for_temporal_tile(
        source.stage_1_audio_latent,
        pixel_start=0,
        local_frames=int(source.padded_frames),
        playback_fps=float(source.fps),
        source_duration=float(source.duration_seconds),
        conditioning_fps=float(prepared.conditioning_fps),
    )


def _prepare_frozen_audio_state(
    prepared: DFRPostTemporalSpatialPreparedHandoff,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial = _whole_canvas_frozen_audio(prepared).to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    batch, channels, frames, mel_bins = (int(value) for value in initial.shape)
    frozen_mask = torch.zeros((batch, 1, frames, 1), device=device, dtype=torch.float32)
    positions = _audio_positions(
        batch=batch,
        frames=frames,
        sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        hop_length=DEFAULT_AUDIO_HOP_LENGTH,
        audio_latent_downsample_factor=DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        is_causal=DEFAULT_AUDIO_CAUSAL,
        shift=DEFAULT_AUDIO_SHIFT,
        device=device,
    )
    source = prepared.source_upscaled_handoff.source_handoff
    state = _new_audio_official_state(
        base_shape=(batch, channels, frames, mel_bins),
        initial_latent=initial,
        clean_latent=initial.clone(),
        denoise_mask=frozen_mask,
        positions=positions,
        duration_seconds=float(source.padded_frames) / float(prepared.conditioning_fps),
        video_alignment={
            "pixel_frames": int(source.padded_frames),
            "fps": float(prepared.conditioning_fps),
            "video_base_shape": tuple(int(value) for value in prepared.video_state["samples"].shape),
            "frozen": True,
        },
        audio_config={
            "sample_rate": DEFAULT_AUDIO_SAMPLE_RATE,
            "hop_length": DEFAULT_AUDIO_HOP_LENGTH,
            "audio_latent_downsample_factor": DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
            "channels": channels,
            "mel_bins": mel_bins,
            "is_causal": DEFAULT_AUDIO_CAUSAL,
            "shift": DEFAULT_AUDIO_SHIFT,
        },
    )
    state["token_state"]["frozen"] = True
    return {"samples": initial, AUDIO_OFFICIAL_STATE_KEY: state}, state["token_state"]


def _materialize_spatial_epilogue_noised_states(
    prepared_handoff: DFRPostTemporalSpatialPreparedHandoff,
    sigma0: float,
    *,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the epilogue's fresh ``seed + 2000`` VIDEO-then-AUDIO noiser."""
    prepared = require_post_temporal_spatial_prepared_handoff(prepared_handoff)
    source = prepared.source_upscaled_handoff.source_handoff
    target = torch.device(device) if device is not None else _preferred_comfy_device(
        prepared.video_state["samples"].device
    )
    generator = torch.Generator(device=target).manual_seed(
        int(source.seed) + POST_TEMPORAL_SPATIAL_SEED_OFFSET
    )

    video_state = clone_or_create_official_state(prepared.video_state)
    video_tokens = ensure_token_state(
        video_state,
        target_samples=prepared.video_state["samples"],
        fps=float(prepared.conditioning_fps),
        scale_factors=tuple(int(value) for value in video_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(video_state.get("causal_fix", True)),
    )
    if video_tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("Prepared post-temporal Spatial video state is already Gaussian-noised.")
    video_latent = video_tokens["latent"].to(device=target, dtype=OFFICIAL_DFR_STATE_DTYPE)
    video_clean = video_tokens["clean_latent"].to(device=target, dtype=OFFICIAL_DFR_STATE_DTYPE)
    video_mask = video_tokens["denoise_mask"].to(device=target, dtype=torch.float32)
    video_tokens["positions"] = video_tokens["positions"].to(device=target, dtype=torch.float32)
    if torch.is_tensor(video_tokens.get("keyframes_mask")):
        video_tokens["keyframes_mask"] = video_tokens["keyframes_mask"].to(device=target, dtype=torch.float32)
    if torch.is_tensor(video_tokens.get("attention_mask")):
        video_tokens["attention_mask"] = video_tokens["attention_mask"].to(device=target)

    audio_latent_state, audio_tokens = _prepare_frozen_audio_state(prepared, target)
    video_noise = _randn_like_with_generator(video_latent, generator)
    # DiffusionStage invokes GaussianNoiser before applying the frozen flag, so
    # audio consumes the second draw even though noise_scale and mask are zero.
    audio_noise = _randn_like_with_generator(audio_tokens["latent"], generator)
    video_noised = official_gaussian_noiser_formula(
        video_latent,
        video_clean,
        video_mask,
        video_noise,
        float(sigma0),
    )
    audio_noised = official_gaussian_noiser_formula(
        audio_tokens["latent"],
        audio_tokens["clean_latent"],
        audio_tokens["denoise_mask"],
        audio_noise,
        0.0,
    )
    common = {
        "type": "GaussianNoiser",
        "stage": POST_TEMPORAL_SPATIAL_NOISER_STAGE,
        "seed": int(source.seed) + POST_TEMPORAL_SPATIAL_SEED_OFFSET,
        "seed_offset": POST_TEMPORAL_SPATIAL_SEED_OFFSET,
        "state_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "device_type": target.type,
        "rng_order": "video_then_audio",
        "fresh_generator": True,
    }
    video_tokens["latent"] = video_noised
    video_tokens["clean_latent"] = video_clean
    video_tokens["denoise_mask"] = video_mask
    video_tokens[NOISER_METADATA_KEY] = {
        **common,
        "modality": "video",
        "noise_scale": float(sigma0),
        "token_shape": tuple(int(value) for value in video_noised.shape),
    }
    video_state["operations"].append(
        {
            "type": "PostTemporalSpatialGaussianNoiser",
            "modality": "video",
            "seed": common["seed"],
            "noise_scale": float(sigma0),
        }
    )

    audio_tokens["latent"] = audio_noised
    audio_tokens[NOISER_METADATA_KEY] = {
        **common,
        "modality": "audio",
        "noise_scale": 0.0,
        "frozen": True,
        "token_shape": tuple(int(value) for value in audio_noised.shape),
    }
    audio_latent_state["samples"] = _unpatchify_audio(
        audio_noised,
        channels=int(audio_tokens["channels"]),
        mel_bins=int(audio_tokens["mel_bins"]),
    )
    audio_latent_state[AUDIO_OFFICIAL_STATE_KEY]["operations"].append(
        {
            "type": "PostTemporalSpatialGaussianNoiser",
            "modality": "audio",
            "seed": common["seed"],
            "noise_scale": 0.0,
            "frozen": True,
        }
    )

    out_video = prepared.video_state.copy()
    out_video[OFFICIAL_STATE_KEY] = video_state
    return out_video, audio_latent_state


def _execute_whole_schedule_direct(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    frozen_audio_state: dict[str, Any],
    prepared: DFRPostTemporalSpatialPreparedHandoff,
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
) -> dict[str, Any]:
    working_video, _working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
        noised_video_state,
        frozen_audio_state,
    )
    positive_fields = _resolve_context_fields(positive, "positive")
    negative_fields = _resolve_context_fields(negative, "negative") if negative is not None else None
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
        video_clean = video_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        video_positions = video_tokens["positions"].to(device=execution_device, dtype=torch.float32)
        video_keyframes = cast_optional(video_tokens.get("keyframes_mask"), torch.float32)
        video_attention = cast_optional(video_tokens.get("attention_mask"))
        audio_mask = audio_tokens["denoise_mask"].to(device=execution_device, dtype=torch.float32)
        audio_clean = audio_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
        audio_positions = audio_tokens["positions"].to(device=execution_device, dtype=torch.float32)

        use_cfg = negative_fields is not None and abs(float(cfg_scale) - 1.0) > 1e-12
        for step_index in range(int(sigma_schedule.numel() - 1)):
            sigma_value = float(sigma_schedule[step_index].item())
            sigma_next = float(sigma_schedule[step_index + 1].item())
            positive_video = _build_modality(
                current_video.to(device=execution_device, dtype=model_dtype),
                video_mask,
                video_positions,
                positive_fields["video_context"].to(device=execution_device, dtype=model_dtype),
                cast_optional(positive_fields["video_context_mask"], torch.float32),
                sigma_value,
                keyframes_mask=video_keyframes,
                attention_mask=video_attention,
            )
            positive_audio = _build_modality(
                current_audio.to(device=execution_device, dtype=model_dtype),
                audio_mask,
                audio_positions,
                positive_fields["audio_context"].to(device=execution_device, dtype=model_dtype),
                cast_optional(positive_fields["audio_context_mask"], torch.float32),
                0.0,
            )
            positive_video_out, positive_audio_out = _direct_tiled_forward(
                forward_module,
                positive_video,
                positive_audio,
                prepared,
                sigma_value,
            )
            if use_cfg:
                assert negative_fields is not None
                negative_video = _build_modality(
                    positive_video.latent,
                    video_mask,
                    video_positions,
                    negative_fields["video_context"].to(device=execution_device, dtype=model_dtype),
                    cast_optional(negative_fields["video_context_mask"], torch.float32),
                    sigma_value,
                    keyframes_mask=video_keyframes,
                    attention_mask=video_attention,
                )
                negative_audio = _build_modality(
                    positive_audio.latent,
                    audio_mask,
                    audio_positions,
                    negative_fields["audio_context"].to(device=execution_device, dtype=model_dtype),
                    cast_optional(negative_fields["audio_context_mask"], torch.float32),
                    0.0,
                )
                negative_video_out, negative_audio_out = _direct_tiled_forward(
                    forward_module,
                    negative_video,
                    negative_audio,
                    prepared,
                    sigma_value,
                )
                raw_video = negative_video_out + (
                    positive_video_out - negative_video_out
                ) * float(cfg_scale)
                raw_audio = negative_audio_out + (
                    positive_audio_out - negative_audio_out
                ) * float(cfg_scale)
            else:
                raw_video, raw_audio = positive_video_out, positive_audio_out

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
    finally:
        if cleanup_needed:
            try:
                model.cleanup()
            except Exception:
                pass
    return _video_latent_with_replaced_token_latent(noised_video_state, current_video)


def _absolute_base_positions(
    noised_video_state: dict[str, Any],
    *,
    fps: float,
) -> torch.Tensor:
    state = clone_or_create_official_state(noised_video_state)
    tokens = ensure_token_state(
        state,
        target_samples=noised_video_state["samples"],
        fps=float(fps),
        scale_factors=tuple(int(value) for value in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    frames, height, width = (int(value) for value in state["base_shape"][2:5])
    count = frames * height * width
    positions = tokens["positions"][:, :, :count].clone()
    positions[:, 0] *= float(fps)
    return positions.reshape(positions.shape[0], 3, frames, height, width, 2)


class _ComfyPackedSpatialTilingWrapper:
    """Tile a native Comfy packed AV ``apply_model`` call and blend its x0 output."""

    def __init__(
        self,
        prepared: DFRPostTemporalSpatialPreparedHandoff,
        reference_input: DFRComfyModelInput,
        comfy_utils: Any,
        upstream_wrapper: Callable[..., torch.Tensor] | None,
        absolute_base_positions: torch.Tensor,
    ) -> None:
        self.prepared = prepared
        self.reference_input = reference_input
        self.comfy_utils = comfy_utils
        self.upstream_wrapper = upstream_wrapper
        self.absolute_base_positions = absolute_base_positions

    def _call_model(
        self,
        apply_model: Callable[..., torch.Tensor],
        packed: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict[str, Any],
        cond_or_uncond: Any,
    ) -> torch.Tensor:
        if self.upstream_wrapper is None:
            return apply_model(packed, timestep, **conditioning)
        return self.upstream_wrapper(
            apply_model,
            {
                "input": packed,
                "timestep": timestep,
                "c": conditioning,
                "cond_or_uncond": cond_or_uncond,
            },
        )

    def __call__(self, apply_model: Callable[..., torch.Tensor], args: dict[str, Any]) -> torch.Tensor:
        conditioning = dict(args["c"])
        latent_shapes = conditioning.get("latent_shapes")
        if not isinstance(latent_shapes, (list, tuple)) or len(latent_shapes) < 2:
            raise RuntimeError("Post-temporal Spatial tiling requires packed [video,audio] latent_shapes.")
        unpacked = self.comfy_utils.unpack_latents(args["input"], latent_shapes)
        full_video, full_audio = unpacked[0], unpacked[1]
        plan = self.prepared.tile_plan
        base_frames, full_height, full_width = (int(value) for value in plan.latent_shape)
        if tuple(full_video.shape[1:]) != tuple(self.reference_input.x.shape[1:]):
            raise RuntimeError(
                "Post-temporal Spatial packed video geometry changed inside the schedule: "
                f"{tuple(full_video.shape[1:])} != {tuple(self.reference_input.x.shape[1:])}."
            )
        full_mask = conditioning.get("denoise_mask")
        full_keyframes = conditioning.get("keyframe_idxs")
        if not torch.is_tensor(full_mask) or not torch.is_tensor(full_keyframes):
            raise RuntimeError("Post-temporal Spatial tiled execution requires denoise_mask and keyframe_idxs.")
        batch = int(full_video.shape[0])
        if int(full_mask.shape[0]) == 1 and batch > 1:
            full_mask = full_mask.expand(batch, *full_mask.shape[1:])
        if int(full_keyframes.shape[0]) == 1 and batch > 1:
            full_keyframes = full_keyframes.expand(batch, *full_keyframes.shape[1:])
        if int(full_mask.shape[0]) != batch or int(full_keyframes.shape[0]) != batch:
            raise RuntimeError(
                "Post-temporal Spatial conditioning batch does not match the packed model input: "
                f"input={batch}, mask={int(full_mask.shape[0])}, positions={int(full_keyframes.shape[0])}."
            )
        mask_channels = int(full_mask.shape[1])
        if mask_channels not in (1, int(full_video.shape[1])):
            raise RuntimeError(
                "Post-temporal Spatial denoise mask must be single-channel or expanded to the video "
                f"latent channels; got mask={mask_channels}, video={int(full_video.shape[1])}."
            )
        suffix_frames = int(full_video.shape[2]) - base_frames
        expected_keyframe_tokens = suffix_frames * full_height * full_width
        if int(full_keyframes.shape[2]) != expected_keyframe_tokens:
            raise RuntimeError(
                f"Expected {expected_keyframe_tokens} materialized suffix positions, got "
                f"{int(full_keyframes.shape[2])}."
            )
        suffix_positions = full_keyframes.reshape(
            batch, 3, suffix_frames, full_height, full_width, 2
        )
        suffix_valid = full_mask[:, :1, base_frames:] >= 0
        base_positions = self.absolute_base_positions.to(device=full_video.device, dtype=torch.float32)
        if int(base_positions.shape[0]) != batch:
            if int(base_positions.shape[0]) == 1:
                base_positions = base_positions.expand(batch, -1, -1, -1, -1, -1)
            else:
                raise RuntimeError("Conditioning batch does not match post-temporal Spatial position batch.")

        tile_keeps: list[torch.Tensor] = []
        condition_counts = torch.zeros(
            (suffix_frames, full_height, full_width),
            device=full_video.device,
            dtype=torch.float32,
        )
        for tile in plan.tiles:
            tile_base_positions = base_positions[
                :,
                :,
                int(tile.temporal.start) : int(tile.temporal.end),
                int(tile.height.start) : int(tile.height.end),
                int(tile.width.start) : int(tile.width.end),
                :,
            ]
            starts = tile_base_positions[..., 0].reshape(batch, 3, -1).amin(dim=2)
            ends = tile_base_positions[..., 1].reshape(batch, 3, -1).amax(dim=2)
            overlaps = (
                (suffix_positions[..., 0] < ends[:, :, None, None, None])
                & (suffix_positions[..., 1] > starts[:, :, None, None, None])
            ).all(dim=1, keepdim=True)
            negative_time = suffix_positions[:, 0:1, ..., 0] < 0
            # Official ownership is common to the batch.  CFG may concatenate
            # positive and negative rows, hence the reduction over batch.
            keep = (suffix_valid & (overlaps | negative_time)).any(dim=0)[0]
            tile_keeps.append(keep)
            condition_counts += keep.to(dtype=torch.float32)
        video_accumulator = torch.zeros_like(full_video)
        audio_accumulator = torch.zeros_like(full_audio)
        for tile, keep in zip(plan.tiles, tile_keeps):
            t0, t1 = int(tile.temporal.start), int(tile.temporal.end)
            h0, h1 = int(tile.height.start), int(tile.height.end)
            w0, w1 = int(tile.width.start), int(tile.width.end)
            tile_height = h1 - h0
            tile_width = w1 - w0
            tile_capacity = tile_height * tile_width
            local_base = full_video[:, :, t0:t1, h0:h1, w0:w1]
            # Conditioning is filtered by its position intervals, not by the
            # storage cell used by Comfy's sparse materialization.  Pack every
            # retained token into the front of its operation frame; keyframe
            # positions remain authoritative and unused cells are grid holes.
            local_suffix = torch.zeros(
                (batch, int(full_video.shape[1]), suffix_frames, tile_height, tile_width),
                device=full_video.device,
                dtype=full_video.dtype,
            )
            local_base_mask = full_mask[:, :, t0:t1, h0:h1, w0:w1]
            local_suffix_mask = torch.full(
                (batch, mask_channels, suffix_frames, tile_height, tile_width),
                -1.0,
                device=full_mask.device,
                dtype=full_mask.dtype,
            )
            tile_base_positions = base_positions[:, :, t0:t1, h0:h1, w0:w1, :]
            offset = tile_base_positions[..., 0].reshape(batch, 3, -1).amin(dim=2)
            local_positions = torch.zeros(
                (batch, 3, suffix_frames, tile_capacity, 2),
                device=suffix_positions.device,
                dtype=suffix_positions.dtype,
            )
            full_suffix_flat = full_video[:, :, base_frames:].reshape(
                batch, int(full_video.shape[1]), suffix_frames, -1
            )
            full_mask_flat = full_mask[:, :, base_frames:].reshape(
                batch, mask_channels, suffix_frames, -1
            )
            full_positions_flat = suffix_positions.reshape(batch, 3, suffix_frames, -1, 2)
            local_suffix_flat = local_suffix.reshape(batch, int(full_video.shape[1]), suffix_frames, tile_capacity)
            local_mask_flat = local_suffix_mask.reshape(
                batch, mask_channels, suffix_frames, tile_capacity
            )
            retained_indices: list[torch.Tensor] = []
            for suffix_index in range(suffix_frames):
                indices = keep[suffix_index].reshape(-1).nonzero(as_tuple=False).squeeze(1)
                retained_indices.append(indices)
                count = int(indices.numel())
                if count > tile_capacity:
                    raise RuntimeError(
                        f"Spatial tile can store {tile_capacity} condition tokens per frame, "
                        f"but position filtering retained {count}."
                    )
                if count == 0:
                    continue
                local_suffix_flat[:, :, suffix_index, :count] = full_suffix_flat[
                    :, :, suffix_index, indices
                ]
                local_mask_flat[:, :, suffix_index, :count] = full_mask_flat[
                    :, :, suffix_index, indices
                ]
                local_positions[:, :, suffix_index, :count] = full_positions_flat[
                    :, :, suffix_index, indices
                ] - offset[:, :, None, None]
            local_video = torch.cat([local_base, local_suffix], dim=2)
            local_mask = torch.cat([local_base_mask, local_suffix_mask], dim=2)
            local_positions = local_positions.reshape(batch, 3, -1, 2)

            local_packed, local_shapes = self.comfy_utils.pack_latents([local_video, full_audio])
            local_conditioning = dict(conditioning)
            local_conditioning["latent_shapes"] = [tuple(int(value) for value in shape) for shape in local_shapes]
            local_conditioning["denoise_mask"] = local_mask
            local_conditioning["keyframe_idxs"] = local_positions
            local_conditioning.pop("generated_keyframes", None)
            if local_conditioning.get("guide_attention_entries"):
                raise RuntimeError(
                    "Per-guide attention entries are not emitted by the current DFR bridge and cannot be "
                    "retiled safely when supplied externally."
                )
            # A non-leading official tile normalizes an ordinary causal token
            # interval to [0,8], [8,16], ... and retains the sliced global
            # keyframe marker (which is false there).  A freshly patchified
            # Comfy clip would otherwise recreate a causal [0,1] first token
            # and mark it as frame zero.  Temporarily selecting non-causal
            # coordinates and suppressing that derived marker exactly restores
            # the already-sliced official modality semantics for this call.
            base_model = getattr(apply_model, "__self__", None)
            diffusion_model = getattr(base_model, "diffusion_model", None)
            restore_causal = getattr(diffusion_model, "causal_temporal_positioning", None)
            restore_marker = getattr(diffusion_model, "keyframes_abs_pos_embedding", None)
            adjust_nonleading_time = t0 > 0 and diffusion_model is not None
            if adjust_nonleading_time:
                diffusion_model.causal_temporal_positioning = False
                diffusion_model.keyframes_abs_pos_embedding = None
            try:
                local_output = self._call_model(
                    apply_model,
                    local_packed,
                    args["timestep"],
                    local_conditioning,
                    args.get("cond_or_uncond"),
                )
            finally:
                if adjust_nonleading_time:
                    diffusion_model.causal_temporal_positioning = restore_causal
                    diffusion_model.keyframes_abs_pos_embedding = restore_marker
            output_video, output_audio = self.comfy_utils.unpack_latents(local_output, local_shapes)[:2]
            blend = _tile_blend_mask(
                tile,
                device=output_video.device,
                dtype=output_video.dtype,
            )
            video_accumulator[:, :, t0:t1, h0:h1, w0:w1] += (
                output_video[:, :, : t1 - t0] * blend[None, None]
            )
            output_suffix = output_video[:, :, t1 - t0 :].reshape(
                batch, int(output_video.shape[1]), suffix_frames, tile_capacity
            )
            accumulator_suffix = video_accumulator[:, :, base_frames:].reshape(
                batch, int(output_video.shape[1]), suffix_frames, -1
            )
            for suffix_index, indices in enumerate(retained_indices):
                count = int(indices.numel())
                if count == 0:
                    continue
                weights = (
                    1.0
                    / condition_counts[suffix_index].reshape(-1)[indices].clamp_min(1.0)
                ).to(dtype=output_suffix.dtype)
                accumulator_suffix[:, :, suffix_index, indices] += (
                    output_suffix[:, :, suffix_index, :count] * weights[None, None]
                )
            audio_accumulator += output_audio

        full_output, _ = self.comfy_utils.pack_latents(
            [video_accumulator, audio_accumulator / float(len(plan.tiles))]
        )
        return full_output


def _execute_whole_schedule_comfy(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    frozen_audio_state: dict[str, Any],
    prepared: DFRPostTemporalSpatialPreparedHandoff,
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    seed: int,
) -> dict[str, Any]:
    samplers, sampler_helpers, model_patcher_mod, comfy_utils = _require_sampling_runtime()
    working_video, working_audio, video_tokens, audio_tokens = _working_stage1_av_states(
        noised_video_state,
        frozen_audio_state,
    )
    reference_input, initial_packed, packed_mask, latent_shapes = _materialize_native_comfy_av_inputs(
        working_video,
        working_audio,
    )
    positive_rt = _inject_ltxav_runtime_metadata(
        positive,
        frame_rate=reference_input.frame_rate,
        denoise_mask=packed_mask,
        latent_shapes=latent_shapes,
        keyframe_idxs=reference_input.keyframe_idxs,
        generated_keyframes=None,
    )
    negative_rt = (
        _inject_ltxav_runtime_metadata(
            negative,
            frame_rate=reference_input.frame_rate,
            denoise_mask=packed_mask,
            latent_shapes=latent_shapes,
            keyframe_idxs=reference_input.keyframe_idxs,
            generated_keyframes=None,
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
            int(seed),
            latent_shapes=latent_shapes,
        )
        runtime_options = model_patcher_mod.create_model_options_clone(model.model_options)
        runtime_options.setdefault("transformer_options", {})["sample_sigmas"] = sigma_schedule.to(
            device=load_device
        )
        upstream_wrapper = runtime_options.get("model_function_wrapper")
        runtime_options["model_function_wrapper"] = _ComfyPackedSpatialTilingWrapper(
            prepared,
            reference_input,
            comfy_utils,
            upstream_wrapper,
            _absolute_base_positions(noised_video_state, fps=prepared.conditioning_fps),
        )
        real_model.latent_shapes = latent_shapes
        model.pre_run()

        for step_index in range(int(sigma_schedule.numel() - 1)):
            video_tokens["latent"] = current_video
            audio_tokens["latent"] = current_audio
            working_audio["samples"] = _unpatchify_audio(
                current_audio,
                channels=int(audio_tokens["channels"]),
                mel_bins=int(audio_tokens["mel_bins"]),
            )
            video_input, packed_latent, _mask, current_shapes = _materialize_native_comfy_av_inputs(
                working_video,
                working_audio,
            )
            if tuple(video_input.x.shape) != tuple(reference_input.x.shape) or current_shapes != latent_shapes:
                raise RuntimeError("Post-temporal Spatial packed geometry changed inside the schedule.")
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
                model_options=runtime_options,
                seed=int(seed),
            )
            packed_denoised = real_model.process_latent_out(packed_denoised.to(torch.float32))
            output_video, output_audio = comfy_utils.unpack_latents(packed_denoised, latent_shapes)[:2]
            # Repack only surviving official tokens.  The established bridge is
            # authoritative for sparse reference extraction.
            raw_video = _extract_official_tokens_from_materialized_output(output_video, video_input)
            raw_audio = output_audio.permute(0, 2, 1, 3).reshape(
                output_audio.shape[0], output_audio.shape[2], -1
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
    return _video_latent_with_replaced_token_latent(noised_video_state, current_video)


def execute_post_temporal_spatial_dfr(
    model: Any,
    positive: Any,
    negative: Any,
    prepared_handoff: DFRPostTemporalSpatialPreparedHandoff,
    *,
    sigmas: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    device: torch.device | None = None,
) -> DFRTemporalHandoff:
    """Run the complete terminal Spatial x2 schedule and return a decode-ready handoff."""
    prepared = require_post_temporal_spatial_prepared_handoff(prepared_handoff)
    _require_stage2_detailing_model(model)
    schedule = _normalize_spatial_epilogue_sigmas(sigmas)
    if float(schedule[0].item()) > 1.0:
        raise ValueError("Post-temporal Spatial sigma0 cannot exceed the Gaussian noiser's [0,1] scale.")
    target = torch.device(device) if device is not None else _preferred_comfy_device(
        prepared.video_state["samples"].device
    )
    noised_video, frozen_audio = _materialize_spatial_epilogue_noised_states(
        prepared,
        float(schedule[0].item()),
        device=target,
    )
    source = require_temporal_handoff(prepared.source_upscaled_handoff.source_handoff)
    if _model_looks_like_comfy_patcher(model):
        final_state = _execute_whole_schedule_comfy(
            model,
            positive,
            negative,
            noised_video,
            frozen_audio,
            prepared,
            schedule,
            float(cfg_scale),
            int(source.seed),
        )
    else:
        final_state = _execute_whole_schedule_direct(
            model,
            positive,
            negative,
            noised_video,
            frozen_audio,
            prepared,
            schedule,
            float(cfg_scale),
        )
    final_video = extract_official_base_latent(final_state)["samples"]
    upscaled = prepared.source_upscaled_handoff
    result = replace(
        source,
        video_latent=final_video,
        carry_keyframes=upscaled.upscaled_carry_keyframes.to(
            device=final_video.device,
            dtype=final_video.dtype,
        ),
        user_keyframes=tuple(
            replace(
                keyframe,
                latent=keyframe.latent.to(device=final_video.device, dtype=final_video.dtype),
            )
            for keyframe in upscaled.upscaled_user_keyframes
        ),
        post_temporal_spatial_completed=True,
    )
    return require_temporal_handoff(result)


def run_post_temporal_spatial_dfr(
    model: Any,
    positive: Any,
    temporal_handoff: DFRTemporalHandoff,
    spatial_upscale_model: Any,
    vae: Any,
    *,
    negative: Any = None,
    sigmas: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    keyframe_strength: float = 1.0,
    device: torch.device | None = None,
) -> DFRTemporalHandoff:
    """Consolidated latent-upscale -> conditioning -> tiled execution path."""
    source = require_temporal_handoff(temporal_handoff)
    upscaled = prepare_post_temporal_spatial_upscale(
        source,
        spatial_upscale_model,
        vae,
        keyframe_strength=float(keyframe_strength),
    )
    prepared = assemble_post_temporal_spatial_conditioning(upscaled)
    schedule = _normalize_spatial_epilogue_sigmas(sigmas)
    plan = prepared.tile_plan
    logger.info(
        "LTX Temporal DFR post-Spatial x2: latent %s -> %s; model tiles=%d "
        "(%d temporal x %d height x %d width); steps=%d; transformer forwards=%d",
        tuple(int(value) for value in source.video_latent.shape),
        tuple(int(value) for value in upscaled.upscaled_video_latent.shape),
        len(plan.tiles),
        int(plan.effective_temporal_tiles),
        int(plan.effective_height_tiles),
        int(plan.effective_width_tiles),
        int(schedule.numel() - 1),
        len(plan.tiles) * int(schedule.numel() - 1),
    )
    result = execute_post_temporal_spatial_dfr(
        model,
        positive,
        negative,
        prepared,
        sigmas=schedule,
        cfg_scale=float(cfg_scale),
        device=device,
    )
    expected_shape = tuple(int(value) for value in upscaled.upscaled_video_latent.shape)
    actual_shape = tuple(int(value) for value in result.video_latent.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "Post-temporal Spatial x2 returned the wrong terminal video shape: "
            f"got {actual_shape}, expected {expected_shape}."
        )
    logger.info(
        "LTX Temporal DFR post-Spatial x2 complete: terminal latent=%s",
        actual_shape,
    )
    return result
