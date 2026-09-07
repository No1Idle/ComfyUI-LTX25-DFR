"""Prepare one Temporal-DFR tile before Gaussian noising and denoising.

This ports the deterministic preparation inside the official temporal loop:
tile-local user-keyframe rebasing, carried seam anchors, seeded generated slots,
and the frozen Stage-1 audio window.  It deliberately performs no RNG draw and
no transformer call, so the next executor can consume it sequentially per tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.latent_state import (
        apply_video_condition_by_keyframe_index,
        apply_video_condition_by_latent_index,
        apply_video_generated_keyframe_slots,
        get_official_state,
    )
else:  # Standalone temporal regression tests.
    from DFR_Spatial.latent_state import (
        apply_video_condition_by_keyframe_index,
        apply_video_condition_by_latent_index,
        apply_video_generated_keyframe_slots,
        get_official_state,
    )

from .temporal_tiles import DFRTemporalTilePlanHandoff, require_temporal_tile_plan_handoff


TEMPORAL_PREPARED_TILE_VERSION = 1
ANCHOR_KEYFRAME_STRENGTH = 0.95
DEFAULT_AUDIO_LATENTS_PER_SECOND = 25.0


@dataclass(frozen=True)
class DFRTemporalPreparedTile:
    version: int
    tile_index: int
    source_plan: DFRTemporalTilePlanHandoff
    video_state: dict[str, Any]
    tile_video_latent: torch.Tensor
    frozen_audio_latent: torch.Tensor
    playback_fps: float
    conditioning_fps: float
    user_positions_global: tuple[int, ...]
    user_positions_local: tuple[int, ...]
    anchor_positions_global: tuple[int, ...]
    anchor_positions_local: tuple[int, ...]
    anchor_token_ranges: tuple[tuple[int, int], ...]
    anchor_strength: float
    slot_positions_global: tuple[int, ...]
    slot_positions_local: tuple[int, ...]
    slot_initial_keyframes: torch.Tensor | None
    audio_frozen: bool
    audio_noise_scale: float


def _resample_audio_time(
    audio_latent: torch.Tensor,
    src_start: float,
    src_end: float,
    out_frames: int,
) -> torch.Tensor:
    """Exact local port of the official linear temporal audio sampler."""
    if not torch.is_tensor(audio_latent) or audio_latent.ndim != 4:
        raise ValueError("audio_latent must be a [B,C,T,F] tensor.")
    if out_frames < 1:
        raise ValueError(f"out_frames must be >= 1, got {out_frames}.")
    full_t = int(audio_latent.shape[2])
    if full_t < 1:
        raise ValueError("Cannot resample an empty audio latent.")
    span = float(src_end) - float(src_start)
    if span <= 0.0:
        raise ValueError(f"Audio window is empty: [{src_start}, {src_end}).")

    step = span / int(out_frames)
    positions = float(src_start) + step * torch.arange(
        out_frames,
        device=audio_latent.device,
        dtype=torch.float32,
    )
    positions = positions.clamp(0, full_t - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=full_t - 1)
    weight = (positions - lower.to(torch.float32)).to(dtype=audio_latent.dtype).view(1, 1, -1, 1)
    return audio_latent[:, :, lower] * (1 - weight) + audio_latent[:, :, upper] * weight


def audio_latent_for_temporal_tile(
    audio_latent: torch.Tensor,
    *,
    pixel_start: int,
    local_frames: int,
    playback_fps: float,
    source_duration: float,
    conditioning_fps: float,
) -> torch.Tensor:
    """Slice Stage-1 audio by playback time and retime it to transformer FPS."""
    if local_frames < 1:
        raise ValueError(f"local_frames must be >= 1, got {local_frames}.")
    if playback_fps <= 0.0 or conditioning_fps <= 0.0:
        raise ValueError("playback_fps and conditioning_fps must be positive.")
    if source_duration <= 0.0:
        raise ValueError(f"source_duration must be positive, got {source_duration}.")

    full_t = int(audio_latent.shape[2])
    src_start = int(pixel_start) / float(playback_fps) / float(source_duration) * full_t
    src_end = (
        (int(pixel_start) + int(local_frames))
        / float(playback_fps)
        / float(source_duration)
        * full_t
    )
    target_frames = round(
        (float(local_frames) / float(conditioning_fps)) * DEFAULT_AUDIO_LATENTS_PER_SECOND
    )
    return _resample_audio_time(audio_latent, src_start, src_end, target_frames)


def _slot_initials_from_video(
    video_latent: torch.Tensor,
    positions: tuple[int, ...],
    temporal_scale: int,
) -> torch.Tensor | None:
    """Match the official nearest-frame initialization for generated slots."""
    if not positions:
        return None
    frames: list[torch.Tensor] = []
    for position in positions:
        index = min(
            max(round(int(position) / int(temporal_scale)), 0),
            int(video_latent.shape[2]) - 1,
        )
        frames.append(video_latent[:, :, index : index + 1])
    return torch.cat(frames, dim=2)


def require_temporal_prepared_tile(value: Any) -> DFRTemporalPreparedTile:
    if not isinstance(value, DFRTemporalPreparedTile):
        raise ValueError(f"Expected DFRTemporalPreparedTile, got {type(value).__name__}.")
    if int(value.version) != TEMPORAL_PREPARED_TILE_VERSION:
        raise ValueError(f"Unsupported prepared temporal tile version {value.version}.")
    plan = require_temporal_tile_plan_handoff(value.source_plan)
    if not 0 <= int(value.tile_index) < len(plan.tiles):
        raise ValueError(f"Prepared tile_index={value.tile_index} is outside the plan.")
    window = plan.tiles[int(value.tile_index)]
    expected_video = plan.source_upscaled_handoff.upscaled_video_latent[
        :, :, window.interval.start : window.interval.end
    ]
    if tuple(value.tile_video_latent.shape) != tuple(expected_video.shape):
        raise ValueError(
            f"Prepared tile video shape {tuple(value.tile_video_latent.shape)} does not match "
            f"planned shape {tuple(expected_video.shape)}."
        )
    if value.anchor_positions_global != window.anchor_global:
        raise ValueError("Prepared tile anchors do not match its tile window.")
    if len(value.anchor_token_ranges) != len(value.anchor_positions_global):
        raise ValueError("Prepared tile anchor token ranges do not match its anchor count.")
    if not 0.0 <= float(value.anchor_strength) <= 1.0:
        raise ValueError(f"Prepared tile anchor_strength must be in [0,1], got {value.anchor_strength}.")
    if value.slot_positions_global != window.slot_global:
        raise ValueError("Prepared tile slots do not match its tile window.")
    if value.audio_frozen is not True or float(value.audio_noise_scale) != 0.0:
        raise ValueError("Temporal tile audio must be frozen with noise_scale=0.")
    if not torch.is_tensor(value.frozen_audio_latent) or value.frozen_audio_latent.ndim != 4:
        raise ValueError("Prepared frozen audio must be a [B,C,T,F] tensor.")

    state = get_official_state(value.video_state)
    token_state = state.get("token_state")
    if not isinstance(token_state, dict):
        raise ValueError("Prepared temporal video has no official token state.")
    token_count = int(token_state["latent"].shape[1])
    for token_start, token_length in value.anchor_token_ranges:
        if token_start < 0 or token_length < 1 or token_start + token_length > token_count:
            raise ValueError(
                f"Prepared anchor token range [{token_start},{token_start + token_length}) "
                f"is outside the {token_count}-token state."
            )
    layout = token_state.get("generated_keyframe_layout")
    if window.slot_local:
        if not isinstance(layout, dict):
            raise ValueError("Prepared temporal video is missing generated-slot metadata.")
        if tuple(layout.get("pixel_frame_indices", ())) != window.slot_local:
            raise ValueError("Prepared generated-slot positions do not match the tile-local plan.")
    return value


def prepare_temporal_tile_conditioning(
    temporal_tile_plan: DFRTemporalTilePlanHandoff,
    tile_index: int,
    *,
    anchor_strength: float = ANCHOR_KEYFRAME_STRENGTH,
) -> DFRTemporalPreparedTile:
    """Prepare one tile in official conditioning order without noising it."""
    plan = require_temporal_tile_plan_handoff(temporal_tile_plan)
    tile_index = int(tile_index)
    if not 0 <= tile_index < len(plan.tiles):
        raise ValueError(f"tile_index={tile_index} is outside 0..{len(plan.tiles) - 1}.")
    anchor_strength = float(anchor_strength)
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError(f"anchor_strength must be in [0,1], got {anchor_strength}.")

    upscaled = plan.source_upscaled_handoff
    source = upscaled.source_handoff
    window = plan.tiles[tile_index]
    tile_video = upscaled.upscaled_video_latent[:, :, window.interval.start : window.interval.end]
    video_state: dict[str, Any] = {"samples": tile_video}

    user_global: list[int] = []
    user_local: list[int] = []
    pixel_scale = 2 ** int(upscaled.round_index)
    for keyframe in source.user_keyframes:
        scaled_position = int(keyframe.pixel_frame_index) * pixel_scale
        if not window.pixel_start <= scaled_position <= window.pixel_end:
            continue
        local_position = scaled_position - window.pixel_start
        conditioning = {"samples": keyframe.latent}
        if local_position == 0:
            video_state = apply_video_condition_by_latent_index(
                target_latent=video_state,
                conditioning_latent=conditioning,
                strength=float(keyframe.strength),
                latent_idx=0,
            )
        else:
            video_state = apply_video_condition_by_keyframe_index(
                target_latent=video_state,
                conditioning_latent=conditioning,
                frame_idx=local_position,
                strength=float(keyframe.strength),
                fps=float(upscaled.conditioning_fps),
                num_pixel_frames=1,
            )
        user_global.append(scaled_position)
        user_local.append(local_position)

    seam_to_index = {
        int(position): index for index, position in enumerate(upscaled.seam_positions)
    }
    anchor_token_ranges: list[tuple[int, int]] = []
    for global_position, local_position in zip(window.anchor_global, window.anchor_local):
        if global_position not in seam_to_index:
            raise ValueError(f"Tile anchor {global_position} is missing from the carry-forward bag.")
        anchor_index = seam_to_index[global_position]
        anchor = upscaled.anchor_keyframes[:, :, anchor_index : anchor_index + 1]
        video_state = apply_video_condition_by_keyframe_index(
            target_latent=video_state,
            conditioning_latent={"samples": anchor},
            frame_idx=int(local_position),
            strength=anchor_strength,
            fps=float(upscaled.conditioning_fps),
            num_pixel_frames=1,
        )
        anchor_op = get_official_state(video_state)["operations"][-1]
        if anchor_op.get("type") != "VideoConditionByKeyframeIndex":
            raise RuntimeError("Temporal anchor conditioning did not record its token range.")
        anchor_token_ranges.append(
            (int(anchor_op["appended_token_start"]), int(anchor_op["appended_token_count"]))
        )

    slot_initials = _slot_initials_from_video(
        tile_video,
        window.slot_local,
        int(source.temporal_scale),
    )
    if window.slot_local:
        assert slot_initials is not None
        video_state = apply_video_generated_keyframe_slots(
            target_latent=video_state,
            pixel_frame_indices=window.slot_local,
            initial_keyframes={"samples": slot_initials},
            fps=float(upscaled.conditioning_fps),
        )

    frozen_audio = audio_latent_for_temporal_tile(
        source.stage_1_audio_latent,
        pixel_start=window.pixel_start,
        local_frames=window.local_frames,
        playback_fps=float(upscaled.target_fps),
        source_duration=float(source.duration_seconds),
        conditioning_fps=float(upscaled.conditioning_fps),
    )

    prepared = DFRTemporalPreparedTile(
        version=TEMPORAL_PREPARED_TILE_VERSION,
        tile_index=tile_index,
        source_plan=plan,
        video_state=video_state,
        tile_video_latent=tile_video,
        frozen_audio_latent=frozen_audio,
        playback_fps=float(upscaled.target_fps),
        conditioning_fps=float(upscaled.conditioning_fps),
        user_positions_global=tuple(user_global),
        user_positions_local=tuple(user_local),
        anchor_positions_global=window.anchor_global,
        anchor_positions_local=window.anchor_local,
        anchor_token_ranges=tuple(anchor_token_ranges),
        anchor_strength=anchor_strength,
        slot_positions_global=window.slot_global,
        slot_positions_local=window.slot_local,
        slot_initial_keyframes=slot_initials,
        audio_frozen=True,
        audio_noise_scale=0.0,
    )
    return require_temporal_prepared_tile(prepared)
