"""Tile geometry and conditioning for the post-temporal Spatial x2 epilogue.

Oracle reference:
  packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py
  packages/ltx-core/src/ltx_core/modality_tiling.py
  packages/ltx-core/src/ltx_core/tiling.py

The final spatial pass samples one full target-resolution canvas while the model
forward is tiled internally.  Temporal windows are cut on the last temporal
round's seams with rectangular ownership (the earlier window keeps the seam).
The 2x2 spatial grid uses ordinary overlap/blending with 12 latent cells on each
spatial axis.  Conditionings stay full-canvas so the future model wrapper can
filter their tokens per tile exactly as the official helper does.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.decoder_official_tiling import DimensionInterval
    from ..DFR_Spatial.dfr_layout import pixel_to_latent_index
    from ..DFR_Spatial.latent_state import (
        apply_video_condition_by_keyframe_index,
        apply_video_condition_by_latent_index,
        apply_video_condition_by_reference_latent,
        get_official_state,
    )
else:  # Standalone temporal regression tests.
    from DFR_Spatial.decoder_official_tiling import DimensionInterval
    from DFR_Spatial.dfr_layout import pixel_to_latent_index
    from DFR_Spatial.latent_state import (
        apply_video_condition_by_keyframe_index,
        apply_video_condition_by_latent_index,
        apply_video_condition_by_reference_latent,
        get_official_state,
    )

from .post_temporal_spatial_upscale import (
    DFRPostTemporalSpatialUpscaledHandoff,
    require_post_temporal_spatial_upscaled_handoff,
)
from .temporal_tiles import _split_canvas_at_seams
from .temporal_upscale import official_temporal_conditioning_fps


POST_TEMPORAL_SPATIAL_TILE_PLAN_VERSION = 1
POST_TEMPORAL_SPATIAL_PREPARED_VERSION = 1
OFFICIAL_EPILOGUE_SPATIAL_TILES = 2
OFFICIAL_EPILOGUE_SPATIAL_OVERLAP = 12
OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE = 2
OFFICIAL_EPILOGUE_REFERENCE_STRENGTH = 1.0


@dataclass(frozen=True)
class DFRPostTemporalSpatialModelTile:
    """One model-forward tile in official temporal/height/width product order."""

    index: int
    temporal: DimensionInterval
    height: DimensionInterval
    width: DimensionInterval
    temporal_rectangular: bool


@dataclass(frozen=True)
class DFRPostTemporalSpatialTilePlan:
    """Exact count-based model tiling requested by the spatial epilogue."""

    version: int
    source_upscaled_handoff: DFRPostTemporalSpatialUpscaledHandoff
    latent_shape: tuple[int, int, int]
    seam_latent_positions: tuple[int, ...]
    requested_temporal_tiles: int
    effective_temporal_tiles: int
    effective_height_tiles: int
    effective_width_tiles: int
    temporal_overlap_latent_cells: int
    height_overlap_latent_cells: int
    width_overlap_latent_cells: int
    temporal_intervals: tuple[DimensionInterval, ...]
    height_intervals: tuple[DimensionInterval, ...]
    width_intervals: tuple[DimensionInterval, ...]
    tiles: tuple[DFRPostTemporalSpatialModelTile, ...]


@dataclass(frozen=True)
class DFRPostTemporalSpatialPreparedHandoff:
    """Full-canvas conditioning and tile plan ready for the future executor."""

    version: int
    source_upscaled_handoff: DFRPostTemporalSpatialUpscaledHandoff
    tile_plan: DFRPostTemporalSpatialTilePlan
    video_state: dict[str, Any]
    conditioning_fps: float
    scaled_user_positions: tuple[int, ...]
    carry_positions: tuple[int, ...]
    reference_downscale_factor: int
    reference_strength: float


def _untiled_interval(length: int) -> tuple[DimensionInterval, ...]:
    return (DimensionInterval(0, int(length), 0, 0),)


def _clamp_count_axis(
    num_tiles: int,
    overlap: int,
    length: int,
) -> tuple[int, int]:
    """Exact oracle clamp for count-based latent-grid tiling."""
    num_tiles = int(num_tiles)
    overlap = int(overlap)
    length = int(length)
    if num_tiles <= 1:
        return num_tiles, overlap
    if length < num_tiles:
        return 1, 0
    return num_tiles, min(overlap, length - num_tiles)


def _split_by_size(length: int, size: int, overlap: int) -> list[DimensionInterval]:
    if length <= size:
        return list(_untiled_interval(length))
    amount = (length + size - 2 * overlap - 1) // (size - overlap)
    return [
        DimensionInterval(0, size, 0, overlap),
        *[
            DimensionInterval(
                index * (size - overlap),
                index * (size - overlap) + size,
                overlap,
                overlap,
            )
            for index in range(1, amount - 1)
        ],
        DimensionInterval((amount - 1) * (size - overlap), length, overlap, 0),
    ]


def _split_by_count(
    length: int,
    num_tiles: int,
    overlap: int,
) -> tuple[DimensionInterval, ...]:
    """Narrow port of official non-causal ``split_by_count``."""
    if num_tiles <= 1:
        return _untiled_interval(length)
    total = int(length) + int(overlap) * (int(num_tiles) - 1)
    tile_size = total // int(num_tiles)
    if tile_size <= overlap:
        raise ValueError(
            f"Cannot split length={length} into {num_tiles} tiles with overlap={overlap}."
        )
    remainder = total % int(num_tiles)
    base_intervals = _split_by_size(int(length) - remainder, tile_size, int(overlap))
    intervals = tuple(
        replace(
            interval,
            start=int(interval.start) + min(index, remainder),
            end=int(interval.end) + min(index, remainder) + (1 if index < remainder else 0),
        )
        for index, interval in enumerate(base_intervals)
    )
    if len(intervals) != int(num_tiles):
        raise RuntimeError(
            f"Count split produced {len(intervals)} intervals, expected {num_tiles}."
        )
    return intervals


def _build_plan_geometry(
    source: DFRPostTemporalSpatialUpscaledHandoff,
) -> dict[str, Any]:
    temporal = source.source_handoff
    frames, height, width = (
        int(value) for value in source.upscaled_video_latent.shape[2:5]
    )
    seam_latents = tuple(
        pixel_to_latent_index(position, int(temporal.temporal_scale))
        for position in temporal.last_window_seams
    )
    requested_temporal_tiles = max(1, 2 ** int(temporal.completed_rounds))
    requested_temporal_overlap = (
        int(seam_latents[0]) + 1
        if requested_temporal_tiles > 1 and seam_latents
        else 0
    )

    temporal_tiles, temporal_overlap = _clamp_count_axis(
        requested_temporal_tiles,
        requested_temporal_overlap,
        frames,
    )
    height_tiles, height_overlap = _clamp_count_axis(
        OFFICIAL_EPILOGUE_SPATIAL_TILES,
        OFFICIAL_EPILOGUE_SPATIAL_OVERLAP,
        height,
    )
    width_tiles, width_overlap = _clamp_count_axis(
        OFFICIAL_EPILOGUE_SPATIAL_TILES,
        OFFICIAL_EPILOGUE_SPATIAL_OVERLAP,
        width,
    )

    interior_seams = tuple(
        sorted({position for position in seam_latents if 0 < position < frames - 1})
    )
    temporal_rectangular = bool(temporal_tiles > 1 and interior_seams)
    if temporal_rectangular:
        temporal_intervals = tuple(
            _split_canvas_at_seams(
                (0, *interior_seams, frames - 1),
                temporal_tiles,
                temporal_overlap,
                frames,
            )
        )
    else:
        temporal_intervals = _split_by_count(frames, temporal_tiles, temporal_overlap)

    height_intervals = _split_by_count(height, height_tiles, height_overlap)
    width_intervals = _split_by_count(width, width_tiles, width_overlap)
    tiles = tuple(
        DFRPostTemporalSpatialModelTile(
            index=index,
            temporal=time_interval,
            height=height_interval,
            width=width_interval,
            temporal_rectangular=temporal_rectangular,
        )
        for index, (time_interval, height_interval, width_interval) in enumerate(
            itertools.product(temporal_intervals, height_intervals, width_intervals)
        )
    )
    return {
        "latent_shape": (frames, height, width),
        "seam_latent_positions": seam_latents,
        "requested_temporal_tiles": requested_temporal_tiles,
        "effective_temporal_tiles": len(temporal_intervals),
        "effective_height_tiles": len(height_intervals),
        "effective_width_tiles": len(width_intervals),
        "temporal_overlap_latent_cells": temporal_overlap,
        "height_overlap_latent_cells": height_overlap,
        "width_overlap_latent_cells": width_overlap,
        "temporal_intervals": temporal_intervals,
        "height_intervals": height_intervals,
        "width_intervals": width_intervals,
        "tiles": tiles,
    }


def require_post_temporal_spatial_tile_plan(
    value: Any,
) -> DFRPostTemporalSpatialTilePlan:
    if not isinstance(value, DFRPostTemporalSpatialTilePlan):
        raise ValueError(
            f"Expected DFRPostTemporalSpatialTilePlan, got {type(value).__name__}."
        )
    if int(value.version) != POST_TEMPORAL_SPATIAL_TILE_PLAN_VERSION:
        raise ValueError(f"Unsupported post-temporal Spatial tile-plan version {value.version}.")
    source = require_post_temporal_spatial_upscaled_handoff(value.source_upscaled_handoff)
    expected = _build_plan_geometry(source)
    for field_name, expected_value in expected.items():
        if getattr(value, field_name) != expected_value:
            raise ValueError(
                f"Post-temporal Spatial tile-plan {field_name} does not match official geometry."
            )
    if len(value.tiles) != (
        int(value.effective_temporal_tiles)
        * int(value.effective_height_tiles)
        * int(value.effective_width_tiles)
    ):
        raise ValueError("Post-temporal Spatial tile product does not match its axis counts.")
    return value


def prepare_post_temporal_spatial_tile_plan(
    upscaled_handoff: DFRPostTemporalSpatialUpscaledHandoff,
) -> DFRPostTemporalSpatialTilePlan:
    source = require_post_temporal_spatial_upscaled_handoff(upscaled_handoff)
    geometry = _build_plan_geometry(source)
    return require_post_temporal_spatial_tile_plan(
        DFRPostTemporalSpatialTilePlan(
            version=POST_TEMPORAL_SPATIAL_TILE_PLAN_VERSION,
            source_upscaled_handoff=source,
            **geometry,
        )
    )


def require_post_temporal_spatial_prepared_handoff(
    value: Any,
) -> DFRPostTemporalSpatialPreparedHandoff:
    if not isinstance(value, DFRPostTemporalSpatialPreparedHandoff):
        raise ValueError(
            f"Expected DFRPostTemporalSpatialPreparedHandoff, got {type(value).__name__}."
        )
    if int(value.version) != POST_TEMPORAL_SPATIAL_PREPARED_VERSION:
        raise ValueError(f"Unsupported post-temporal Spatial prepared version {value.version}.")
    source = require_post_temporal_spatial_upscaled_handoff(value.source_upscaled_handoff)
    plan = require_post_temporal_spatial_tile_plan(value.tile_plan)
    if plan.source_upscaled_handoff is not source:
        raise ValueError("Spatial conditioning and tile plan must share the same upscaled handoff.")

    expected_fps = official_temporal_conditioning_fps(source.source_handoff.fps)
    if float(value.conditioning_fps) != expected_fps:
        raise ValueError(f"Spatial conditioning_fps={value.conditioning_fps}, expected {expected_fps}.")
    expected_user_positions = tuple(
        int(keyframe.pixel_frame_index) * 2 ** int(source.source_handoff.completed_rounds)
        for keyframe in source.upscaled_user_keyframes
    )
    if tuple(int(position) for position in value.scaled_user_positions) != expected_user_positions:
        raise ValueError("Post-temporal Spatial user-keyframe positions were not scaled correctly.")
    if tuple(int(position) for position in value.carry_positions) != tuple(
        int(position) for position in source.source_handoff.carry_positions
    ):
        raise ValueError("Post-temporal Spatial conditioning must retain every carry position.")
    if int(value.reference_downscale_factor) != OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE:
        raise ValueError("Post-temporal Spatial reference downscale factor must be 2.")
    if float(value.reference_strength) != OFFICIAL_EPILOGUE_REFERENCE_STRENGTH:
        raise ValueError("Post-temporal Spatial reference strength must be 1.0.")

    if not isinstance(value.video_state, dict):
        raise ValueError("Post-temporal Spatial video_state must be a LATENT dict.")
    if value.video_state.get("samples") is not source.upscaled_video_latent:
        raise ValueError("Post-temporal Spatial video_state must retain the exact upscaled video tensor.")
    state = get_official_state(value.video_state)
    operations = list(state.get("operations", ()))
    user_types = tuple(
        "VideoConditionByLatentIndex" if position == 0 else "VideoConditionByKeyframeIndex"
        for position in expected_user_positions
    )
    carry_types = ("VideoConditionByKeyframeIndex",) * len(value.carry_positions)
    expected_types = (*user_types, *carry_types, "VideoConditionByReferenceLatent")
    actual_types = tuple(str(operation.get("type")) for operation in operations)
    if actual_types != expected_types:
        raise ValueError(
            "Post-temporal Spatial conditioning order must be user guides -> carry keyframes -> reference."
        )

    carry_start = len(user_types)
    carry_operations = operations[carry_start : carry_start + len(carry_types)]
    for operation, position in zip(carry_operations, value.carry_positions):
        if int(operation.get("frame_idx", -1)) != int(position):
            raise ValueError("A post-temporal Spatial carry keyframe has the wrong frame index.")
        if float(operation.get("strength", -1.0)) != float(source.keyframe_strength):
            raise ValueError("A post-temporal Spatial carry keyframe has the wrong strength.")

    reference = operations[-1]
    if int(reference.get("downscale_factor", -1)) != int(value.reference_downscale_factor):
        raise ValueError("Post-temporal Spatial reference operation has the wrong downscale factor.")
    if int(reference.get("temporal_scale_factor", -1)) != 1:
        raise ValueError("Post-temporal Spatial reference must preserve temporal resolution.")
    if float(reference.get("strength", -1.0)) != float(value.reference_strength):
        raise ValueError("Post-temporal Spatial reference operation has the wrong strength.")
    token_state = state.get("token_state")
    if not isinstance(token_state, dict):
        raise ValueError("Post-temporal Spatial conditioning did not materialize a token state.")
    if token_state.get("generated_keyframe_layout") is not None:
        raise ValueError("The terminal Spatial epilogue must not create generated-keyframe slots.")
    return value


def assemble_post_temporal_spatial_conditioning(
    upscaled_handoff: DFRPostTemporalSpatialUpscaledHandoff,
) -> DFRPostTemporalSpatialPreparedHandoff:
    """Build full-canvas epilogue conditioning in official operation order."""
    source = require_post_temporal_spatial_upscaled_handoff(upscaled_handoff)
    temporal = source.source_handoff
    plan = prepare_post_temporal_spatial_tile_plan(source)
    conditioning_fps = official_temporal_conditioning_fps(temporal.fps)
    temporal_position_scale = 2 ** int(temporal.completed_rounds)

    video_state: dict[str, Any] = {"samples": source.upscaled_video_latent}
    scaled_user_positions: list[int] = []
    for keyframe in source.upscaled_user_keyframes:
        scaled_position = int(keyframe.pixel_frame_index) * temporal_position_scale
        if not 0 <= scaled_position < int(temporal.padded_frames):
            raise ValueError(
                f"Scaled user keyframe {scaled_position} lies outside 0..{temporal.padded_frames - 1}."
            )
        if scaled_position == 0:
            video_state = apply_video_condition_by_latent_index(
                target_latent=video_state,
                conditioning_latent={"samples": keyframe.latent},
                strength=float(keyframe.strength),
                latent_idx=0,
            )
        else:
            video_state = apply_video_condition_by_keyframe_index(
                target_latent=video_state,
                conditioning_latent={"samples": keyframe.latent},
                frame_idx=scaled_position,
                strength=float(keyframe.strength),
                fps=conditioning_fps,
                num_pixel_frames=1,
            )
        scaled_user_positions.append(scaled_position)

    carry_positions = tuple(int(position) for position in temporal.carry_positions)
    for index, position in enumerate(carry_positions):
        video_state = apply_video_condition_by_keyframe_index(
            target_latent=video_state,
            conditioning_latent={
                "samples": source.upscaled_carry_keyframes[:, :, index : index + 1]
            },
            frame_idx=position,
            strength=float(source.keyframe_strength),
            fps=conditioning_fps,
            num_pixel_frames=1,
        )

    reference = temporal.video_latent
    expected_reference_shape = (
        int(source.upscaled_video_latent.shape[0]),
        int(source.upscaled_video_latent.shape[1]),
        int(source.upscaled_video_latent.shape[2]),
        int(source.upscaled_video_latent.shape[3]) // OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE,
        int(source.upscaled_video_latent.shape[4]) // OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE,
    )
    if tuple(int(value) for value in reference.shape) != expected_reference_shape:
        raise ValueError(
            "Post-temporal reference must be exactly half the target spatial latent resolution; "
            f"expected {expected_reference_shape}, got {tuple(reference.shape)}."
        )
    video_state = apply_video_condition_by_reference_latent(
        target_latent=video_state,
        reference_latent={"samples": reference},
        downscale_factor=OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE,
        temporal_scale_factor=1,
        strength=OFFICIAL_EPILOGUE_REFERENCE_STRENGTH,
        fps=conditioning_fps,
    )

    return require_post_temporal_spatial_prepared_handoff(
        DFRPostTemporalSpatialPreparedHandoff(
            version=POST_TEMPORAL_SPATIAL_PREPARED_VERSION,
            source_upscaled_handoff=source,
            tile_plan=plan,
            video_state=video_state,
            conditioning_fps=conditioning_fps,
            scaled_user_positions=tuple(scaled_user_positions),
            carry_positions=carry_positions,
            reference_downscale_factor=OFFICIAL_EPILOGUE_REFERENCE_DOWNSCALE,
            reference_strength=OFFICIAL_EPILOGUE_REFERENCE_STRENGTH,
        )
    )
