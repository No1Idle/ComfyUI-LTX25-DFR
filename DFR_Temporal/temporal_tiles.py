"""Temporal DFR seam-window planning.

Oracle reference:
  packages/ltx-pipelines/src/ltx_pipelines/dfr_layout.py::split_canvas_at_seams
  packages/ltx-pipelines/src/ltx_pipelines/dfr_layout.py::TemporalTilePlan
  packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py (temporal round loop)

This phase ports the official temporal seam-window construction into DFR_Temporal,
keeping all temporal-specific logic out of DFR_Spatial.  It consumes the x2
video-latent handoff from T2 and emits explicit tile windows carrying:
- latent interval and dropped lead-in (``left_ramp``);
- pixel-frame extents;
- global and local anchor positions;
- global and local generated-slot positions.

The implementation intentionally matches the Lightricks DFR logic:
- boundaries are the known seam cells at latent positions ``[0, *seams]``;
- overlap is one full canvas segment in latent cells plus the shared seam cell;
- leftover segments go to the leading tiles;
- each non-first tile contains a lead-in context prefix that is later dropped
  instead of blended.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.decoder_official_tiling import DimensionInterval
    from ..DFR_Spatial.dfr_layout import pixel_to_latent_index
else:  # Standalone temporal regression tests.
    from DFR_Spatial.decoder_official_tiling import DimensionInterval
    from DFR_Spatial.dfr_layout import pixel_to_latent_index

from .temporal_upscale import DFRTemporalUpscaledHandoff, require_temporal_upscaled_handoff


TEMPORAL_TILE_PLAN_VERSION = 1


@dataclass(frozen=True)
class DFRTemporalTileWindow:
    """One official Temporal-DFR tile window.

    ``interval.left_ramp`` is the lead-in latent prefix denoised for context but
    dropped when stitching. ``pixel_end`` is inclusive, matching the oracle.
    """

    interval: DimensionInterval
    pixel_start: int
    pixel_end: int
    local_frames: int
    anchor_global: tuple[int, ...]
    anchor_local: tuple[int, ...]
    slot_global: tuple[int, ...]
    slot_local: tuple[int, ...]


@dataclass(frozen=True)
class DFRTemporalTilePlanHandoff:
    """Serializable Temporal-DFR seam-window plan for one temporal round."""

    version: int
    source_upscaled_handoff: DFRTemporalUpscaledHandoff
    seam_positions: tuple[int, ...]
    requested_num_tiles: int
    effective_num_tiles: int
    latent_length: int
    overlap_latent_cells: int
    tiles: tuple[DFRTemporalTileWindow, ...]


def _split_canvas_at_seams(
    seams: tuple[int, ...],
    num_tiles: int,
    overlap: int,
    dim_size: int,
) -> list[DimensionInterval]:
    """Exact port of official ``split_canvas_at_seams`` / ``split_at_seams``.

    ``seams`` are the latent-space known-boundary cells beginning with 0.
    ``num_tiles`` larger than the number of segments is clamped.  Every tile but
    the first starts ``overlap`` cells before the cell it resumes after; that
    run-up becomes ``left_ramp`` and is later discarded instead of blended.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if len(seams) < 2 or seams[0] != 0:
        raise ValueError(f"seams must start at 0 and contain at least one segment, got {list(seams)}")
    if any(right <= left for left, right in itertools.pairwise(seams)):
        raise ValueError(f"seams must be strictly increasing, got {list(seams)}")
    if seams[-1] != dim_size - 1:
        raise ValueError(f"seams must end at the last latent cell ({dim_size - 1}), got {seams[-1]}")

    n_segments = len(seams) - 1
    effective_num_tiles = min(int(num_tiles), n_segments)
    base, leftover = divmod(n_segments, effective_num_tiles)
    counts = [base + (1 if idx < leftover else 0) for idx in range(effective_num_tiles)]

    intervals: list[DimensionInterval] = []
    cursor = 0
    for tile_index, count in enumerate(counts):
        resume = seams[cursor] + 1
        start = 0 if tile_index == 0 else max(0, resume - overlap)
        cursor += count
        intervals.append(
            DimensionInterval(
                start=start,
                end=seams[cursor] + 1,
                left_ramp=0 if tile_index == 0 else resume - start,
                right_ramp=0,
            )
        )
    return intervals


def _build_plan(
    *,
    seam_positions: tuple[int, ...],
    num_frames: int,
    num_tiles: int,
    temporal_scale: int,
) -> tuple[tuple[DFRTemporalTileWindow, ...], int, int]:
    if not seam_positions:
        raise ValueError("TemporalTilePlan requires at least one seam position.")
    latent_seams = tuple(pixel_to_latent_index(position, temporal_scale) for position in seam_positions)
    seams = (0, *latent_seams)
    latent_length = (int(num_frames) - 1) // int(temporal_scale) + 1
    overlap_latent_cells = (seams[1] - seams[0]) + 1 if len(seams) > 1 else 0
    intervals = _split_canvas_at_seams(seams, int(num_tiles), overlap_latent_cells, latent_length)

    tiles: list[DFRTemporalTileWindow] = []
    for interval in intervals:
        pixel_start = int(interval.start) * int(temporal_scale)
        pixel_end = (int(interval.end) - 1) * int(temporal_scale)
        local_frames = (int(interval.end) - int(interval.start) - 1) * int(temporal_scale) + 1
        anchor_global = tuple(position for position in seam_positions if pixel_start <= position <= pixel_end)
        anchor_local = tuple(int(position) - pixel_start for position in anchor_global)
        marks = [pixel_start, *[position for position in seam_positions if pixel_start < position <= pixel_end]]
        slot_global = tuple((left + right) // 2 for left, right in itertools.pairwise(marks))
        slot_local = tuple(int(position) - pixel_start for position in slot_global)
        tiles.append(
            DFRTemporalTileWindow(
                interval=interval,
                pixel_start=pixel_start,
                pixel_end=pixel_end,
                local_frames=local_frames,
                anchor_global=anchor_global,
                anchor_local=anchor_local,
                slot_global=slot_global,
                slot_local=slot_local,
            )
        )
    return tuple(tiles), latent_length, overlap_latent_cells


def require_temporal_tile_plan_handoff(value: Any) -> DFRTemporalTilePlanHandoff:
    if not isinstance(value, DFRTemporalTilePlanHandoff):
        raise ValueError(f"Expected DFRTemporalTilePlanHandoff, got {type(value).__name__}.")
    if int(value.version) != TEMPORAL_TILE_PLAN_VERSION:
        raise ValueError(f"Unsupported TemporalTilePlan version {value.version}.")
    source = require_temporal_upscaled_handoff(value.source_upscaled_handoff)
    expected_num_tiles = 2 ** int(source.round_index)
    if int(value.requested_num_tiles) != expected_num_tiles:
        raise ValueError(
            f"TemporalTilePlan requested_num_tiles={value.requested_num_tiles}, expected {expected_num_tiles}."
        )
    if tuple(int(x) for x in value.seam_positions) != tuple(int(x) for x in source.seam_positions):
        raise ValueError(
            "TemporalTilePlan seam_positions do not match the upscaled handoff: "
            f"plan={value.seam_positions}, source={source.seam_positions}."
        )
    expected_tiles, expected_latent_length, expected_overlap = _build_plan(
        seam_positions=tuple(int(x) for x in source.seam_positions),
        num_frames=int(source.target_padded_frames),
        num_tiles=expected_num_tiles,
        temporal_scale=int(source.source_handoff.temporal_scale),
    )
    if int(value.latent_length) != expected_latent_length:
        raise ValueError(f"TemporalTilePlan latent_length={value.latent_length}, expected {expected_latent_length}.")
    if int(value.overlap_latent_cells) != expected_overlap:
        raise ValueError(
            f"TemporalTilePlan overlap_latent_cells={value.overlap_latent_cells}, expected {expected_overlap}."
        )
    if int(value.effective_num_tiles) != len(expected_tiles):
        raise ValueError(
            f"TemporalTilePlan effective_num_tiles={value.effective_num_tiles}, expected {len(expected_tiles)}."
        )
    if len(value.tiles) != len(expected_tiles):
        raise ValueError(f"TemporalTilePlan tile count={len(value.tiles)}, expected {len(expected_tiles)}.")
    if value.tiles != expected_tiles:
        raise ValueError("TemporalTilePlan tiles do not match the official seam-window construction.")
    return value


def prepare_temporal_tile_plan(
    temporal_upscaled_handoff: DFRTemporalUpscaledHandoff,
) -> tuple[DFRTemporalTilePlanHandoff, str]:
    """Build the official temporal seam-window plan for the current round.

    For round ``r`` the oracle requests ``2**r`` windows, then clamps at the
    number of available seam segments.  With our current T2 boundary this yields
    two windows for round 1 on the usual 97->193-frame canvas.
    """
    source = require_temporal_upscaled_handoff(temporal_upscaled_handoff)
    round_index = int(source.round_index)
    requested_num_tiles = 2 ** round_index
    tiles, latent_length, overlap_latent_cells = _build_plan(
        seam_positions=tuple(int(x) for x in source.seam_positions),
        num_frames=int(source.target_padded_frames),
        num_tiles=requested_num_tiles,
        temporal_scale=int(source.source_handoff.temporal_scale),
    )
    result = DFRTemporalTilePlanHandoff(
        version=TEMPORAL_TILE_PLAN_VERSION,
        source_upscaled_handoff=source,
        seam_positions=tuple(int(x) for x in source.seam_positions),
        requested_num_tiles=requested_num_tiles,
        effective_num_tiles=len(tiles),
        latent_length=latent_length,
        overlap_latent_cells=overlap_latent_cells,
        tiles=tiles,
    )
    result = require_temporal_tile_plan_handoff(result)

    tile_parts = []
    for index, tile in enumerate(result.tiles):
        tile_parts.append(
            f"tile{index + 1}="
            f"latent[{tile.interval.start}:{tile.interval.end})/drop={tile.interval.left_ramp};"
            f"pixel[{tile.pixel_start}:{tile.pixel_end}];"
            f"anchors={tile.anchor_global};slots={tile.slot_global}"
        )
    report = (
        "PASS=True; stage=T3_temporal_tile_plan; "
        f"round={round_index}; requested_num_tiles={requested_num_tiles}; effective_num_tiles={len(result.tiles)}; "
        f"source={source.source_handoff.source_stage}; target_frames={source.target_padded_frames}; "
        f"latent_length={latent_length}; overlap_latent_cells={overlap_latent_cells}; "
        + " | ".join(tile_parts)
    )
    return result, report
