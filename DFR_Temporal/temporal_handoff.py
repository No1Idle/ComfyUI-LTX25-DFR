"""Temporal-DFR source handoff normalization.

Oracle reference:
  packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py

Official Temporal DFR starts after Spatial Stage 2 with:
- the current base video latent;
- the generated DFR keyframes from the previous round;
- their pixel-frame positions;
- the frozen Stage-1 audio latent;
- the shared Gaussian-noiser generator at its current continuation point.

For experimentation we intentionally support the same normalized boundary directly
from Spatial Stage 1.  That branch is not an upstream DFR mode, but it preserves the
same data contract so every later Temporal-DFR operation can be shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.dfr_layout import validate_layout
    from ..DFR_Spatial.stage2_handoff import DFRStage2Handoff
    from ..DFR_Spatial.stage2_result import DFRStage2ResultHandoff, require_stage2_result_handoff
else:  # Standalone temporal regression tests.
    from DFR_Spatial.dfr_layout import validate_layout
    from DFR_Spatial.stage2_handoff import DFRStage2Handoff
    from DFR_Spatial.stage2_result import DFRStage2ResultHandoff, require_stage2_result_handoff

from .temporal_user import (
    DFRTemporalUserKeyframe,
    capture_temporal_user_keyframes,
    validate_temporal_user_keyframes,
)


TEMPORAL_HANDOFF_VERSION = 2
TEMPORAL_SOURCE_STAGE1 = "stage1"
TEMPORAL_SOURCE_STAGE2 = "stage2"


@dataclass(frozen=True)
class DFRTemporalHandoff:
    """Normalized input boundary for the first Temporal-DFR round.

    ``source_stage='stage2'`` is the official-parity entry point.
    ``source_stage='stage1'`` is the explicitly experimental entry point requested
    for motion-quality testing without Spatial Stage 2.
    """

    version: int
    source_stage: str
    completed_rounds: int
    video_latent: torch.Tensor
    carry_keyframes: torch.Tensor
    carry_positions: tuple[int, ...]
    # Boundaries used by the official temporal-window tiler in the most recently
    # completed round.  This is intentionally only the scaled pre-round carry
    # set, not every keyframe in the merged post-round carry bag.
    last_window_seams: tuple[int, ...]
    user_keyframes: tuple[DFRTemporalUserKeyframe, ...]
    stage_1_audio_latent: torch.Tensor
    rng_state_before_temporal: torch.Tensor
    rng_device_type: str
    seed: int
    fps: float
    duration_seconds: float
    temporal_scale: int
    requested_frames: int
    padded_frames: int
    # Once the post-temporal Spatial x2 epilogue has run, another temporal round
    # would no longer match the supported ordering of the DFR pipeline.
    post_temporal_spatial_completed: bool


def _validate_common_geometry(
    *,
    video: torch.Tensor,
    keyframes: torch.Tensor,
    positions: tuple[int, ...],
    audio: torch.Tensor,
    temporal_scale: int,
    requested_frames: int,
    padded_frames: int,
) -> None:
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("Temporal handoff video must be a [B,C,T,H,W] tensor.")
    if not torch.is_tensor(keyframes) or keyframes.ndim != 5:
        raise ValueError("Temporal handoff keyframes must be a [B,C,K,H,W] tensor.")
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError("Temporal handoff Stage-1 audio must be a [B,C,T,F] tensor.")
    if temporal_scale <= 0:
        raise ValueError(f"temporal_scale must be positive, got {temporal_scale}.")
    if requested_frames <= 0 or padded_frames <= 0 or requested_frames > padded_frames:
        raise ValueError(
            f"Invalid requested/padded frame counts: requested={requested_frames}, padded={padded_frames}."
        )

    expected_padded = (int(video.shape[2]) - 1) * temporal_scale + 1
    if expected_padded != padded_frames:
        raise ValueError(
            "Temporal handoff video does not match the DFR padded canvas: "
            f"video implies {expected_padded} pixel frames, layout says {padded_frames}."
        )
    if (requested_frames - 1) % temporal_scale != 0 or (padded_frames - 1) % temporal_scale != 0:
        raise ValueError("Temporal handoff frame counts must lie on the LTX temporal latent grid.")

    if int(keyframes.shape[2]) != len(positions):
        raise ValueError(
            "Carry keyframe count must exactly match carry_positions: "
            f"keyframes={int(keyframes.shape[2])}, positions={len(positions)}."
        )
    if not positions:
        raise ValueError("Temporal DFR requires at least one carry-forward generated keyframe.")
    if positions[0] <= 0 or any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError(f"carry_positions must be positive, strictly increasing and unique, got {positions}.")
    if positions[-1] >= padded_frames:
        raise ValueError(
            f"carry_positions must stay inside padded canvas 0..{padded_frames - 1}, got {positions[-1]}."
        )
    if positions[-1] != padded_frames - 1:
        raise ValueError(
            "Official Temporal DFR requires the final carry-forward keyframe to be the padded "
            f"canvas endpoint ({padded_frames - 1}); got {positions[-1]}."
        )
    off_grid = tuple(position for position in positions if position % temporal_scale != 0)
    if off_grid:
        raise ValueError(
            f"carry_positions must lie on x{temporal_scale} latent borders; invalid={off_grid}."
        )

    if int(video.shape[0]) != int(keyframes.shape[0]) or int(video.shape[1]) != int(keyframes.shape[1]):
        raise ValueError("Temporal handoff video/keyframes batch and channel dimensions must match.")
    if tuple(video.shape[-2:]) != tuple(keyframes.shape[-2:]):
        raise ValueError(
            "Temporal handoff video/keyframes spatial latent geometry must match: "
            f"video={tuple(video.shape[-2:])}, keyframes={tuple(keyframes.shape[-2:])}."
        )


def require_temporal_handoff(value: Any) -> DFRTemporalHandoff:
    if not isinstance(value, DFRTemporalHandoff):
        raise ValueError(f"Expected DFRTemporalHandoff, got {type(value).__name__}.")
    if int(value.version) != TEMPORAL_HANDOFF_VERSION:
        raise ValueError(f"Unsupported Temporal-DFR handoff version {value.version}.")
    if value.source_stage not in (TEMPORAL_SOURCE_STAGE1, TEMPORAL_SOURCE_STAGE2):
        raise ValueError(f"Unsupported Temporal-DFR source_stage {value.source_stage!r}.")
    if int(value.completed_rounds) < 0:
        raise ValueError(f"completed_rounds must be non-negative, got {value.completed_rounds}.")
    if not isinstance(value.post_temporal_spatial_completed, bool):
        raise ValueError("post_temporal_spatial_completed must be a bool.")
    if value.post_temporal_spatial_completed and int(value.completed_rounds) < 1:
        raise ValueError("Post-temporal Spatial x2 requires at least one completed temporal round.")
    if not torch.is_tensor(value.rng_state_before_temporal):
        raise ValueError("Temporal handoff must contain a torch RNG continuation state tensor.")
    if float(value.fps) <= 0.0:
        raise ValueError(f"Temporal handoff fps must be positive, got {value.fps}.")
    if float(value.duration_seconds) <= 0.0:
        raise ValueError(
            f"Temporal handoff duration_seconds must be positive, got {value.duration_seconds}."
        )
    _validate_common_geometry(
        video=value.video_latent,
        keyframes=value.carry_keyframes,
        positions=tuple(int(x) for x in value.carry_positions),
        audio=value.stage_1_audio_latent,
        temporal_scale=int(value.temporal_scale),
        requested_frames=int(value.requested_frames),
        padded_frames=int(value.padded_frames),
    )
    seams = tuple(int(position) for position in value.last_window_seams)
    if int(value.completed_rounds) == 0:
        if seams:
            raise ValueError(
                "An entry Temporal handoff cannot have last_window_seams before a round has completed."
            )
    else:
        if not seams:
            raise ValueError("A completed Temporal round must retain its last_window_seams.")
        if seams[0] <= 0 or any(right <= left for left, right in zip(seams, seams[1:])):
            raise ValueError(
                f"last_window_seams must be positive, strictly increasing and unique, got {seams}."
            )
        carry_position_set = set(int(position) for position in value.carry_positions)
        missing_seams = tuple(position for position in seams if position not in carry_position_set)
        if missing_seams:
            raise ValueError(
                "last_window_seams must be a subset of the complete carry_positions; "
                f"missing={missing_seams}."
            )
        if seams[-1] != int(value.padded_frames) - 1:
            raise ValueError(
                "The last temporal window seam must be the padded canvas endpoint "
                f"({int(value.padded_frames) - 1}); got {seams[-1]}."
            )
        off_grid_seams = tuple(
            position for position in seams if position % int(value.temporal_scale) != 0
        )
        if off_grid_seams:
            raise ValueError(
                f"last_window_seams must lie on x{value.temporal_scale} latent borders; "
                f"invalid={off_grid_seams}."
            )
    validate_temporal_user_keyframes(
        tuple(value.user_keyframes),
        source_video=value.video_latent,
    )
    return value


def _validated_layout_fields(dfr_layout: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]]:
    layout = validate_layout(dfr_layout)
    positions = tuple(int(position) for position in layout["pixel_frame_indices"])
    return layout, positions


def _build_handoff(
    *,
    source_stage: str,
    video: torch.Tensor,
    keyframes: torch.Tensor,
    positions: tuple[int, ...],
    user_conditioning_state: dict[str, Any] | None,
    audio: torch.Tensor,
    rng_state: torch.Tensor,
    rng_device_type: str,
    seed: int,
    fps: float,
    duration_seconds: float,
    layout: dict[str, Any],
) -> DFRTemporalHandoff:
    user_keyframes = capture_temporal_user_keyframes(
        user_conditioning_state,
        source_video=video,
    )
    handoff = DFRTemporalHandoff(
        version=TEMPORAL_HANDOFF_VERSION,
        source_stage=source_stage,
        completed_rounds=0,
        video_latent=video,
        carry_keyframes=keyframes,
        carry_positions=positions,
        last_window_seams=(),
        user_keyframes=user_keyframes,
        stage_1_audio_latent=audio,
        rng_state_before_temporal=rng_state.detach().cpu().clone(),
        rng_device_type=str(rng_device_type),
        seed=int(seed),
        fps=float(fps),
        duration_seconds=float(duration_seconds),
        temporal_scale=int(layout["temporal_scale"]),
        requested_frames=int(layout["requested_frames"]),
        padded_frames=int(layout["padded_frames"]),
        post_temporal_spatial_completed=False,
    )
    return require_temporal_handoff(handoff)


def prepare_temporal_handoff_from_stage1(
    stage_1_handoff: DFRStage2Handoff,
    dfr_layout: dict[str, Any],
    user_conditioning_state: dict[str, Any] | None = None,
) -> tuple[DFRTemporalHandoff, str]:
    """Experimental Stage-1 -> Temporal-DFR boundary.

    The source tensors/RNG are exactly the frozen Stage-1 boundary already used
    by Spatial Stage 2.  No spatial upscaling and no tensor copies are performed
    except for the small CPU RNG state capture.
    """
    if not isinstance(stage_1_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_1_handoff).__name__}.")
    layout, positions = _validated_layout_fields(dfr_layout)
    handoff = _build_handoff(
        source_stage=TEMPORAL_SOURCE_STAGE1,
        video=stage_1_handoff.reserved_half_res_video,
        keyframes=stage_1_handoff.stage_1_generated_keyframes,
        positions=positions,
        user_conditioning_state=user_conditioning_state,
        audio=stage_1_handoff.stage_1_audio_latent,
        rng_state=stage_1_handoff.rng_state_after_stage1_av,
        rng_device_type=stage_1_handoff.rng_device_type,
        seed=stage_1_handoff.seed,
        fps=stage_1_handoff.fps,
        duration_seconds=stage_1_handoff.duration_seconds,
        layout=layout,
    )
    report = (
        "PASS=True; stage=T1_temporal_handoff; source=stage1_experimental; "
        f"completed_rounds={handoff.completed_rounds}; seed={handoff.seed}; fps={handoff.fps:.9g}; "
        f"requested_frames={handoff.requested_frames}; "
        f"padded_frames={handoff.padded_frames}; carry_positions={handoff.carry_positions}; "
        f"user_keyframes={len(handoff.user_keyframes)}; "
        f"video_shape={tuple(int(x) for x in handoff.video_latent.shape)}; "
        f"keyframes_shape={tuple(int(x) for x in handoff.carry_keyframes.shape)}; "
        f"audio_shape={tuple(int(x) for x in handoff.stage_1_audio_latent.shape)}."
    )
    return handoff, report


def prepare_temporal_handoff_from_stage2(
    stage_2_handoff: DFRStage2ResultHandoff,
    dfr_layout: dict[str, Any],
    user_conditioning_state: dict[str, Any] | None = None,
) -> tuple[DFRTemporalHandoff, str]:
    """Official Spatial Stage-2 -> Temporal-DFR boundary."""
    stage2 = require_stage2_result_handoff(stage_2_handoff)
    layout, positions = _validated_layout_fields(dfr_layout)
    decoder = stage2.decoder_handoff
    handoff = _build_handoff(
        source_stage=TEMPORAL_SOURCE_STAGE2,
        video=stage2.padded_video_latent,
        keyframes=decoder.stage_2_generated_keyframes,
        positions=positions,
        user_conditioning_state=user_conditioning_state,
        audio=stage2.stage_1_audio_latent,
        rng_state=decoder.rng_state_after_stage2_av,
        rng_device_type=decoder.rng_device_type,
        seed=decoder.seed,
        fps=stage2.fps,
        duration_seconds=stage2.duration_seconds,
        layout=layout,
    )
    report = (
        "PASS=True; stage=T1_temporal_handoff; source=stage2_official; "
        f"completed_rounds={handoff.completed_rounds}; seed={handoff.seed}; fps={handoff.fps:.9g}; "
        f"requested_frames={handoff.requested_frames}; "
        f"padded_frames={handoff.padded_frames}; carry_positions={handoff.carry_positions}; "
        f"user_keyframes={len(handoff.user_keyframes)}; "
        f"video_shape={tuple(int(x) for x in handoff.video_latent.shape)}; "
        f"keyframes_shape={tuple(int(x) for x in handoff.carry_keyframes.shape)}; "
        f"audio_shape={tuple(int(x) for x in handoff.stage_1_audio_latent.shape)}."
    )
    return handoff, report
