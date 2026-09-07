"""Consolidated production entry point for the Spatial-DFR DiffVAE decoder."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .decoder_det_dual_stream_exact import (
    execute_exact_deterministic_stages,
    prepare_exact_deterministic_dual_stream_port,
)
from .decoder_det_stage_context import prepare_deterministic_decoder_keyframe_context
from .decoder_det_stage_official_preflight import prepare_official_deterministic_stage_preflight
from .decoder_full_tiled_exact import (
    _runtime_decoder_geometry,
    execute_official_tiled_decode,
    prepare_official_auto_tiled_decode_schedule,
    prepare_official_tiled_decode_schedule,
)
from .decoder_joint_attention import prepare_joint_decoder_attention
from .decoder_keyframe_checkpoint import extract_decoder_keyframe_checkpoint_weights
from .decoder_keyframe_substrate import build_decoder_keyframe_substrate
from .decoder_handoff import DECODER_HANDOFF_VERSION, DFRStage2DecoderHandoff
from .decoder_keyframes import build_final_decode_keyframes
from .decoder_official_tiling import (
    compute_tile_halos,
    recommended_pixel_overlaps,
    stage4_to_pixel_scale_factors,
)
from .decoder_stage5_dual_stream_exact import prepare_exact_stage5_dual_stream_port
from .dfr_layout import validate_layout
from .stage2_output import prepare_official_final_decode
from .stage2_handoff import DFRStage2Handoff, HANDOFF_VERSION
from .stage2_result import require_stage2_result_handoff


logger = logging.getLogger(__name__)


def _validated_final_video_latent(
    stage_2_handoff: Any,
    final_video_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> dict[str, Any]:
    handoff = require_stage2_result_handoff(stage_2_handoff)
    layout = validate_layout(dfr_layout)
    if not isinstance(final_video_latent, dict):
        raise ValueError(f"final_video_latent must be a LATENT dict, got {type(final_video_latent).__name__}.")
    samples = final_video_latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("final_video_latent must contain a [B,C,T,H,W] tensor named 'samples'.")

    padded = handoff.padded_video_latent
    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(padded.shape[2]) != expected_padded_t:
        raise ValueError(
            "stage_2_handoff and dfr_layout disagree: "
            f"handoff T={int(padded.shape[2])}, layout padded T={expected_padded_t}."
        )
    expected = padded[:, :, :requested_t]
    if tuple(samples.shape) != tuple(expected.shape):
        raise ValueError(
            f"final_video_latent shape {tuple(samples.shape)} does not match the Stage-2/layout result "
            f"{tuple(expected.shape)}."
        )
    if samples.device != expected.device or samples.dtype != expected.dtype:
        raise ValueError(
            "final_video_latent must retain the Stage-2 tensor device and dtype: "
            f"got {samples.device}/{samples.dtype}, expected {expected.device}/{expected.dtype}."
        )
    return final_video_latent


def _required_manual_overlaps(vae: Any) -> tuple[int, int]:
    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if decoder is None:
        raise ValueError("Connected VAE does not expose first_stage_model.decoder.")
    stage_kernels, stage5_kernel, strides, patch_size = _runtime_decoder_geometry(decoder)
    tile_halos = compute_tile_halos(
        stage_kernels[3],
        int(len(decoder.det_stages[3])),
        stage5_kernel,
        int(len(decoder.diff_blocks)),
        strides[3],
    )
    pixel_scale = stage4_to_pixel_scale_factors(strides[3], patch_size)
    return recommended_pixel_overlaps(tile_halos, pixel_scale)


def _prepare_stage1_decoder_inputs(
    stage_1_handoff: Any,
    final_video_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, Any], DFRStage2DecoderHandoff, Any]:
    """Adapt the frozen Stage-1 boundary to the validated decoder contract."""
    if not isinstance(stage_1_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_1_handoff).__name__}.")
    if int(stage_1_handoff.version) != HANDOFF_VERSION:
        raise ValueError(f"Unsupported Stage-1 handoff version {stage_1_handoff.version}.")

    layout = validate_layout(dfr_layout)
    if not isinstance(final_video_latent, dict):
        raise ValueError(f"final_video_latent must be a LATENT dict, got {type(final_video_latent).__name__}.")
    samples = final_video_latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("final_video_latent must contain a [B,C,T,H,W] tensor named 'samples'.")

    padded = stage_1_handoff.reserved_half_res_video
    if not torch.is_tensor(padded) or padded.ndim != 5:
        raise ValueError("stage_1_handoff must contain a [B,C,T,H,W] reserved video tensor.")
    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(padded.shape[2]) != expected_padded_t:
        raise ValueError(
            "stage_1_handoff and dfr_layout disagree: "
            f"handoff T={int(padded.shape[2])}, layout padded T={expected_padded_t}."
        )
    expected = padded[:, :, :requested_t]
    if tuple(samples.shape) != tuple(expected.shape):
        raise ValueError(
            f"final_video_latent shape {tuple(samples.shape)} does not match the Stage-1/layout result "
            f"{tuple(expected.shape)}."
        )
    if samples.device != expected.device or samples.dtype != expected.dtype:
        raise ValueError(
            "final_video_latent must retain the Stage-1 tensor device and dtype: "
            f"got {samples.device}/{samples.dtype}, expected {expected.device}/{expected.dtype}."
        )
    keyframes = stage_1_handoff.stage_1_generated_keyframes
    if not torch.is_tensor(keyframes) or keyframes.ndim != 5:
        raise ValueError("stage_1_handoff must contain [B,C,K,H,W] generated keyframes.")
    if int(keyframes.shape[0]) != int(samples.shape[0]):
        raise ValueError(
            "Stage-1 generated-keyframe batch does not match the reserved Stage-1 video: "
            f"keyframes={int(keyframes.shape[0])}, video={int(samples.shape[0])}."
        )
    if tuple(keyframes.shape[-2:]) != tuple(samples.shape[-2:]):
        raise ValueError(
            "Stage-1 generated keyframes must share the Stage-1 video latent spatial size: "
            f"keyframes={tuple(keyframes.shape[-2:])}, video={tuple(samples.shape[-2:])}."
        )
    rng_state = stage_1_handoff.rng_state_after_stage1_av
    if not torch.is_tensor(rng_state):
        raise ValueError("stage_1_handoff does not contain a valid post-Stage-1 AV RNG state.")

    # The decoder implementation consumes one common continuation package. At
    # this preview boundary its fields contain the Stage-1 slots and the RNG
    # state immediately after Stage-1 VIDEO then AUDIO noising.
    decoder_handoff = DFRStage2DecoderHandoff(
        version=DECODER_HANDOFF_VERSION,
        seed=int(stage_1_handoff.seed),
        rng_state_after_stage2_av=rng_state.detach().cpu().clone(),
        rng_device_type=str(stage_1_handoff.rng_device_type),
        stage_2_generated_keyframes=keyframes,
        generated_keyframe_count=int(keyframes.shape[2]),
        generated_keyframe_shape=tuple(int(v) for v in keyframes.shape),
    )
    final_decode_keyframes, _kept, _dropped, _report = build_final_decode_keyframes(
        decoder_handoff,
        layout,
    )
    return final_video_latent, decoder_handoff, final_decode_keyframes


def _decode_keyframe_aware_video(  # noqa: PLR0913
    final_video_latent: dict[str, Any],
    decoder_handoff: DFRStage2DecoderHandoff,
    final_decode_keyframes: Any,
    dfr_layout: dict[str, Any],
    vae: Any,
    vae_name: str,
    *,
    use_auto_tiling: bool = True,
    tile_frames: int = 104,
    tile_height: int = 416,
    tile_width: int = 544,
    log_stage: str = "Stage 2",
) -> torch.Tensor:
    substrate, _ = build_decoder_keyframe_substrate(
        vae,
        decoder_handoff,
        final_decode_keyframes,
        dfr_layout,
        collect_diagnostics=False,
    )
    checkpoint_weights, _ = extract_decoder_keyframe_checkpoint_weights(substrate, str(vae_name))
    joint_attention, _ = prepare_joint_decoder_attention(substrate, checkpoint_weights)
    deterministic_context, _ = prepare_deterministic_decoder_keyframe_context(
        vae,
        substrate,
        checkpoint_weights,
        joint_attention,
        final_decode_keyframes,
        dfr_layout,
    )
    preflight, _ = prepare_official_deterministic_stage_preflight(
        vae,
        deterministic_context,
        collect_diagnostics=False,
    )
    deterministic_port, _ = prepare_exact_deterministic_dual_stream_port(preflight, joint_attention)
    deterministic_execution, _ = execute_exact_deterministic_stages(
        vae,
        final_video_latent,
        deterministic_context,
        deterministic_port,
        collect_diagnostics=False,
    )
    stage5_port, _ = prepare_exact_stage5_dual_stream_port(deterministic_port)

    if bool(use_auto_tiling):
        schedule, _ = prepare_official_auto_tiled_decode_schedule(
            vae,
            deterministic_execution,
            deterministic_context,
        )
        tiling_mode = "auto"
    else:
        temporal_overlap, spatial_overlap = _required_manual_overlaps(vae)
        schedule, _ = prepare_official_tiled_decode_schedule(
            vae,
            deterministic_execution,
            deterministic_context,
            tile_frames=int(tile_frames),
            temporal_overlap=int(temporal_overlap),
            tile_height=int(tile_height),
            tile_width=int(tile_width),
            spatial_overlap=int(spatial_overlap),
        )
        tiling_mode = "manual"

    images, _execution, _ = execute_official_tiled_decode(
        vae,
        deterministic_execution,
        deterministic_context,
        decoder_handoff,
        stage5_port,
        schedule,
        collect_diagnostics=False,
    )
    config = schedule.tiling_config
    logger.info(
        "LTX Spatial DFR %s video decode: tiling=%s, frames=%d/%d, height=%d/%d, width=%d/%d, tiles=%d, groups=%d",
        log_stage,
        tiling_mode,
        int(config.frames.tile_size),
        int(config.frames.overlap),
        int(config.height.tile_size),
        int(config.height.overlap),
        int(config.width.tile_size),
        int(config.width.overlap),
        int(schedule.total_tiles),
        int(schedule.temporal_groups),
    )
    return images


def decode_spatial_dfr_video(  # noqa: PLR0913
    stage_2_handoff: Any,
    final_video_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
    vae: Any,
    vae_name: str,
    *,
    use_auto_tiling: bool = True,
    tile_frames: int = 104,
    tile_height: int = 416,
    tile_width: int = 544,
) -> torch.Tensor:
    """Run the complete validated final Stage-2 Spatial-DFR decode."""
    final_video_latent = _validated_final_video_latent(
        stage_2_handoff,
        final_video_latent,
        dfr_layout,
    )
    decoder_handoff, final_decode_keyframes = prepare_official_final_decode(stage_2_handoff, dfr_layout)
    return _decode_keyframe_aware_video(
        final_video_latent,
        decoder_handoff,
        final_decode_keyframes,
        dfr_layout,
        vae,
        vae_name,
        use_auto_tiling=use_auto_tiling,
        tile_frames=tile_frames,
        tile_height=tile_height,
        tile_width=tile_width,
        log_stage="Stage 2",
    )


def decode_stage1_spatial_dfr_video(  # noqa: PLR0913
    stage_1_handoff: Any,
    final_video_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
    vae: Any,
    vae_name: str,
    *,
    use_auto_tiling: bool = True,
    tile_frames: int = 104,
    tile_height: int = 416,
    tile_width: int = 544,
) -> torch.Tensor:
    """Decode a keyframe-aware Stage-1 preview from its exact AV RNG boundary."""
    final_video_latent, decoder_handoff, final_decode_keyframes = _prepare_stage1_decoder_inputs(
        stage_1_handoff,
        final_video_latent,
        dfr_layout,
    )
    return _decode_keyframe_aware_video(
        final_video_latent,
        decoder_handoff,
        final_decode_keyframes,
        dfr_layout,
        vae,
        vae_name,
        use_auto_tiling=use_auto_tiling,
        tile_frames=tile_frames,
        tile_height=tile_height,
        tile_width=tile_width,
        log_stage="Stage 1 preview",
    )
