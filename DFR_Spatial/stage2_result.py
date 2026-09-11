"""Compact production boundary after the complete Spatial-DFR Stage-2 loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_handoff import DFRStage2DecoderHandoff, prepare_stage2_decoder_handoff
from .stage2_spatial import DFRUpscaledStage1Handoff, require_upscaled_stage1_handoff


STAGE2_RESULT_HANDOFF_VERSION = 1


@dataclass(frozen=True)
class DFRStage2ResultHandoff:
    """Only the tensors and metadata needed after Stage-2 denoising."""

    version: int
    padded_video_latent: torch.Tensor
    stage_1_audio_latent: torch.Tensor
    fps: float
    duration_seconds: float
    decoder_handoff: DFRStage2DecoderHandoff
    sigma_schedule: torch.Tensor
    used_official_sigmas: bool
    temporal_seams: tuple[int, ...] = ()
    temporal_tiles: int = 1
    dfr_layout: dict[str, Any] | None = None


def require_stage2_result_handoff(value: Any) -> DFRStage2ResultHandoff:
    if not isinstance(value, DFRStage2ResultHandoff):
        raise ValueError(f"Expected DFRStage2ResultHandoff, got {type(value).__name__}.")
    if int(value.version) != STAGE2_RESULT_HANDOFF_VERSION:
        raise ValueError(f"Unsupported Stage-2 result handoff version {value.version}.")
    if not torch.is_tensor(value.padded_video_latent) or value.padded_video_latent.ndim != 5:
        raise ValueError("Stage-2 result video must be a [B,C,T,H,W] tensor.")
    if not torch.is_tensor(value.stage_1_audio_latent) or value.stage_1_audio_latent.ndim != 4:
        raise ValueError("Stage-2 result audio must be a [B,C,T,F] tensor.")
    if not isinstance(value.decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError("Stage-2 result does not contain a valid final-decoder handoff.")
    if not torch.is_tensor(value.sigma_schedule) or value.sigma_schedule.ndim != 1:
        raise ValueError("Stage-2 result sigma schedule must be a rank-1 tensor.")
    if float(value.fps) <= 0.0:
        raise ValueError(f"Stage-2 result fps must be positive, got {value.fps}.")
    return value


def prepare_stage2_result_handoff(
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    final_stage_2_video_state: dict[str, Any],
    final_stage_2_audio_state: dict[str, Any],
    stage_2_base_latent: dict[str, Any],
    sigma_schedule: torch.Tensor,
    *,
    used_official_sigmas: bool,
) -> DFRStage2ResultHandoff:
    """Freeze Stage-2 outputs without retaining the large final token states."""
    upstream = require_upscaled_stage1_handoff(upscaled_stage_1_handoff)
    base = stage_2_base_latent.get("samples") if isinstance(stage_2_base_latent, dict) else None
    if not torch.is_tensor(base) or base.ndim != 5:
        raise ValueError("stage_2_base_latent must contain [B,C,T,H,W] tensor 'samples'.")

    decoder_handoff, _generated_keyframes, _report = prepare_stage2_decoder_handoff(
        final_stage_2_video_state,
        final_stage_2_audio_state,
        collect_diagnostics=False,
    )
    result = DFRStage2ResultHandoff(
        version=STAGE2_RESULT_HANDOFF_VERSION,
        padded_video_latent=base,
        stage_1_audio_latent=upstream.stage_1_handoff.stage_1_audio_latent,
        fps=float(upstream.stage_1_handoff.fps),
        duration_seconds=float(upstream.stage_1_handoff.duration_seconds),
        decoder_handoff=decoder_handoff,
        sigma_schedule=sigma_schedule.detach().to(device="cpu", dtype=torch.float32).clone(),
        used_official_sigmas=bool(used_official_sigmas),
        temporal_seams=tuple(getattr(upstream.stage_1_handoff, "temporal_seams", ())),
        temporal_tiles=int(getattr(upstream.stage_1_handoff, "temporal_tiles", 1)),
        dfr_layout=(dict(upstream.stage_1_handoff.dfr_layout) if getattr(upstream.stage_1_handoff, "dfr_layout", None) is not None else None),
    )
    return require_stage2_result_handoff(result)
