"""Consolidated public Stage-2 Spatial DFR execution path."""

from __future__ import annotations

from typing import Any
from dataclasses import replace
from .stage2_temporal_tiling import stage2_temporal_tiling

import torch

from .dfr_sigmas import stage_2_sigmas
from .feed_forward_chunking import stage2_feed_forward_chunking
from .output_chunking import stage2_output_chunking
from .compact_timestep import stage2_compact_timestep
from .stage2_loop import execute_stage2_av_denoising_loop_from_upscaled_handoff
from .stage2_result import DFRStage2ResultHandoff, prepare_stage2_result_handoff
from .stage2_spatial import DFRUpscaledStage1Handoff


def run_stage2_spatial_dfr(
    model: Any,
    positive: Any,
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    stage_2_video_state: dict[str, Any],
    *,
    negative: Any = None,
    sigmas: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    ff_chunk_tokens: int = 4096,
    diagnose_memory: bool = False,
    output_chunk_tokens: int = 4096,
    compact_timesteps: bool = True,
    temporal_tiles: int = 1,
) -> DFRStage2ResultHandoff:
    """Noise, denoise, extract, and freeze the complete Stage-2 boundary."""
    custom_sigmas = sigmas is not None
    selected_sigmas = sigmas if custom_sigmas else stage_2_sigmas()
    with stage2_temporal_tiling(model, upscaled_stage_1_handoff, stage_2_video_state, temporal_tiles) as temporal_plan, stage2_compact_timestep(model, compact_timesteps), stage2_output_chunking(model, output_chunk_tokens), stage2_feed_forward_chunking(model, ff_chunk_tokens, diagnose=diagnose_memory):
        final_video, final_audio, base_latent, _report = (
            execute_stage2_av_denoising_loop_from_upscaled_handoff(
                model=model,
                positive=positive,
                negative=negative,
                upscaled_stage_1_handoff=upscaled_stage_1_handoff,
                stage_2_video_state=stage_2_video_state,
                sigmas=selected_sigmas,
                cfg_scale=float(cfg_scale),
                require_official_sigmas=not custom_sigmas,
                collect_diagnostics=False,
            )
        )
    result = prepare_stage2_result_handoff(
        upscaled_stage_1_handoff=upscaled_stage_1_handoff,
        final_stage_2_video_state=final_video,
        final_stage_2_audio_state=final_audio,
        stage_2_base_latent=base_latent,
        sigma_schedule=selected_sigmas,
        used_official_sigmas=not custom_sigmas,
    )
    if temporal_plan is not None:
        result = replace(result, temporal_seams=temporal_plan.tile_plan.pixel_seams,
                         temporal_tiles=len(temporal_plan.tile_plan.tiles))
    return result
