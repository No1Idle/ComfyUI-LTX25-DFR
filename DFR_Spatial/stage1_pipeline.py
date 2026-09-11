"""Consolidated public Stage-1 Spatial DFR execution path."""

from __future__ import annotations

import logging
from dataclasses import replace
from .dfr_layout import validate_layout
from typing import Any

import torch

from . import av_noiser
from .av_execution import execute_stage1_av_denoising_loop
from .dfr_sigmas import stage_1_sigmas
from .stage1_profile import Stage1RuntimeProfiler
from .stage2_handoff import DFRStage2Handoff, prepare_stage1_handoff


# Temporary internal instrumentation requested while Stage-1 optimization is evaluated.
PROFILE_CONSOLIDATED_STAGE1 = False


def run_stage1_spatial_dfr(
    model: Any,
    positive: Any,
    video_official_state: dict[str, Any],
    audio_official_state: dict[str, Any],
    *,
    seed: int,
    negative: Any = None,
    sigmas: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
) -> DFRStage2Handoff:
    """Noise, denoise, extract, and freeze the complete Stage-1 boundary."""
    profile_device = av_noiser._preferred_comfy_device(video_official_state["samples"].device)
    profiler = Stage1RuntimeProfiler(enabled=PROFILE_CONSOLIDATED_STAGE1, device=profile_device)
    custom_sigmas = sigmas is not None
    selected_sigmas = sigmas if custom_sigmas else stage_1_sigmas()

    with profiler.section("noising + state preparation"):
        noised_video, noised_audio, _noiser_stats = av_noiser.materialize_stage1_av_gaussian_noised_states(
            video_official_state,
            audio_official_state,
            seed=int(seed),
            noise_scale=1.0,
            device=profile_device,
        )
    final_video, final_audio, base_latent, generated_keyframes, _report = (
        execute_stage1_av_denoising_loop(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_video,
            noised_audio_state=noised_audio,
            sigmas=selected_sigmas,
            cfg_scale=float(cfg_scale),
            require_official_sigmas=not custom_sigmas,
            collect_diagnostics=False,
            profiler=profiler,
        )
    )
    with profiler.section("final extraction + handoff"):
        handoff = prepare_stage1_handoff(
            final_video,
            final_audio,
            base_latent,
            generated_keyframes,
        )
    if video_official_state.get("dfr_layout") is not None:
        handoff = replace(handoff, dfr_layout=dict(validate_layout(video_official_state["dfr_layout"])))
    if profiler.enabled:
        logging.warning("\n%s", profiler.format_report())
    return handoff
