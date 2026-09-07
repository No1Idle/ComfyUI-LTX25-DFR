"""Stage-2G joint AV post-process + deterministic Euler state advance.

This is the Stage-2 counterpart of the validated Stage-1 AV Euler bridge.
Strict Spatial-DFR parity uses the exact same per-modality post-process and Euler
algebra as Stage 1, but applied to the Stage-2 noised/raw-denoised states and the
exact 4-point / 3-transition Stage-2 sigma schedule.
"""

from __future__ import annotations

from typing import Any

import torch

from .av_step import stage1_av_euler_step_from_denoised, validate_stage1_av_euler_step
from .stage2_execution import resolve_stage2_shared_av_seed
from .stage2_noiser import _require_exact_stage2_sigmas


def stage2_av_euler_step_from_denoised(
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
    raw_denoised_stage_2_video_state: dict[str, Any],
    raw_denoised_stage_2_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
    *,
    allow_experimental_extra_step: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Advance both Stage-2 modalities through one exact post-process + Euler transition."""
    sigma_schedule = _require_exact_stage2_sigmas(
        sigmas,
        allow_experimental_extra_step=allow_experimental_extra_step,
    )
    shared_seed = resolve_stage2_shared_av_seed(
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
    )
    next_video, next_audio, post_video, post_audio, core_report = stage1_av_euler_step_from_denoised(
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
        raw_denoised_stage_2_video_state,
        raw_denoised_stage_2_audio_state,
        sigma_schedule,
        step_index,
    )
    report = (
        f"PASS=True; stage=2G_av_euler; sigma={float(sigma_schedule[int(step_index)].item()):.9g}; "
        f"sigma_next={float(sigma_schedule[int(step_index)+1].item()):.9g}; step_index={int(step_index)}; "
        f"seed={int(shared_seed)}; seed_mode=continued_not_reseeded; core=({core_report})."
    )
    return next_video, next_audio, post_video, post_audio, report


def validate_stage2_av_euler_step(
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
    raw_denoised_stage_2_video_state: dict[str, Any],
    raw_denoised_stage_2_audio_state: dict[str, Any],
    next_stage_2_video_state: dict[str, Any],
    next_stage_2_audio_state: dict[str, Any],
    post_processed_stage_2_video_state: dict[str, Any],
    post_processed_stage_2_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
) -> dict[str, Any]:
    """Independent exact-value validator for the Stage-2 AV post-process + Euler transition."""
    metadata_error = 0.0
    seed = None
    try:
        sigma_schedule = _require_exact_stage2_sigmas(sigmas)
        seed = resolve_stage2_shared_av_seed(noised_stage_2_video_state, noised_stage_2_audio_state)
    except Exception:
        sigma_schedule = sigmas.detach().to(dtype=torch.float32)
        metadata_error = 1.0

    base = validate_stage1_av_euler_step(
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
        raw_denoised_stage_2_video_state,
        raw_denoised_stage_2_audio_state,
        next_stage_2_video_state,
        next_stage_2_audio_state,
        post_processed_stage_2_video_state,
        post_processed_stage_2_audio_state,
        sigma_schedule,
        step_index,
    )
    passed = bool(base["passed"] and metadata_error == 0.0)
    return {
        **base,
        "passed": passed,
        "metadata_error": float(metadata_error),
        "seed": seed,
    }
