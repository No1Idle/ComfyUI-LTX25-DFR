"""Exact joint AV post-process + deterministic Euler state advance for Stage 1.

This is the AV counterpart of the previously validated video-only Euler bridge.
It mirrors upstream LTX-2 ``euler_denoising_loop`` semantics for both modalities:

    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    next     = EulerDiffusionStep.step(state.latent, denoised, sigmas, step_index)

The two modalities are advanced independently after the shared joint transformer
prediction, exactly as upstream does in ``_step_state``.
"""

from __future__ import annotations

from typing import Any

import torch

from .audio_state import clone_audio_official_state, get_audio_official_state
from .av_execution import _audio_latent_with_replaced_token_latent
from .dfr_execution import (
    _normalize_sigma_schedule,
    euler_step_from_official_denoised,
    post_process_official_denoised,
)
from .latent_state import clone_or_create_official_state, ensure_token_state


def _audio_post_process_official_denoised(
    noised_audio_state: dict[str, Any],
    raw_denoised_audio_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    noised_state = clone_audio_official_state(noised_audio_state)
    denoised_state = clone_audio_official_state(raw_denoised_audio_state)
    noised_tokens = noised_state["token_state"]
    denoised_tokens = denoised_state["token_state"]

    if tuple(denoised_tokens["latent"].shape) != tuple(noised_tokens["latent"].shape):
        raise ValueError(
            "Raw denoised audio token shape does not match the noised official audio state: "
            f"denoised={tuple(denoised_tokens['latent'].shape)}, noised={tuple(noised_tokens['latent'].shape)}."
        )

    execution_device = denoised_tokens["latent"].device
    denoised_f32 = denoised_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    mask_f32 = noised_tokens["denoise_mask"].to(device=execution_device, dtype=torch.float32)
    clean_f32 = noised_tokens["clean_latent"].to(device=execution_device, dtype=torch.float32)
    blended = (
        denoised_f32 * mask_f32
        + clean_f32 * (1.0 - mask_f32)
    ).to(device=execution_device, dtype=denoised_tokens["latent"].dtype)

    out = _audio_latent_with_replaced_token_latent(noised_audio_state, blended)
    report = (
        f"token_shape={tuple(int(x) for x in blended.shape)}; execution_device={execution_device}; "
        f"raw_denoised_device={denoised_tokens['latent'].device}; "
        f"clean_source_device={noised_tokens['clean_latent'].device}; "
        f"mask_source_device={noised_tokens['denoise_mask'].device}; "
        f"mask_min={float(noised_tokens['denoise_mask'].min().item()):.9g}; "
        f"mask_max={float(noised_tokens['denoise_mask'].max().item()):.9g}."
    )
    return out, report


def _audio_euler_step_from_official_denoised(
    noised_audio_state: dict[str, Any],
    raw_denoised_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    step_index = int(step_index)
    if step_index < 0 or step_index >= sigma_schedule.numel() - 1:
        raise ValueError(f"step_index must be in [0, {sigma_schedule.numel() - 2}], got {step_index}.")

    noised_state = clone_audio_official_state(noised_audio_state)
    noised_tokens = noised_state["token_state"]
    post_processed_state, pp_report = _audio_post_process_official_denoised(
        noised_audio_state,
        raw_denoised_audio_state,
    )
    pp_tokens = get_audio_official_state(post_processed_state)["token_state"]

    execution_device = pp_tokens["latent"].device
    sample = noised_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    denoised = pp_tokens["latent"].to(device=execution_device, dtype=torch.float32)
    sigma = float(sigma_schedule[step_index].item())
    sigma_next = float(sigma_schedule[step_index + 1].item())

    # Algebraically identical to upstream EulerDiffusionStep:
    # velocity = (sample - denoised) / sigma
    # sample_next = sample + velocity * (sigma_next - sigma)
    if sigma_next == 0.0:
        sample_next = denoised
    else:
        if sigma == 0.0:
            raise ValueError("Current sigma is zero before the terminal step; cannot perform an Euler update.")
        ratio = sigma_next / sigma
        sample_next = ratio * sample + (1.0 - ratio) * denoised

    next_state = _audio_latent_with_replaced_token_latent(
        noised_audio_state,
        sample_next.to(device=execution_device, dtype=noised_tokens["latent"].dtype),
    )
    report = (
        f"sigma={sigma:.9g}; sigma_next={sigma_next:.9g}; step_index={step_index}; "
        f"execution_device={execution_device}; token_shape={tuple(int(x) for x in sample_next.shape)}; "
        f"post_process=({pp_report})."
    )
    return next_state, post_processed_state, report


def stage1_av_euler_step_from_denoised(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    raw_denoised_video_state: dict[str, Any],
    raw_denoised_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Advance both Stage-1 modalities through one exact post-process + Euler transition."""
    next_video, post_video, video_report = euler_step_from_official_denoised(
        noised_video_state,
        raw_denoised_video_state,
        sigmas,
        step_index,
    )
    next_audio, post_audio, audio_report = _audio_euler_step_from_official_denoised(
        noised_audio_state,
        raw_denoised_audio_state,
        sigmas,
        step_index,
    )
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    report = (
        f"step_index={int(step_index)}; sigma={float(sigma_schedule[int(step_index)].item()):.9g}; "
        f"sigma_next={float(sigma_schedule[int(step_index)+1].item()):.9g}; "
        f"video=({video_report}); audio=({audio_report})."
    )
    return next_video, next_audio, post_video, post_audio, report


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 1.0
    if a.numel() == 0:
        return 0.0
    return float((a.to(device=b.device, dtype=torch.float32) - b.to(dtype=torch.float32)).abs().max().item())


def validate_stage1_av_euler_step(
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    raw_denoised_video_state: dict[str, Any],
    raw_denoised_audio_state: dict[str, Any],
    next_video_state: dict[str, Any],
    next_audio_state: dict[str, Any],
    post_processed_video_state: dict[str, Any],
    post_processed_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
) -> dict[str, Any]:
    """Independent exact-value validator for both modality updates."""
    exp_nv, exp_na, exp_pv, exp_pa, _ = stage1_av_euler_step_from_denoised(
        noised_video_state,
        noised_audio_state,
        raw_denoised_video_state,
        raw_denoised_audio_state,
        sigmas,
        step_index,
    )

    def video_tokens(latent: dict[str, Any]):
        st = clone_or_create_official_state(latent)
        return ensure_token_state(
            st,
            target_samples=latent["samples"],
            fps=float(st.get("fps") or 24.0),
            scale_factors=tuple(int(x) for x in st.get("scale_factors", (8, 32, 32))),
            causal_fix=bool(st.get("causal_fix", True)),
        )

    nv = video_tokens(next_video_state)
    env = video_tokens(exp_nv)
    pv = video_tokens(post_processed_video_state)
    epv = video_tokens(exp_pv)
    na = get_audio_official_state(next_audio_state)["token_state"]
    ena = get_audio_official_state(exp_na)["token_state"]
    pa = get_audio_official_state(post_processed_audio_state)["token_state"]
    epa = get_audio_official_state(exp_pa)["token_state"]

    video_next_error = _max_abs(nv["latent"], env["latent"])
    audio_next_error = _max_abs(na["latent"], ena["latent"])
    video_post_error = _max_abs(pv["latent"], epv["latent"])
    audio_post_error = _max_abs(pa["latent"], epa["latent"])

    # Metadata invariants: post/Euler may change only the latent; masks, clean
    # state and positions must survive unchanged.
    nvs = video_tokens(noised_video_state)
    nas = get_audio_official_state(noised_audio_state)["token_state"]
    video_mask_error = _max_abs(nv["denoise_mask"], nvs["denoise_mask"])
    audio_mask_error = _max_abs(na["denoise_mask"], nas["denoise_mask"])
    video_position_error = _max_abs(nv["positions"], nvs["positions"])
    audio_position_error = _max_abs(na["positions"], nas["positions"])
    video_clean_error = _max_abs(nv["clean_latent"], nvs["clean_latent"])
    audio_clean_error = _max_abs(na["clean_latent"], nas["clean_latent"])

    passed = all(
        x == 0.0
        for x in (
            video_next_error,
            audio_next_error,
            video_post_error,
            audio_post_error,
            video_mask_error,
            audio_mask_error,
            video_position_error,
            audio_position_error,
            video_clean_error,
            audio_clean_error,
        )
    )
    sigma_schedule = _normalize_sigma_schedule(sigmas)
    return {
        "passed": bool(passed),
        "video_next_error": video_next_error,
        "audio_next_error": audio_next_error,
        "video_post_error": video_post_error,
        "audio_post_error": audio_post_error,
        "video_mask_error": video_mask_error,
        "audio_mask_error": audio_mask_error,
        "video_position_error": video_position_error,
        "audio_position_error": audio_position_error,
        "video_clean_error": video_clean_error,
        "audio_clean_error": audio_clean_error,
        "video_token_shape": tuple(int(x) for x in nv["latent"].shape),
        "audio_token_shape": tuple(int(x) for x in na["latent"].shape),
        "sigma": float(sigma_schedule[int(step_index)].item()),
        "sigma_next": float(sigma_schedule[int(step_index) + 1].item()),
    }
