"""Stage-2H full AV loop for strict Spatial-DFR parity.

This closes the detail-enabled Stage-2 execution path by chaining the already
validated Stage-2 primitives:

- 2E: continue the shared Stage-1 RNG stream and materialize Stage-2 VIDEO/AUDIO noised states;
- 2F: run one real joint LTXAV transformer/x0 evaluation at the current Stage-2 sigma;
- 2G: apply official per-modality post-process + Euler advance.

The exact Stage-2 sigma schedule contains 4 sigma points, therefore this loop
executes exactly 3 denoising transitions.
"""

from __future__ import annotations

from typing import Any
import time

import torch

from .av_execution import (
    _execute_stage1_av_whole_schedule_comfy,
    _execute_stage1_av_whole_schedule_direct,
    _model_looks_like_comfy_patcher,
)
from .audio_state import get_audio_official_state, _unpatchify_audio
from .dfr_sigmas import (
    experimental_stage_2_extra_step_sigmas,
    validate_custom_sigma_schedule,
)
from .dfr_execution import extract_official_base_latent
from .stage2_execution import _require_stage2_detailing_model, resolve_stage2_shared_av_seed
from .stage2_noiser import _require_exact_stage2_sigmas, materialize_stage2_av_gaussian_noised_states
from .latent_state import clone_or_create_official_state, ensure_token_state
from .stage2_spatial import DFRUpscaledStage1Handoff, require_upscaled_stage1_handoff


def _execute_stage2_av_whole_schedule(
    model: Any,
    positive: Any,
    negative: Any,
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    sigma_schedule: torch.Tensor,
    cfg_scale: float,
    *,
    collect_diagnostics: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    """Run every Stage-2 transition inside one model/conditioning lifecycle.

    The low-level AV bridge and Euler algebra are identical for Stage 1 and
    Stage 2. Stage-specific parity remains enforced here by requiring the
    detailing-model contract and the continued Stage-2 AV noiser metadata
    before entering the shared whole-schedule backend.
    """
    _require_stage2_detailing_model(model)
    shared_seed = resolve_stage2_shared_av_seed(noised_video_state, noised_audio_state)

    backend = (
        _execute_stage1_av_whole_schedule_comfy
        if _model_looks_like_comfy_patcher(model)
        else _execute_stage1_av_whole_schedule_direct
    )
    return backend(
        model=model,
        positive=positive,
        negative=negative,
        noised_video_state=noised_video_state,
        noised_audio_state=noised_audio_state,
        sigma_schedule=sigma_schedule,
        cfg_scale=float(cfg_scale),
        seed=shared_seed,
        collect_diagnostics=collect_diagnostics,
        profiler=None,
        **({"memory_profile": True} if _model_looks_like_comfy_patcher(model) else {}),
    )


def execute_stage2_av_denoising_loop(
    model: Any,
    positive: Any,
    negative: Any,
    stage_2_handoff: Any,
    stage_2_video_state: dict[str, Any],
    stage_1_audio_for_stage2: dict[str, Any],
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    *,
    allow_experimental_extra_step: bool = False,
    require_official_sigmas: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], str]:
    """Run the official trajectory, or the explicit one-extra-step experiment."""
    sigma_schedule = (
        _require_exact_stage2_sigmas(
            sigmas,
            allow_experimental_extra_step=allow_experimental_extra_step,
        )
        if require_official_sigmas
        else validate_custom_sigma_schedule(sigmas, "Custom Stage-2 sigmas")
    )
    cfg_scale = float(cfg_scale)

    total_started = time.perf_counter() if collect_diagnostics else 0.0
    noiser_started = time.perf_counter() if collect_diagnostics else 0.0
    current_video_state, current_audio_state, noiser_report = materialize_stage2_av_gaussian_noised_states(
        stage_2_handoff=stage_2_handoff,
        stage_2_video_state=stage_2_video_state,
        stage_1_audio_for_stage2=stage_1_audio_for_stage2,
        stage_2_sigmas_tensor=sigma_schedule,
        allow_experimental_extra_step=allow_experimental_extra_step,
        require_official_sigmas=require_official_sigmas,
        collect_diagnostics=collect_diagnostics,
    )
    noiser_elapsed = time.perf_counter() - noiser_started if collect_diagnostics else 0.0
    initial_shared_seed = resolve_stage2_shared_av_seed(current_video_state, current_audio_state)

    current_video_state, current_audio_state, step_reports, backend_report = (
        _execute_stage2_av_whole_schedule(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=current_video_state,
            noised_audio_state=current_audio_state,
            sigma_schedule=sigma_schedule,
            cfg_scale=cfg_scale,
            collect_diagnostics=collect_diagnostics,
        )
    )

    stage_2_base_latent = extract_official_base_latent(current_video_state)

    resolved_final_seed = resolve_stage2_shared_av_seed(current_video_state, current_audio_state)
    if resolved_final_seed != initial_shared_seed:
        raise RuntimeError("Stage-2 AV seed metadata changed during the trajectory.")
    if not collect_diagnostics:
        return current_video_state, current_audio_state, stage_2_base_latent, ""

    final_video_state = clone_or_create_official_state(current_video_state)
    final_video_tokens = ensure_token_state(
        final_video_state,
        target_samples=current_video_state["samples"],
        fps=float(final_video_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in final_video_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(final_video_state.get("causal_fix", True)),
    )
    final_audio_state = get_audio_official_state(current_audio_state)
    final_audio_tokens = final_audio_state["token_state"]

    total_elapsed = time.perf_counter() - total_started
    shared_seed = resolve_stage2_shared_av_seed(current_video_state, current_audio_state)

    stage_name = "2H_experimental_extra_step_av_loop" if allow_experimental_extra_step else "2H_av_loop"
    schedule_kind = (
        "experimental_final_interval_midpoint_split"
        if allow_experimental_extra_step
        else ("official" if require_official_sigmas else "custom")
    )
    schedule_text = ",".join(f"{float(x):.9g}" for x in sigma_schedule.tolist())
    report = (
        f"PASS=True; EXPERIMENTAL={allow_experimental_extra_step}; stage={stage_name}; "
        f"schedule_kind={schedule_kind}; sigmas=({schedule_text}); "
        f"steps={int(sigma_schedule.numel()-1)}; cfg_scale={cfg_scale:.9g}; "
        f"seed={int(shared_seed)}; video_device={final_video_tokens['latent'].device}; "
        f"audio_device={final_audio_tokens['latent'].device}; "
        f"video_token_shape={tuple(int(x) for x in final_video_tokens['latent'].shape)}; "
        f"audio_token_shape={tuple(int(x) for x in final_audio_tokens['latent'].shape)}; "
        f"final_base_latent_shape={tuple(int(x) for x in stage_2_base_latent['samples'].shape)}; "
        f"duration_seconds={float(final_audio_state.get('duration_seconds', 0.0)):.9g}; "
        f"noiser_time={noiser_elapsed:.2f}s; total_time={total_elapsed:.2f}s; "
        f"noiser=({noiser_report}); backend=({backend_report}); trajectory=[{' | '.join(step_reports)}]."
    )
    return current_video_state, current_audio_state, stage_2_base_latent, report


def execute_stage2_av_denoising_loop_from_upscaled_handoff(
    model: Any,
    positive: Any,
    negative: Any,
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    stage_2_video_state: dict[str, Any],
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    *,
    allow_experimental_extra_step: bool = False,
    require_official_sigmas: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], str]:
    """Run Stage 2 using audio and RNG continuation stored in the enriched handoff."""
    bundle = require_upscaled_stage1_handoff(upscaled_stage_1_handoff)
    return execute_stage2_av_denoising_loop(
        model=model,
        positive=positive,
        negative=negative,
        stage_2_handoff=bundle.stage_1_handoff,
        stage_2_video_state=stage_2_video_state,
        stage_1_audio_for_stage2={"samples": bundle.stage_1_handoff.stage_1_audio_latent},
        sigmas=sigmas,
        cfg_scale=cfg_scale,
        allow_experimental_extra_step=allow_experimental_extra_step,
        require_official_sigmas=require_official_sigmas,
        collect_diagnostics=collect_diagnostics,
    )


def execute_experimental_stage2_av_denoising_loop_extra_step(
    model: Any,
    positive: Any,
    negative: Any,
    stage_2_handoff: Any,
    stage_2_video_state: dict[str, Any],
    stage_1_audio_for_stage2: dict[str, Any],
    official_stage_2_sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], str]:
    """Run a controlled four-transition comparison derived from the official schedule."""
    # Require the normal schedule at the node boundary so an unrelated custom
    # scheduler cannot silently turn this controlled experiment into another test.
    _require_exact_stage2_sigmas(official_stage_2_sigmas)
    return execute_stage2_av_denoising_loop(
        model=model,
        positive=positive,
        negative=negative,
        stage_2_handoff=stage_2_handoff,
        stage_2_video_state=stage_2_video_state,
        stage_1_audio_for_stage2=stage_1_audio_for_stage2,
        sigmas=experimental_stage_2_extra_step_sigmas(),
        cfg_scale=cfg_scale,
        allow_experimental_extra_step=True,
    )


def execute_experimental_stage2_av_denoising_loop_extra_step_from_upscaled_handoff(
    model: Any,
    positive: Any,
    negative: Any,
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    stage_2_video_state: dict[str, Any],
    official_stage_2_sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], str]:
    """Run the controlled extra-step variant from the lean public handoff."""
    _require_exact_stage2_sigmas(official_stage_2_sigmas)
    return execute_stage2_av_denoising_loop_from_upscaled_handoff(
        model=model,
        positive=positive,
        negative=negative,
        upscaled_stage_1_handoff=upscaled_stage_1_handoff,
        stage_2_video_state=stage_2_video_state,
        sigmas=experimental_stage_2_extra_step_sigmas(),
        cfg_scale=cfg_scale,
        allow_experimental_extra_step=True,
    )


def validate_stage2_av_loop_outputs(
    final_stage_2_video_state: dict[str, Any],
    final_stage_2_audio_state: dict[str, Any],
    stage_2_base_latent: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate the exported outputs of the full Stage-2 AV loop."""
    expected_base = extract_official_base_latent(final_stage_2_video_state)["samples"]
    actual_base = stage_2_base_latent.get("samples")
    if actual_base is None:
        raise ValueError("stage_2_base_latent must contain a 'samples' tensor.")
    if tuple(expected_base.shape) != tuple(actual_base.shape):
        base_max_abs_error = 1.0
    elif expected_base.numel() == 0:
        base_max_abs_error = 0.0
    else:
        base_max_abs_error = float(
            (expected_base.to(device=actual_base.device, dtype=torch.float32) - actual_base.to(dtype=torch.float32))
            .abs()
            .max()
            .item()
        )

    audio_state = get_audio_official_state(final_stage_2_audio_state)
    audio_tokens = audio_state["token_state"]
    rebuilt_audio = _unpatchify_audio(
        audio_tokens["latent"],
        channels=int(audio_tokens["channels"]),
        mel_bins=int(audio_tokens["mel_bins"]),
    )
    actual_audio = final_stage_2_audio_state.get("samples")
    if actual_audio is None or tuple(rebuilt_audio.shape) != tuple(actual_audio.shape):
        audio_preview_error = 1.0
    elif rebuilt_audio.numel() == 0:
        audio_preview_error = 0.0
    else:
        audio_preview_error = float(
            (rebuilt_audio.to(device=actual_audio.device, dtype=torch.float32) - actual_audio.to(dtype=torch.float32))
            .abs()
            .max()
            .item()
        )

    metadata_error = 0.0
    try:
        shared_seed = resolve_stage2_shared_av_seed(final_stage_2_video_state, final_stage_2_audio_state)
    except Exception:
        shared_seed = None
        metadata_error = 1.0

    video_state = clone_or_create_official_state(final_stage_2_video_state)
    video_tokens = ensure_token_state(
        video_state,
        target_samples=final_stage_2_video_state["samples"],
        fps=float(video_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in video_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(video_state.get("causal_fix", True)),
    )

    passed = (
        base_max_abs_error == 0.0
        and audio_preview_error <= 1e-6
        and metadata_error == 0.0
    )

    return {
        "passed": bool(passed),
        "base_max_abs_error": float(base_max_abs_error),
        "audio_preview_error": float(audio_preview_error),
        "metadata_error": float(metadata_error),
        "shared_seed": shared_seed,
        "video_token_shape": tuple(int(x) for x in video_tokens["latent"].shape),
        "audio_shape": tuple(int(x) for x in final_stage_2_audio_state["samples"].shape),
        "audio_token_shape": tuple(int(x) for x in audio_tokens["latent"].shape),
        "base_latent_shape": tuple(int(x) for x in actual_base.shape),
        "duration_seconds": float(audio_state.get("duration_seconds", 0.0)),
    }
