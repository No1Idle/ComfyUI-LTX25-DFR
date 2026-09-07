"""Stage-2A handoff for strict Spatial-DFR parity.

Upstream DFR keeps four things after Stage 1:

1. the Stage-1 base video latent, cloned as ``reserved_half_res_video``;
2. the Stage-1 generated keyframe-slot stack;
3. the Stage-1 audio latent;
4. the *same* GaussianNoiser/generator object, whose RNG state has already
   consumed Stage-1 VIDEO noise first and AUDIO noise second.

This module freezes that boundary before any spatial upsampling or Stage-2
conditioning is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .audio_state import get_audio_official_state
from .dfr_execution import extract_official_base_latent, extract_official_generated_keyframes
from .dfr_noiser import NOISER_METADATA_KEY, OFFICIAL_DFR_STATE_DTYPE
from .latent_state import clone_or_create_official_state, ensure_token_state, get_official_state


@dataclass(frozen=True)
class DFRStage2Handoff:
    version: int
    seed: int
    rng_state_after_stage1_av: torch.Tensor
    rng_device_type: str
    reserved_half_res_video: torch.Tensor
    stage_1_generated_keyframes: torch.Tensor
    stage_1_audio_latent: torch.Tensor
    video_token_shape_at_stage1_noise: tuple[int, ...]
    audio_token_shape_at_stage1_noise: tuple[int, ...]
    fps: float
    duration_seconds: float


HANDOFF_VERSION = 1


def _video_tokens(latent: dict[str, Any]) -> dict[str, Any]:
    state = clone_or_create_official_state(latent)
    return ensure_token_state(
        state,
        target_samples=latent["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 1.0
    if a.numel() == 0:
        return 0.0
    return float(
        (a.to(device=b.device, dtype=torch.float32) - b.to(dtype=torch.float32))
        .abs()
        .max()
        .item()
    )


def _require_shared_rng_metadata(
    final_video_state: dict[str, Any],
    final_audio_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, int]:
    vt = _video_tokens(final_video_state)
    at = get_audio_official_state(final_audio_state)["token_state"]
    vm = vt.get(NOISER_METADATA_KEY) or {}
    am = at.get(NOISER_METADATA_KEY) or {}

    if vm.get("shared_av_rng") is not True or am.get("shared_av_rng") is not True:
        raise ValueError(
            "Stage-2 handoff requires outputs derived from 'LTX Official: Stage 1 AV Gaussian Noiser'."
        )
    if vm.get("rng_order") != "video_then_audio" or am.get("rng_order") != "video_then_audio":
        raise ValueError("Stage-1 AV RNG metadata does not declare the required video_then_audio order.")
    if "seed" not in vm or "seed" not in am or int(vm["seed"]) != int(am["seed"]):
        raise ValueError("Video/audio Stage-1 shared RNG seed metadata is missing or inconsistent.")

    vr = vm.get("rng_state_after_stage1_av")
    ar = am.get("rng_state_after_stage1_av")
    if not torch.is_tensor(vr) or not torch.is_tensor(ar):
        raise ValueError(
            "Stage-1 states do not contain the captured RNG continuation state. "
            "Rerun Stage 1 with Phase B14's shared AV noiser before entering Stage 2."
        )
    vr_cpu = vr.detach().cpu().clone()
    ar_cpu = ar.detach().cpu()
    if not torch.equal(vr_cpu, ar_cpu):
        raise ValueError("Video/audio Stage-1 RNG continuation states differ.")

    return vm, am, vr_cpu, int(vm["seed"])


def prepare_stage2_handoff(
    final_video_state: dict[str, Any],
    final_audio_state: dict[str, Any],
    stage_1_base_latent: dict[str, Any],
    generated_keyframes: dict[str, Any],
) -> tuple[DFRStage2Handoff, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    """Freeze the exact Stage-1 -> Stage-2 boundary without changing resolution."""
    vm, am, rng_state, seed = _require_shared_rng_metadata(final_video_state, final_audio_state)

    expected_base = extract_official_base_latent(final_video_state)["samples"]
    actual_base = stage_1_base_latent.get("samples")
    if not torch.is_tensor(actual_base):
        raise ValueError("stage_1_base_latent must contain a tensor under 'samples'.")
    if _max_abs(expected_base, actual_base) != 0.0:
        raise ValueError("stage_1_base_latent is not an exact extraction of final_video_state.")

    expected_kf = extract_official_generated_keyframes(final_video_state)["samples"]
    actual_kf = generated_keyframes.get("samples")
    if not torch.is_tensor(actual_kf):
        raise ValueError("generated_keyframes must contain a tensor under 'samples'.")
    if _max_abs(expected_kf, actual_kf) != 0.0:
        raise ValueError("generated_keyframes is not an exact extraction of final_video_state.")

    # Upstream explicitly keeps only the first video batch for DFR continuation.
    reserved_video_tensor = actual_base[:1].detach().clone()
    reserved_half_res_video = {"samples": reserved_video_tensor}

    stage1_audio_tensor = final_audio_state["samples"].detach().clone()
    stage_1_audio_latent = {"samples": stage1_audio_tensor}

    stage1_kf_tensor = actual_kf.detach().clone()
    stage_1_generated_keyframes = {"samples": stage1_kf_tensor}

    vstate = clone_or_create_official_state(final_video_state)
    fps = float(vstate.get("fps") or 24.0)
    astate = get_audio_official_state(final_audio_state)
    duration_seconds = float(astate.get("duration_seconds", 0.0))

    handoff = DFRStage2Handoff(
        version=HANDOFF_VERSION,
        seed=seed,
        rng_state_after_stage1_av=rng_state,
        rng_device_type=str(vm.get("device_type", am.get("device_type", "unknown"))),
        reserved_half_res_video=reserved_video_tensor,
        stage_1_generated_keyframes=stage1_kf_tensor,
        stage_1_audio_latent=stage1_audio_tensor,
        video_token_shape_at_stage1_noise=tuple(int(x) for x in vm.get("token_shape", ())),
        audio_token_shape_at_stage1_noise=tuple(int(x) for x in am.get("token_shape", ())),
        fps=fps,
        duration_seconds=duration_seconds,
    )

    report = (
        f"PASS=True; stage=2A_handoff; seed={seed}; rng_device_type={handoff.rng_device_type}; "
        f"rng_state_bytes={int(rng_state.numel())}; reserved_half_res_video_shape={tuple(int(x) for x in reserved_video_tensor.shape)}; "
        f"generated_keyframes_shape={tuple(int(x) for x in stage1_kf_tensor.shape)}; "
        f"stage_1_audio_shape={tuple(int(x) for x in stage1_audio_tensor.shape)}; "
        f"video_noise_token_shape={handoff.video_token_shape_at_stage1_noise}; "
        f"audio_noise_token_shape={handoff.audio_token_shape_at_stage1_noise}; fps={fps:.9g}; duration_seconds={duration_seconds:.9g}."
    )
    return handoff, reserved_half_res_video, stage_1_audio_latent, stage_1_generated_keyframes, report


def prepare_stage1_handoff(
    final_video_state: dict[str, Any],
    final_audio_state: dict[str, Any],
    stage_1_base_latent: dict[str, Any],
    generated_keyframes: dict[str, Any],
) -> DFRStage2Handoff:
    """Build the lean public Stage-1 bundle from internally produced loop outputs.

    Unlike the atomic Stage-2A helper above, this path does not re-compare tensors
    that were extracted by the same consolidated Stage-1 operation and does not
    construct a diagnostic report.
    """
    vm, am, rng_state, seed = _require_shared_rng_metadata(final_video_state, final_audio_state)

    base = stage_1_base_latent.get("samples")
    keyframes = generated_keyframes.get("samples")
    audio = final_audio_state.get("samples")
    if not torch.is_tensor(base) or base.ndim != 5:
        raise ValueError("stage_1_base_latent must contain [B,C,T,H,W] tensor 'samples'.")
    if not torch.is_tensor(keyframes) or keyframes.ndim != 5:
        raise ValueError("generated_keyframes must contain [B,C,K,H,W] tensor 'samples'.")
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError("final_audio_state must contain [B,C,T,F] tensor 'samples'.")

    reserved_video = base[:1].detach().clone()
    stage_1_audio = audio.detach().clone()
    stage_1_keyframes = keyframes.detach().clone()
    video_state = get_official_state(final_video_state)
    audio_state = get_audio_official_state(final_audio_state)

    return DFRStage2Handoff(
        version=HANDOFF_VERSION,
        seed=seed,
        rng_state_after_stage1_av=rng_state,
        rng_device_type=str(vm.get("device_type", am.get("device_type", "unknown"))),
        reserved_half_res_video=reserved_video,
        stage_1_generated_keyframes=stage_1_keyframes,
        stage_1_audio_latent=stage_1_audio,
        video_token_shape_at_stage1_noise=tuple(int(x) for x in vm.get("token_shape", ())),
        audio_token_shape_at_stage1_noise=tuple(int(x) for x in am.get("token_shape", ())),
        fps=float(video_state.get("fps") or 24.0),
        duration_seconds=float(audio_state.get("duration_seconds", 0.0)),
    )


def generator_from_stage2_handoff(handoff: DFRStage2Handoff, device: torch.device) -> torch.Generator:
    """Recreate the exact upstream generator continuation point for Stage 2."""
    if not isinstance(handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(handoff).__name__}.")
    if int(handoff.version) != HANDOFF_VERSION:
        raise ValueError(f"Unsupported Stage-2 handoff version {handoff.version}.")
    generator = torch.Generator(device=torch.device(device))
    generator.set_state(handoff.rng_state_after_stage1_av.detach().cpu())
    return generator


def _recompute_expected_rng_state(handoff: DFRStage2Handoff, device: torch.device) -> torch.Tensor:
    if not handoff.video_token_shape_at_stage1_noise or not handoff.audio_token_shape_at_stage1_noise:
        raise ValueError("Handoff lacks Stage-1 noiser token shapes required for independent RNG validation.")
    generator = torch.Generator(device=device).manual_seed(int(handoff.seed))
    torch.randn(
        *handoff.video_token_shape_at_stage1_noise,
        device=device,
        dtype=OFFICIAL_DFR_STATE_DTYPE,
        generator=generator,
    )
    torch.randn(
        *handoff.audio_token_shape_at_stage1_noise,
        device=device,
        dtype=OFFICIAL_DFR_STATE_DTYPE,
        generator=generator,
    )
    return generator.get_state().detach().cpu()


def validate_stage2_handoff(
    handoff: DFRStage2Handoff,
    final_video_state: dict[str, Any],
    final_audio_state: dict[str, Any],
    reserved_half_res_video: dict[str, Any],
    stage_1_audio_latent: dict[str, Any],
    stage_1_generated_keyframes: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(handoff).__name__}.")

    expected_base = extract_official_base_latent(final_video_state)["samples"][:1]
    expected_kf = extract_official_generated_keyframes(final_video_state)["samples"]
    expected_audio = final_audio_state["samples"]

    video_error = _max_abs(reserved_half_res_video["samples"], expected_base)
    keyframes_error = _max_abs(stage_1_generated_keyframes["samples"], expected_kf)
    audio_error = _max_abs(stage_1_audio_latent["samples"], expected_audio)
    handoff_video_error = _max_abs(handoff.reserved_half_res_video, expected_base)
    handoff_keyframes_error = _max_abs(handoff.stage_1_generated_keyframes, expected_kf)
    handoff_audio_error = _max_abs(handoff.stage_1_audio_latent, expected_audio)

    vt = _video_tokens(final_video_state)
    device = vt["latent"].device
    expected_rng_state = _recompute_expected_rng_state(handoff, device)
    rng_state_error = 0.0 if torch.equal(
        handoff.rng_state_after_stage1_av.detach().cpu(), expected_rng_state
    ) else 1.0

    # Strong continuation check: the *next* random draw from the restored state
    # must match the next draw from an independently replayed Stage-1 generator.
    restored = generator_from_stage2_handoff(handoff, device)
    replayed = torch.Generator(device=device).manual_seed(int(handoff.seed))
    torch.randn(
        *handoff.video_token_shape_at_stage1_noise,
        device=device,
        dtype=OFFICIAL_DFR_STATE_DTYPE,
        generator=replayed,
    )
    torch.randn(
        *handoff.audio_token_shape_at_stage1_noise,
        device=device,
        dtype=OFFICIAL_DFR_STATE_DTYPE,
        generator=replayed,
    )
    next_restored = torch.randn((32,), device=device, dtype=OFFICIAL_DFR_STATE_DTYPE, generator=restored)
    next_replayed = torch.randn((32,), device=device, dtype=OFFICIAL_DFR_STATE_DTYPE, generator=replayed)
    next_rng_draw_error = _max_abs(next_restored, next_replayed)

    passed = all(
        x == 0.0
        for x in (
            video_error,
            keyframes_error,
            audio_error,
            handoff_video_error,
            handoff_keyframes_error,
            handoff_audio_error,
            rng_state_error,
            next_rng_draw_error,
        )
    )

    return {
        "passed": bool(passed),
        "video_error": float(video_error),
        "keyframes_error": float(keyframes_error),
        "audio_error": float(audio_error),
        "handoff_video_error": float(handoff_video_error),
        "handoff_keyframes_error": float(handoff_keyframes_error),
        "handoff_audio_error": float(handoff_audio_error),
        "rng_state_error": float(rng_state_error),
        "next_rng_draw_error": float(next_rng_draw_error),
        "seed": int(handoff.seed),
        "rng_state_bytes": int(handoff.rng_state_after_stage1_av.numel()),
        "device": str(device),
        "reserved_video_shape": tuple(int(x) for x in expected_base.shape),
        "keyframes_shape": tuple(int(x) for x in expected_kf.shape),
        "audio_shape": tuple(int(x) for x in expected_audio.shape),
    }
