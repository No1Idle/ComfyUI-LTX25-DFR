
"""U3.1 final-decoder handoff for strict Spatial-DFR parity.

This module does not decode anything.  It only preserves the two pieces of
Stage-2 state that current upstream DFR carries into its final DiffVAE decoder:

1. the final Stage-2 generated-keyframe slot latents;
2. the shared generator state after Stage-2 VIDEO then AUDIO Gaussian noise.

The Stage-2 denoising trajectory is deterministic, so no RNG is consumed after
that captured state before the final decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .audio_state import get_audio_official_state
from .dfr_noiser import NOISER_METADATA_KEY
from .latent_state import clone_or_create_official_state, ensure_token_state, extract_generated_keyframes
from .stage2_noiser import STAGE2_NOISER_STAGE, STAGE2_RNG_ORDER


DECODER_HANDOFF_VERSION = 1


@dataclass(frozen=True)
class DFRStage2DecoderHandoff:
    """Immutable Stage-2 -> final-decoder continuation state."""

    version: int
    seed: int
    rng_state_after_stage2_av: torch.Tensor
    rng_device_type: str
    stage_2_generated_keyframes: torch.Tensor
    generated_keyframe_count: int
    generated_keyframe_shape: tuple[int, ...]


def _video_tokens(latent: dict[str, Any]) -> dict[str, Any]:
    state = clone_or_create_official_state(latent)
    return ensure_token_state(
        state,
        target_samples=latent["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )


def prepare_stage2_decoder_handoff(
    final_stage_2_video_state: dict[str, Any],
    final_stage_2_audio_state: dict[str, Any],
    *,
    collect_diagnostics: bool = True,
) -> tuple[DFRStage2DecoderHandoff, dict[str, torch.Tensor], str]:
    """Capture final Stage-2 generated slots + continued decoder RNG state."""
    video_tokens = _video_tokens(final_stage_2_video_state)
    audio_tokens = get_audio_official_state(final_stage_2_audio_state)["token_state"]

    vm = video_tokens.get(NOISER_METADATA_KEY) or {}
    am = audio_tokens.get(NOISER_METADATA_KEY) or {}

    if vm.get("stage") != STAGE2_NOISER_STAGE or am.get("stage") != STAGE2_NOISER_STAGE:
        raise ValueError(
            "U3.1 decoder handoff requires final states derived from the Stage-2 continued AV Gaussian noiser."
        )
    if vm.get("shared_av_rng") is not True or am.get("shared_av_rng") is not True:
        raise ValueError("Stage-2 final VIDEO/AUDIO states do not declare one shared AV RNG stream.")
    if vm.get("rng_order") != STAGE2_RNG_ORDER or am.get("rng_order") != STAGE2_RNG_ORDER:
        raise ValueError(
            f"Stage-2 decoder handoff requires rng_order={STAGE2_RNG_ORDER!r} for VIDEO and AUDIO."
        )

    if "seed" not in vm or "seed" not in am or int(vm["seed"]) != int(am["seed"]):
        raise ValueError("Stage-2 final VIDEO/AUDIO RNG seed metadata is missing or inconsistent.")
    seed = int(vm["seed"])

    vr = vm.get("rng_state_after_stage2_av")
    ar = am.get("rng_state_after_stage2_av")
    if not torch.is_tensor(vr) or not torch.is_tensor(ar):
        raise ValueError(
            "Stage-2 final states do not contain rng_state_after_stage2_av. Rerun Stage 2 through the shared AV noiser."
        )
    rng_state = vr.detach().cpu().clone()
    if not torch.equal(rng_state, ar.detach().cpu()):
        raise ValueError("Stage-2 final VIDEO/AUDIO rng_state_after_stage2_av values differ.")

    v_device_type = str(vm.get("device_type") or "")
    a_device_type = str(am.get("device_type") or "")
    if not v_device_type or not a_device_type or v_device_type != a_device_type:
        raise ValueError(
            "Stage-2 final VIDEO/AUDIO RNG device metadata is missing or inconsistent: "
            f"video={v_device_type!r}, audio={a_device_type!r}."
        )

    generated = extract_generated_keyframes(final_stage_2_video_state).detach().clone()
    if generated.ndim != 5:
        raise ValueError(
            "Extracted Stage-2 generated keyframes must be [B,C,K,H,W], "
            f"got {tuple(generated.shape)}."
        )
    generated_count = int(generated.shape[2])
    if generated_count <= 0:
        raise ValueError("Stage-2 final video state contains no generated keyframe slots.")

    handoff = DFRStage2DecoderHandoff(
        version=DECODER_HANDOFF_VERSION,
        seed=seed,
        rng_state_after_stage2_av=rng_state,
        rng_device_type=v_device_type,
        stage_2_generated_keyframes=generated,
        generated_keyframe_count=generated_count,
        generated_keyframe_shape=tuple(int(x) for x in generated.shape),
    )
    generated_latent = {"samples": generated.detach().clone()} if collect_diagnostics else {"samples": generated}

    report = ""
    if collect_diagnostics:
        report = (
            f"PASS=True; stage=U3.1_decoder_handoff; seed={seed}; rng_device_type={v_device_type}; "
            f"rng_state_bytes={int(rng_state.numel())}; generated_keyframes={generated_count}; "
            f"generated_keyframes_shape={tuple(int(x) for x in generated.shape)}; "
            "decoder_consumed=False; output_latent_changed=False."
        )
    return handoff, generated_latent, report


def validate_stage2_decoder_handoff(
    final_stage_2_video_state: dict[str, Any],
    final_stage_2_audio_state: dict[str, Any],
    decoder_handoff: DFRStage2DecoderHandoff,
    stage_2_generated_keyframes: dict[str, Any],
) -> dict[str, Any]:
    """Validate U3.1 capture without consuming or mutating decoder inputs."""
    expected_handoff, expected_latent, _ = prepare_stage2_decoder_handoff(
        final_stage_2_video_state,
        final_stage_2_audio_state,
    )
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")

    actual = stage_2_generated_keyframes.get("samples") if isinstance(stage_2_generated_keyframes, dict) else None
    expected = expected_latent["samples"]
    generated_error = 1.0
    if torch.is_tensor(actual) and tuple(actual.shape) == tuple(expected.shape):
        generated_error = 0.0 if torch.equal(actual, expected) else float(
            (actual.to(device=expected.device, dtype=torch.float32) - expected.to(dtype=torch.float32)).abs().max().item()
        )

    rng_error = 0.0 if torch.equal(
        decoder_handoff.rng_state_after_stage2_av.detach().cpu(),
        expected_handoff.rng_state_after_stage2_av.detach().cpu(),
    ) else 1.0
    seed_error = 0.0 if int(decoder_handoff.seed) == int(expected_handoff.seed) else 1.0
    device_error = 0.0 if decoder_handoff.rng_device_type == expected_handoff.rng_device_type else 1.0
    count_error = 0.0 if int(decoder_handoff.generated_keyframe_count) == int(expected_handoff.generated_keyframe_count) else 1.0

    passed = generated_error == 0.0 and rng_error == 0.0 and seed_error == 0.0 and device_error == 0.0 and count_error == 0.0
    return {
        "passed": bool(passed),
        "generated_keyframes_error": float(generated_error),
        "rng_state_error": float(rng_error),
        "seed_error": float(seed_error),
        "device_error": float(device_error),
        "count_error": float(count_error),
        "seed": int(expected_handoff.seed),
        "rng_device_type": expected_handoff.rng_device_type,
        "rng_state_bytes": int(expected_handoff.rng_state_after_stage2_av.numel()),
        "generated_keyframe_count": int(expected_handoff.generated_keyframe_count),
        "generated_keyframe_shape": expected_handoff.generated_keyframe_shape,
    }
