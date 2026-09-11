"""Stage-2E shared AV Gaussian noiser for strict Spatial-DFR parity.

Current upstream DFR constructs ONE ``torch.Generator`` / ``GaussianNoiser`` before
Stage 1, consumes Stage-1 VIDEO noise first and AUDIO noise second, then reuses that
same noiser object for Stage 2.  Stage 2 again builds/noises VIDEO first and AUDIO
second, using ``noise_scale = stage_2_sigmas[0]`` for both modalities.

This module restores the exact generator continuation captured by Stage 2A and
materializes the Stage-2 video/audio official states without reseeding.
"""

from __future__ import annotations

from typing import Any

import torch

from .audio_state import (
    AUDIO_OFFICIAL_STATE_KEY,
    DEFAULT_AUDIO_CAUSAL,
    DEFAULT_AUDIO_HOP_LENGTH,
    DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_AUDIO_SHIFT,
    _audio_positions,
    _new_audio_official_state,
    _patchify_audio,
    _unpatchify_audio,
    get_audio_official_state,
)
from .dfr_noiser import NOISER_METADATA_KEY, OFFICIAL_DFR_STATE_DTYPE, official_gaussian_noiser_formula
from .dfr_sigmas import (
    experimental_stage_2_extra_step_sigmas,
    stage_2_sigmas,
    validate_custom_sigma_schedule,
    validate_sigma_tensor,
)
from .latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, ensure_token_state, get_official_state
from .stage2_handoff import DFRStage2Handoff, generator_from_stage2_handoff


STAGE2_RNG_ORDER = "video_then_audio"
STAGE2_NOISER_STAGE = "stage2_continued"


def _preferred_comfy_device(fallback: torch.device) -> torch.device:
    try:
        import comfy.model_management as mm  # type: ignore

        return torch.device(mm.get_torch_device())
    except Exception:
        return torch.device(fallback)


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


def _randn_like_with_generator(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _require_exact_stage2_sigmas(
    sigmas: torch.Tensor,
    *,
    allow_experimental_extra_step: bool = False,
) -> torch.Tensor:
    expected = experimental_stage_2_extra_step_sigmas() if allow_experimental_extra_step else stage_2_sigmas()
    schedule_name = "experimental_stage_2_extra_step_sigmas" if allow_experimental_extra_step else "stage_2_sigmas"
    passed, max_error, details = validate_sigma_tensor(sigmas, expected, schedule_name)
    if not passed:
        expected_description = (
            "fixed experimental 5-point / 4-transition Stage-2 schedule"
            if allow_experimental_extra_step
            else "exact official 4-point / 3-transition Stage-2 DFR sigma schedule"
        )
        raise ValueError(
            f"Stage 2E requires the {expected_description}. "
            f"{details}; max_error={max_error:.9g}."
        )
    return sigmas.detach().to(dtype=torch.float32)


def _prepare_stage2_video(
    stage_2_video_state: dict[str, Any],
    device: torch.device,
    *,
    require_reference: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    state = clone_or_create_official_state(stage_2_video_state)
    tokens = ensure_token_state(
        state,
        target_samples=stage_2_video_state["samples"],
        fps=float(state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    if tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError(
            "Stage-2 video state is already Gaussian-noised. Connect the validated Stage-2C state before noising."
        )

    operations = list(state.get("operations", []))
    expected_last = "VideoConditionByReferenceLatent" if require_reference else "VideoGeneratedKeyframeSlots"
    if not operations or operations[-1].get("type") != expected_last:
        raise ValueError(
            f"Stage 2 noising requires conditioning ending in {expected_last}."
        )
    if not require_reference and any(op.get("type") == "VideoConditionByReferenceLatent" for op in operations):
        raise ValueError("Same-resolution refinement must not contain reference-video conditioning.")

    latent = tokens["latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = tokens["clean_latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = tokens["denoise_mask"].to(device=device, dtype=torch.float32)
    tokens["positions"] = tokens["positions"].to(device=device, dtype=torch.float32)
    if torch.is_tensor(tokens.get("keyframes_mask")):
        tokens["keyframes_mask"] = tokens["keyframes_mask"].to(device=device, dtype=torch.float32)
    if torch.is_tensor(tokens.get("attention_mask")):
        tokens["attention_mask"] = tokens["attention_mask"].to(device=device)
    return state, tokens, latent, clean, mask


def _build_stage2_audio_state(
    handoff: DFRStage2Handoff,
    stage_1_audio_for_stage2: dict[str, Any],
    device: torch.device,
    *,
    verify_handoff_audio: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = stage_1_audio_for_stage2.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 4:
        raise ValueError(
            "stage_1_audio_for_stage2 must be an LTX audio LATENT shaped [B,C,T,F]."
        )
    if tuple(samples.shape) != tuple(handoff.stage_1_audio_latent.shape):
        raise ValueError(
            "stage_1_audio_for_stage2 shape no longer matches the Stage-2A Stage-1 audio latent."
        )
    if verify_handoff_audio and _max_abs(samples, handoff.stage_1_audio_latent) != 0.0:
        raise ValueError(
            "stage_1_audio_for_stage2 no longer matches the exact Stage-2A Stage-1 audio latent."
        )

    initial = samples.to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    b, c, t, f = (int(x) for x in initial.shape)
    clean = initial.clone()
    mask_4d = torch.ones((b, 1, t, 1), device=device, dtype=torch.float32)
    positions = _audio_positions(
        batch=b,
        frames=t,
        sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        hop_length=DEFAULT_AUDIO_HOP_LENGTH,
        audio_latent_downsample_factor=DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        is_causal=DEFAULT_AUDIO_CAUSAL,
        shift=DEFAULT_AUDIO_SHIFT,
        device=device,
    )

    state = _new_audio_official_state(
        base_shape=(b, c, t, f),
        initial_latent=initial,
        clean_latent=clean,
        denoise_mask=mask_4d,
        positions=positions,
        duration_seconds=float(handoff.duration_seconds),
        video_alignment={
            "fps": float(handoff.fps),
            "stage": 2,
        },
        audio_config={
            "sample_rate": DEFAULT_AUDIO_SAMPLE_RATE,
            "hop_length": DEFAULT_AUDIO_HOP_LENGTH,
            "audio_latent_downsample_factor": DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
            "channels": c,
            "mel_bins": f,
            "is_causal": DEFAULT_AUDIO_CAUSAL,
            "shift": DEFAULT_AUDIO_SHIFT,
        },
    )
    state["operations"].append(
        {
            "type": "Stage2AudioInitialLatent",
            "source": "stage_1_audio_latent",
            "shape": tuple(int(x) for x in initial.shape),
        }
    )
    tokens = state["token_state"]
    latent = tokens["latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean_tokens = tokens["clean_latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = tokens["denoise_mask"].to(device=device, dtype=torch.float32)
    tokens["positions"] = tokens["positions"].to(device=device, dtype=torch.float32)
    return state, tokens, latent, clean_tokens, mask


def materialize_stage2_av_gaussian_noised_states(
    stage_2_handoff: DFRStage2Handoff,
    stage_2_video_state: dict[str, Any],
    stage_1_audio_for_stage2: dict[str, Any],
    stage_2_sigmas_tensor: torch.Tensor,
    *,
    device: torch.device | None = None,
    allow_experimental_extra_step: bool = False,
    require_official_sigmas: bool = True,
    collect_diagnostics: bool = True,
    require_reference: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Continue the Stage-1 RNG stream and noise Stage-2 VIDEO then AUDIO."""
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")
    sigmas = (
        _require_exact_stage2_sigmas(
            stage_2_sigmas_tensor,
            allow_experimental_extra_step=allow_experimental_extra_step,
        )
        if require_official_sigmas
        else validate_custom_sigma_schedule(stage_2_sigmas_tensor, "Custom Stage-2 sigmas")
    )
    noise_scale = float(sigmas[0].item())

    fallback = stage_2_video_state["samples"].device
    target_device = torch.device(device) if device is not None else _preferred_comfy_device(fallback)

    v_state, v_tokens, v_latent, v_clean, v_mask = _prepare_stage2_video(
        stage_2_video_state, target_device, require_reference=require_reference
    )
    a_state, a_tokens, a_latent, a_clean, a_mask = _build_stage2_audio_state(
        stage_2_handoff,
        stage_1_audio_for_stage2,
        target_device,
        verify_handoff_audio=collect_diagnostics,
    )

    if getattr(stage_2_handoff, "temporal_seams", ()):
        # Temporal resampling preserves the original audio. N -> 2N-1 at
        # doubled FPS changes N/fps by half a source frame, not the audio.
        # Record both timelines explicitly; never resize/reseed the audio.
        video_frames = (int(stage_2_video_state["samples"].shape[2])-1) * int(v_state.get("scale_factors", (8,32,32))[0]) + 1
        a_state["video_alignment"].update(
            mode="preserved_stage1_audio", pixel_frames=video_frames,
            fps=float(v_state.get("fps") or stage_2_handoff.fps),
            source_duration_seconds=float(stage_2_handoff.duration_seconds),
        )

    # Exact upstream continuation: the same generator object used in Stage 1 is
    # now consumed by Stage-2 video first and Stage-2 audio second.
    generator = generator_from_stage2_handoff(stage_2_handoff, target_device)
    rng_state_before = generator.get_state().detach().cpu().clone()
    if not torch.equal(rng_state_before, stage_2_handoff.rng_state_after_stage1_av.detach().cpu()):
        raise ValueError("Restored Stage-2 generator state does not match the Stage-2A continuation state.")

    video_noise = _randn_like_with_generator(v_latent, generator)
    audio_noise = _randn_like_with_generator(a_latent, generator)
    rng_state_after = generator.get_state().detach().cpu().clone()

    v_noised = official_gaussian_noiser_formula(v_latent, v_clean, v_mask, video_noise, noise_scale)
    a_noised = official_gaussian_noiser_formula(a_latent, a_clean, a_mask, audio_noise, noise_scale)

    common_meta = {
        "type": "GaussianNoiser",
        "stage": STAGE2_NOISER_STAGE,
        "seed": int(stage_2_handoff.seed),
        "noise_scale": noise_scale,
        "state_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "device_type": target_device.type,
        "shared_av_rng": True,
        "rng_order": STAGE2_RNG_ORDER,
        "rng_continuation": True,
        "rng_state_source": "stage2_handoff_after_stage1_av",
        "rng_state_before_stage2_av": rng_state_before,
        "rng_state_after_stage2_av": rng_state_after,
    }

    v_tokens["latent"] = v_noised
    v_tokens["clean_latent"] = v_clean
    v_tokens["denoise_mask"] = v_mask
    v_tokens[NOISER_METADATA_KEY] = {
        **common_meta,
        "modality": "video",
        "token_shape": tuple(int(x) for x in v_latent.shape),
    }
    v_state["operations"].append(
        {
            "type": "Stage2GaussianNoiser",
            "modality": "video",
            "noise_scale": noise_scale,
            "shared_av_rng": True,
            "rng_order": STAGE2_RNG_ORDER,
        }
    )

    a_tokens["latent"] = a_noised
    a_tokens["clean_latent"] = a_clean
    a_tokens["denoise_mask"] = a_mask
    a_tokens[NOISER_METADATA_KEY] = {
        **common_meta,
        "modality": "audio",
        "token_shape": tuple(int(x) for x in a_latent.shape),
    }
    a_state["operations"].append(
        {
            "type": "Stage2GaussianNoiser",
            "modality": "audio",
            "noise_scale": noise_scale,
            "shared_av_rng": True,
            "rng_order": STAGE2_RNG_ORDER,
        }
    )

    out_video = stage_2_video_state.copy()
    out_video[OFFICIAL_STATE_KEY] = v_state

    out_audio = {
        "samples": _unpatchify_audio(
            a_noised,
            channels=int(a_tokens["channels"]),
            mel_bins=int(a_tokens["mel_bins"]),
        ),
        AUDIO_OFFICIAL_STATE_KEY: a_state,
    }

    if not collect_diagnostics:
        return out_video, out_audio, ""

    fully_clean_video = int((v_mask == 0).sum().item())
    fully_noised_video = int((v_mask == 1).sum().item())
    report = (
        f"PASS=True; stage=2E_av_gaussian_noiser; seed={int(stage_2_handoff.seed)}; "
        f"seed_mode=continued_not_reseeded; noise_scale={noise_scale:.9g}; "
        f"sigma0={float(sigmas[0].item()):.9g}; rng_order={STAGE2_RNG_ORDER}; "
        f"device={target_device}; video_token_shape={tuple(int(x) for x in v_noised.shape)}; "
        f"audio_token_shape={tuple(int(x) for x in a_noised.shape)}; "
        f"video_fully_clean_tokens={fully_clean_video}; video_fully_noised_tokens={fully_noised_video}; "
        f"rng_state_bytes={int(rng_state_after.numel())}."
    )
    return out_video, out_audio, report


def _independently_replay_generator_to_stage2_start(
    handoff: DFRStage2Handoff,
    device: torch.device,
) -> torch.Generator:
    if not handoff.video_token_shape_at_stage1_noise or not handoff.audio_token_shape_at_stage1_noise:
        raise ValueError("Stage-2 handoff lacks Stage-1 token shapes required for independent RNG replay.")
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
    return generator


def validate_stage2_av_gaussian_noiser(
    stage_2_handoff: DFRStage2Handoff,
    pre_noise_stage_2_video_state: dict[str, Any],
    stage_1_audio_for_stage2: dict[str, Any],
    stage_2_sigmas_tensor: torch.Tensor,
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
) -> dict[str, Any]:
    """Independently replay the RNG sequence and verify exact Stage-2 AV noising."""
    sigmas = _require_exact_stage2_sigmas(stage_2_sigmas_tensor)
    noise_scale = float(sigmas[0].item())

    post_v_state = get_official_state(noised_stage_2_video_state)
    post_v = post_v_state["token_state"]
    post_a_state = get_audio_official_state(noised_stage_2_audio_state)
    post_a = post_a_state["token_state"]
    device = post_v["latent"].device
    if post_a["latent"].device != device:
        raise ValueError(
            f"Stage-2 AV noised states are on different devices: video={device}, audio={post_a['latent'].device}."
        )

    _, _, v_latent, v_clean, v_mask = _prepare_stage2_video(pre_noise_stage_2_video_state, device)
    _, _, a_latent, a_clean, a_mask = _build_stage2_audio_state(
        stage_2_handoff,
        stage_1_audio_for_stage2,
        device,
    )

    # Independent path: begin from seed, consume the exact Stage-1 video/audio
    # draws, then consume Stage-2 video/audio draws.  This proves continuation,
    # not just agreement with a saved metadata tensor.
    generator = _independently_replay_generator_to_stage2_start(stage_2_handoff, device)
    replayed_stage2_start = generator.get_state().detach().cpu().clone()
    expected_v_noise = _randn_like_with_generator(v_latent, generator)
    expected_a_noise = _randn_like_with_generator(a_latent, generator)
    expected_rng_after = generator.get_state().detach().cpu().clone()

    expected_v = official_gaussian_noiser_formula(v_latent, v_clean, v_mask, expected_v_noise, noise_scale)
    expected_a = official_gaussian_noiser_formula(a_latent, a_clean, a_mask, expected_a_noise, noise_scale)

    video_error = _max_abs(post_v["latent"], expected_v)
    audio_error = _max_abs(post_a["latent"], expected_a)
    video_clean_error = _max_abs(post_v["clean_latent"], v_clean)
    audio_clean_error = _max_abs(post_a["clean_latent"], a_clean)
    video_mask_error = _max_abs(post_v["denoise_mask"], v_mask)
    audio_mask_error = _max_abs(post_a["denoise_mask"], a_mask)

    vm = post_v.get(NOISER_METADATA_KEY) or {}
    am = post_a.get(NOISER_METADATA_KEY) or {}
    v_before = vm.get("rng_state_before_stage2_av")
    a_before = am.get("rng_state_before_stage2_av")
    v_after = vm.get("rng_state_after_stage2_av")
    a_after = am.get("rng_state_after_stage2_av")

    start_state_ok = (
        torch.equal(replayed_stage2_start, stage_2_handoff.rng_state_after_stage1_av.detach().cpu())
        and torch.is_tensor(v_before)
        and torch.is_tensor(a_before)
        and torch.equal(v_before.detach().cpu(), replayed_stage2_start)
        and torch.equal(a_before.detach().cpu(), replayed_stage2_start)
    )
    end_state_ok = (
        torch.is_tensor(v_after)
        and torch.is_tensor(a_after)
        and torch.equal(v_after.detach().cpu(), expected_rng_after)
        and torch.equal(a_after.detach().cpu(), expected_rng_after)
    )
    rng_start_error = 0.0 if start_state_ok else 1.0
    rng_end_error = 0.0 if end_state_ok else 1.0

    metadata_ok = (
        vm.get("stage") == STAGE2_NOISER_STAGE
        and am.get("stage") == STAGE2_NOISER_STAGE
        and vm.get("shared_av_rng") is True
        and am.get("shared_av_rng") is True
        and vm.get("rng_order") == STAGE2_RNG_ORDER
        and am.get("rng_order") == STAGE2_RNG_ORDER
        and vm.get("rng_continuation") is True
        and am.get("rng_continuation") is True
        and int(vm.get("seed", -1)) == int(stage_2_handoff.seed)
        and int(am.get("seed", -1)) == int(stage_2_handoff.seed)
        and abs(float(vm.get("noise_scale", -1.0)) - noise_scale) <= 1e-12
        and abs(float(am.get("noise_scale", -1.0)) - noise_scale) <= 1e-12
    )
    metadata_error = 0.0 if metadata_ok else 1.0

    # Stage-2 audio starts from Stage-1 audio; the clean copy in create_initial_state
    # must therefore equal that same latent before the all-one denoise mask makes the
    # noiser blend it with fresh noise.
    stage1_audio_initial_error = _max_abs(
        a_clean,
        _patchify_audio(stage_1_audio_for_stage2["samples"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)),
    )

    passed = all(
        value == 0.0
        for value in (
            video_error,
            audio_error,
            video_clean_error,
            audio_clean_error,
            video_mask_error,
            audio_mask_error,
            rng_start_error,
            rng_end_error,
            metadata_error,
            stage1_audio_initial_error,
        )
    )

    return {
        "passed": bool(passed),
        "video_error": float(video_error),
        "audio_error": float(audio_error),
        "video_clean_error": float(video_clean_error),
        "audio_clean_error": float(audio_clean_error),
        "video_mask_error": float(video_mask_error),
        "audio_mask_error": float(audio_mask_error),
        "rng_start_error": float(rng_start_error),
        "rng_end_error": float(rng_end_error),
        "metadata_error": float(metadata_error),
        "stage1_audio_initial_error": float(stage1_audio_initial_error),
        "seed": int(stage_2_handoff.seed),
        "noise_scale": noise_scale,
        "video_token_shape": tuple(int(x) for x in post_v["latent"].shape),
        "audio_token_shape": tuple(int(x) for x in post_a["latent"].shape),
        "device": str(device),
        "video_clean_tokens": int((post_v["denoise_mask"] == 0).sum().item()),
        "video_noised_tokens": int((post_v["denoise_mask"] == 1).sum().item()),
    }
