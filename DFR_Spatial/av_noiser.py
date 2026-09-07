"""Joint Stage-1 AV Gaussian noiser with the exact shared-RNG ordering used upstream.

Official DFR creates one torch.Generator(seed), wraps it in one GaussianNoiser, then
builds/noises VIDEO first and AUDIO second.  Therefore audio noise must be sampled
from the generator state *after* the video draw; reseeding the audio branch is wrong.
"""
from __future__ import annotations

from typing import Any
import torch

from .dfr_noiser import OFFICIAL_DFR_STATE_DTYPE, NOISER_METADATA_KEY, official_gaussian_noiser_formula
from .latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state
from .audio_state import (
    AUDIO_OFFICIAL_STATE_KEY,
    clone_audio_official_state,
    get_audio_official_state,
    _unpatchify_audio,
)


def _preferred_comfy_device(fallback: torch.device) -> torch.device:
    try:
        import comfy.model_management as mm  # type: ignore
        return torch.device(mm.get_torch_device())
    except Exception:
        return fallback


def _randn_with_generator(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _prepare_video(pre_video: dict[str, Any], device: torch.device):
    state = clone_or_create_official_state(pre_video)
    tokens = state.get("token_state")
    if tokens is None:
        raise ValueError("Video official state has no token_state. Apply generated-keyframe slots before AV noising.")
    if tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("Video state is already noised. Connect the pre-noise official video state.")
    latent = tokens["latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = tokens["clean_latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = tokens["denoise_mask"].to(device=device, dtype=torch.float32)
    return state, tokens, latent, clean, mask


def _prepare_audio(pre_audio: dict[str, Any], device: torch.device):
    state = clone_audio_official_state(pre_audio)
    tokens = state["token_state"]
    if tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("Audio state is already noised. Connect the pre-noise official audio state.")
    latent = tokens["latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = tokens["clean_latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = tokens["denoise_mask"].to(device=device, dtype=torch.float32)
    return state, tokens, latent, clean, mask


def materialize_stage1_av_gaussian_noised_states(
    video_official_state: dict[str, Any],
    audio_official_state: dict[str, Any],
    *,
    seed: int,
    noise_scale: float = 1.0,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    noise_scale = float(noise_scale)
    if not 0.0 <= noise_scale <= 1.0:
        raise ValueError(f"noise_scale must be in [0,1], got {noise_scale}.")

    fallback = video_official_state["samples"].device
    target_device = torch.device(device) if device is not None else _preferred_comfy_device(fallback)

    v_state, v_tokens, v_latent, v_clean, v_mask = _prepare_video(video_official_state, target_device)
    a_state, a_tokens, a_latent, a_clean, a_mask = _prepare_audio(audio_official_state, target_device)

    # Upstream uses ONE generator/noiser object. DiffusionStage builds video first, audio second.
    generator = torch.Generator(device=target_device).manual_seed(int(seed))
    video_noise = _randn_with_generator(v_latent, generator)
    audio_noise = _randn_with_generator(a_latent, generator)
    # Upstream DFR reuses this *same* GaussianNoiser/generator object for Stage 2.
    # Capture the generator state immediately after the Stage-1 video->audio draws so
    # the Comfy parity path can continue the random stream instead of reseeding.
    rng_state_after_stage1_av = generator.get_state().detach().cpu().clone()

    v_noised = official_gaussian_noiser_formula(v_latent, v_clean, v_mask, video_noise, noise_scale)
    a_noised = official_gaussian_noiser_formula(a_latent, a_clean, a_mask, audio_noise, noise_scale)

    common_meta = {
        "type": "GaussianNoiser",
        "seed": int(seed),
        "noise_scale": noise_scale,
        "state_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "device_type": target_device.type,
        "shared_av_rng": True,
        "rng_order": "video_then_audio",
        "rng_state_schema": "torch.Generator state after Stage-1 video then audio draws",
        "rng_state_after_stage1_av": rng_state_after_stage1_av,
    }

    v_tokens["latent"] = v_noised
    v_tokens["clean_latent"] = v_clean
    v_tokens["denoise_mask"] = v_mask
    v_tokens["positions"] = v_tokens["positions"].to(device=target_device, dtype=torch.float32)
    if torch.is_tensor(v_tokens.get("keyframes_mask")):
        v_tokens["keyframes_mask"] = v_tokens["keyframes_mask"].to(device=target_device, dtype=torch.float32)
    v_tokens[NOISER_METADATA_KEY] = {**common_meta, "modality": "video", "token_shape": tuple(v_latent.shape)}
    v_state["operations"].append({
        **{k: v for k, v in common_meta.items() if k != "rng_state_after_stage1_av"},
        "modality": "video",
        "rng_state_bytes": int(rng_state_after_stage1_av.numel()),
    })

    a_tokens["latent"] = a_noised
    a_tokens["clean_latent"] = a_clean
    a_tokens["denoise_mask"] = a_mask
    a_tokens["positions"] = a_tokens["positions"].to(device=target_device, dtype=torch.float32)
    a_tokens[NOISER_METADATA_KEY] = {**common_meta, "modality": "audio", "token_shape": tuple(a_latent.shape)}
    a_state["operations"].append({
        **{k: v for k, v in common_meta.items() if k != "rng_state_after_stage1_av"},
        "modality": "audio",
        "rng_state_bytes": int(rng_state_after_stage1_av.numel()),
    })

    out_video = video_official_state.copy()
    out_video[OFFICIAL_STATE_KEY] = v_state

    out_audio = {
        "samples": _unpatchify_audio(a_noised, channels=int(a_tokens["channels"]), mel_bins=int(a_tokens["mel_bins"])),
        AUDIO_OFFICIAL_STATE_KEY: a_state,
    }

    report = {
        "seed": int(seed),
        "noise_scale": noise_scale,
        "device": str(target_device),
        "video_token_shape": tuple(int(x) for x in v_noised.shape),
        "audio_token_shape": tuple(int(x) for x in a_noised.shape),
        "rng_order": "video_then_audio",
        "rng_state_bytes": int(rng_state_after_stage1_av.numel()),
    }
    return out_video, out_audio, report


def validate_stage1_av_gaussian_noiser(
    pre_video_state: dict[str, Any],
    pre_audio_state: dict[str, Any],
    noised_video_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    *,
    seed: int,
    noise_scale: float,
) -> dict[str, Any]:
    v_post = noised_video_state[OFFICIAL_STATE_KEY]["token_state"]
    a_post = get_audio_official_state(noised_audio_state)["token_state"]
    device = v_post["latent"].device
    if a_post["latent"].device != device:
        raise ValueError(f"AV noised states are on different devices: video={device}, audio={a_post['latent'].device}.")

    _, _, v_latent, v_clean, v_mask = _prepare_video(pre_video_state, device)
    _, _, a_latent, a_clean, a_mask = _prepare_audio(pre_audio_state, device)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    expected_v_noise = _randn_with_generator(v_latent, generator)
    expected_a_noise = _randn_with_generator(a_latent, generator)
    expected_rng_state = generator.get_state().detach().cpu()
    expected_v = official_gaussian_noiser_formula(v_latent, v_clean, v_mask, expected_v_noise, float(noise_scale))
    expected_a = official_gaussian_noiser_formula(a_latent, a_clean, a_mask, expected_a_noise, float(noise_scale))

    video_err = float((v_post["latent"] - expected_v).abs().max().item()) if expected_v.numel() else 0.0
    audio_err = float((a_post["latent"] - expected_a).abs().max().item()) if expected_a.numel() else 0.0
    video_clean_err = float((v_post["clean_latent"] - v_clean).abs().max().item()) if v_clean.numel() else 0.0
    audio_clean_err = float((a_post["clean_latent"] - a_clean).abs().max().item()) if a_clean.numel() else 0.0
    video_mask_err = float((v_post["denoise_mask"] - v_mask).abs().max().item()) if v_mask.numel() else 0.0
    audio_mask_err = float((a_post["denoise_mask"] - a_mask).abs().max().item()) if a_mask.numel() else 0.0

    vm = v_post.get(NOISER_METADATA_KEY) or {}
    am = a_post.get(NOISER_METADATA_KEY) or {}
    v_rng_state = vm.get("rng_state_after_stage1_av")
    a_rng_state = am.get("rng_state_after_stage1_av")
    rng_state_ok = (
        torch.is_tensor(v_rng_state)
        and torch.is_tensor(a_rng_state)
        and torch.equal(v_rng_state.detach().cpu(), expected_rng_state)
        and torch.equal(a_rng_state.detach().cpu(), expected_rng_state)
    )
    metadata_ok = (
        vm.get("shared_av_rng") is True
        and am.get("shared_av_rng") is True
        and vm.get("rng_order") == "video_then_audio"
        and am.get("rng_order") == "video_then_audio"
        and int(vm.get("seed", -1)) == int(seed)
        and int(am.get("seed", -1)) == int(seed)
        and abs(float(vm.get("noise_scale", -1)) - float(noise_scale)) <= 1e-12
        and abs(float(am.get("noise_scale", -1)) - float(noise_scale)) <= 1e-12
        and rng_state_ok
    )
    metadata_error = 0.0 if metadata_ok else 1.0
    passed = all(x == 0.0 for x in (video_err, audio_err, video_clean_err, audio_clean_err, video_mask_err, audio_mask_err, metadata_error))
    return {
        "passed": passed,
        "video_error": video_err,
        "audio_error": audio_err,
        "video_clean_error": video_clean_err,
        "audio_clean_error": audio_clean_err,
        "video_mask_error": video_mask_err,
        "audio_mask_error": audio_mask_err,
        "metadata_error": metadata_error,
        "metadata_ok": metadata_ok,
        "rng_state_ok": bool(rng_state_ok),
        "rng_state_bytes": int(expected_rng_state.numel()),
        "device": str(device),
        "video_token_shape": tuple(int(x) for x in v_post["latent"].shape),
        "audio_token_shape": tuple(int(x) for x in a_post["latent"].shape),
    }
