"""Strict official LTXAV audio latent-state helpers for the Spatial DFR parity path.

This module intentionally keeps audio separate from the existing video official state so the
Phase-B video sampler work remains untouched.  The audio state is carried in its own Comfy
``LATENT`` dict, mirroring the upstream LTX-2 `AudioLatentTools + create_noised_state`
semantics for the no-conditioning Stage-1 audio branch.
"""

from __future__ import annotations

from typing import Any

import torch

from .dfr_noiser import (
    NOISER_METADATA_KEY,
    OFFICIAL_DFR_STATE_DTYPE,
    official_gaussian_noiser_formula,
    _sample_official_noise,
)
from .latent_state import get_official_state


AUDIO_OFFICIAL_STATE_KEY = "_ltx_dfr_audio_official_state"
AUDIO_OFFICIAL_STATE_VERSION = 1

DEFAULT_AUDIO_CHANNELS = 8
DEFAULT_AUDIO_MEL_BINS = 16
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_HOP_LENGTH = 160
DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
DEFAULT_AUDIO_CAUSAL = True
DEFAULT_AUDIO_SHIFT = 0


def _clone_tensor_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.clone() if torch.is_tensor(value) else None


def _validate_audio_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,T,F], got {tuple(tensor.shape)}.")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(f"{name} must have strictly positive dimensions, got {tuple(tensor.shape)}.")


def _patchify_audio(audio_latent: torch.Tensor) -> torch.Tensor:
    _validate_audio_tensor(audio_latent, "audio latent")
    return audio_latent.permute(0, 2, 1, 3).reshape(audio_latent.shape[0], audio_latent.shape[2], -1).contiguous()


def _patchify_audio_mask(audio_mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(audio_mask) or audio_mask.ndim != 4 or audio_mask.shape[1] != 1 or audio_mask.shape[3] != 1:
        raise ValueError(f"audio mask must have shape [B,1,T,1], got {getattr(audio_mask, 'shape', None)}.")
    return audio_mask.permute(0, 2, 1, 3).reshape(audio_mask.shape[0], audio_mask.shape[2], 1).contiguous()


def _unpatchify_audio(audio_tokens: torch.Tensor, *, channels: int, mel_bins: int) -> torch.Tensor:
    if not torch.is_tensor(audio_tokens) or audio_tokens.ndim != 3:
        raise ValueError(f"audio token state must have shape [B,T,CF], got {getattr(audio_tokens, 'shape', None)}.")
    b, t, cf = audio_tokens.shape
    expected = int(channels) * int(mel_bins)
    if cf != expected:
        raise ValueError(f"audio token dimension {cf} does not match channels*mel_bins={expected}.")
    return audio_tokens.reshape(b, t, channels, mel_bins).permute(0, 2, 1, 3).contiguous()


def _audio_latent_time_in_sec(
    start_latent: int,
    end_latent: int,
    *,
    sample_rate: int,
    hop_length: int,
    audio_latent_downsample_factor: int,
    is_causal: bool,
    device: torch.device,
) -> torch.Tensor:
    frame = torch.arange(start_latent, end_latent, dtype=torch.float32, device=device)
    mel_frame = frame * float(audio_latent_downsample_factor)
    if is_causal:
        mel_frame = (mel_frame + 1.0 - float(audio_latent_downsample_factor)).clamp(min=0.0)
    return mel_frame * (float(hop_length) / float(sample_rate))


def _audio_positions(
    *,
    batch: int,
    frames: int,
    sample_rate: int,
    hop_length: int,
    audio_latent_downsample_factor: int,
    is_causal: bool,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    starts = _audio_latent_time_in_sec(
        shift,
        frames + shift,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
        is_causal=is_causal,
        device=device,
    )
    ends = _audio_latent_time_in_sec(
        shift + 1,
        frames + shift + 1,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
        is_causal=is_causal,
        device=device,
    )
    starts = starts.unsqueeze(0).expand(batch, -1).unsqueeze(1)
    ends = ends.unsqueeze(0).expand(batch, -1).unsqueeze(1)
    return torch.stack([starts, ends], dim=-1).contiguous()


def _audio_frames_from_duration(
    *,
    duration: float,
    sample_rate: int,
    hop_length: int,
    audio_latent_downsample_factor: int,
) -> int:
    latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
    return int(round(float(duration) * latents_per_second))


def _video_pixel_frames_from_state(video_official_state: dict[str, Any]) -> int:
    state = get_official_state(video_official_state)
    base_shape = tuple(int(x) for x in state["base_shape"])
    if len(base_shape) != 5:
        raise ValueError(f"Official video state base_shape must be [B,C,T,H,W], got {base_shape}.")
    token_state = state.get("token_state") or {}
    scale = tuple(int(x) for x in (token_state.get("scale_factors") or state.get("scale_factors") or (8, 32, 32)))
    temporal_scale = int(scale[0])
    return (int(base_shape[2]) - 1) * temporal_scale + 1


def _video_fps_from_state(video_official_state: dict[str, Any]) -> float:
    state = get_official_state(video_official_state)
    token_state = state.get("token_state") or {}
    fps = token_state.get("fps", state.get("fps"))
    if fps is None:
        raise ValueError(
            "Could not derive fps from the official video state. Connect the state after a node that records fps "
            "(for example the official keyframe/generated-slot nodes)."
        )
    return float(fps)


def _new_audio_official_state(
    *,
    base_shape: tuple[int, int, int, int],
    initial_latent: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    positions: torch.Tensor,
    duration_seconds: float,
    video_alignment: dict[str, Any],
    audio_config: dict[str, Any],
) -> dict[str, Any]:
    token_state = {
        "latent": _patchify_audio(initial_latent),
        "clean_latent": _patchify_audio(clean_latent),
        "denoise_mask": _patchify_audio_mask(denoise_mask).to(torch.float32),
        "positions": positions.to(torch.float32),
        "attention_mask": None,
        "audio_shape": tuple(int(x) for x in base_shape),
        "channels": int(base_shape[1]),
        "mel_bins": int(base_shape[3]),
        "audio_token_count": int(base_shape[2]),
        "sample_rate": int(audio_config["sample_rate"]),
        "hop_length": int(audio_config["hop_length"]),
        "audio_latent_downsample_factor": int(audio_config["audio_latent_downsample_factor"]),
        "is_causal": bool(audio_config["is_causal"]),
        "shift": int(audio_config["shift"]),
        NOISER_METADATA_KEY: None,
    }
    return {
        "version": AUDIO_OFFICIAL_STATE_VERSION,
        "base_shape": tuple(int(x) for x in base_shape),
        "duration_seconds": float(duration_seconds),
        "audio_config": dict(audio_config),
        "video_alignment": dict(video_alignment),
        "operations": [
            {
                "type": "AudioInitialState",
                "audio_shape": tuple(int(x) for x in base_shape),
                "duration_seconds": float(duration_seconds),
            }
        ],
        "token_state": token_state,
    }


def get_audio_official_state(latent: dict[str, Any]) -> dict[str, Any]:
    if "samples" not in latent:
        raise ValueError("Input LATENT has no 'samples' tensor.")
    samples = latent["samples"]
    _validate_audio_tensor(samples, "audio latent")
    existing = latent.get(AUDIO_OFFICIAL_STATE_KEY)
    if existing is None:
        raise ValueError(
            f"LATENT does not carry {AUDIO_OFFICIAL_STATE_KEY}. Use 'LTX Official: Create Stage 1 Audio State' first."
        )
    if not isinstance(existing, dict):
        raise ValueError(f"Malformed {AUDIO_OFFICIAL_STATE_KEY}: expected dict.")
    if int(existing.get("version", -1)) != AUDIO_OFFICIAL_STATE_VERSION:
        raise ValueError(
            f"Unsupported audio official-state version {existing.get('version')}; expected {AUDIO_OFFICIAL_STATE_VERSION}."
        )
    if tuple(existing.get("base_shape", ())) != tuple(samples.shape):
        raise ValueError(
            f"Audio official state was created for shape {existing.get('base_shape')} but current samples are {tuple(samples.shape)}."
        )
    return existing


def clone_audio_official_state(latent: dict[str, Any]) -> dict[str, Any]:
    state = get_audio_official_state(latent)
    token_state = state["token_state"]
    return {
        "version": int(state["version"]),
        "base_shape": tuple(int(x) for x in state["base_shape"]),
        "duration_seconds": float(state["duration_seconds"]),
        "audio_config": dict(state.get("audio_config", {})),
        "video_alignment": dict(state.get("video_alignment", {})),
        "operations": list(state.get("operations", [])),
        "token_state": {
            "latent": token_state["latent"].clone(),
            "clean_latent": token_state["clean_latent"].clone(),
            "denoise_mask": token_state["denoise_mask"].clone(),
            "positions": token_state["positions"].clone(),
            "attention_mask": _clone_tensor_or_none(token_state.get("attention_mask")),
            "audio_shape": tuple(int(x) for x in token_state["audio_shape"]),
            "channels": int(token_state["channels"]),
            "mel_bins": int(token_state["mel_bins"]),
            "audio_token_count": int(token_state["audio_token_count"]),
            "sample_rate": int(token_state["sample_rate"]),
            "hop_length": int(token_state["hop_length"]),
            "audio_latent_downsample_factor": int(token_state["audio_latent_downsample_factor"]),
            "is_causal": bool(token_state["is_causal"]),
            "shift": int(token_state["shift"]),
            NOISER_METADATA_KEY: dict(token_state[NOISER_METADATA_KEY]) if isinstance(token_state.get(NOISER_METADATA_KEY), dict) else None,
        },
    }


def create_audio_official_state_from_video(
    video_official_state: dict[str, Any],
    *,
    channels: int = DEFAULT_AUDIO_CHANNELS,
    mel_bins: int = DEFAULT_AUDIO_MEL_BINS,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    hop_length: int = DEFAULT_AUDIO_HOP_LENGTH,
    audio_latent_downsample_factor: int = DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = DEFAULT_AUDIO_CAUSAL,
    shift: int = DEFAULT_AUDIO_SHIFT,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    video_state = get_official_state(video_official_state)
    video_samples = video_official_state["samples"]
    base_b = int(video_samples.shape[0])
    pixel_frames = _video_pixel_frames_from_state(video_official_state)
    fps = _video_fps_from_state(video_official_state)
    duration_seconds = float(pixel_frames) / float(fps)
    audio_frames = _audio_frames_from_duration(
        duration=duration_seconds,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
    )
    if audio_frames <= 0:
        raise ValueError(f"Resolved non-positive audio latent frame count {audio_frames} from duration {duration_seconds} s.")

    resolved_dtype = dtype or video_samples.dtype
    base_shape = (base_b, int(channels), int(audio_frames), int(mel_bins))
    initial_latent = torch.zeros(base_shape, dtype=resolved_dtype, device=video_samples.device)
    clean_latent = initial_latent.clone()
    denoise_mask = torch.ones((base_b, 1, audio_frames, 1), dtype=torch.float32, device=video_samples.device)
    positions = _audio_positions(
        batch=base_b,
        frames=audio_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
        is_causal=is_causal,
        shift=shift,
        device=video_samples.device,
    )

    audio_state = _new_audio_official_state(
        base_shape=base_shape,
        initial_latent=initial_latent,
        clean_latent=clean_latent,
        denoise_mask=denoise_mask,
        positions=positions,
        duration_seconds=duration_seconds,
        video_alignment={
            "pixel_frames": int(pixel_frames),
            "fps": float(fps),
            "video_base_shape": tuple(int(x) for x in video_state["base_shape"]),
        },
        audio_config={
            "sample_rate": int(sample_rate),
            "hop_length": int(hop_length),
            "audio_latent_downsample_factor": int(audio_latent_downsample_factor),
            "channels": int(channels),
            "mel_bins": int(mel_bins),
            "is_causal": bool(is_causal),
            "shift": int(shift),
        },
    )

    out = {"samples": initial_latent.clone()}
    out[AUDIO_OFFICIAL_STATE_KEY] = audio_state
    return out


def audio_state_stats(audio_official_state: dict[str, Any]) -> dict[str, Any]:
    state = get_audio_official_state(audio_official_state)
    token_state = state["token_state"]
    latent = token_state["latent"]
    positions = token_state["positions"]
    return {
        "audio_shape": tuple(int(x) for x in state["base_shape"]),
        "token_shape": tuple(int(x) for x in latent.shape),
        "duration_seconds": float(state["duration_seconds"]),
        "pixel_frames": int(state.get("video_alignment", {}).get("pixel_frames", -1)),
        "fps": float(state.get("video_alignment", {}).get("fps", -1.0)),
        "sample_rate": int(token_state["sample_rate"]),
        "hop_length": int(token_state["hop_length"]),
        "audio_latent_downsample_factor": int(token_state["audio_latent_downsample_factor"]),
        "pos0_start": float(positions[0, 0, 0, 0].item()) if positions.numel() else 0.0,
        "pos0_end": float(positions[0, 0, 0, 1].item()) if positions.numel() else 0.0,
        "pos_last_start": float(positions[0, 0, -1, 0].item()) if positions.numel() else 0.0,
        "pos_last_end": float(positions[0, 0, -1, 1].item()) if positions.numel() else 0.0,
    }


def validate_audio_state_against_video(audio_official_state: dict[str, Any], video_official_state: dict[str, Any]) -> dict[str, Any]:
    audio_state = get_audio_official_state(audio_official_state)
    token_state = audio_state["token_state"]
    expected_pixel_frames = _video_pixel_frames_from_state(video_official_state)
    expected_fps = _video_fps_from_state(video_official_state)
    expected_duration = float(expected_pixel_frames) / float(expected_fps)
    expected_frames = _audio_frames_from_duration(
        duration=expected_duration,
        sample_rate=int(token_state["sample_rate"]),
        hop_length=int(token_state["hop_length"]),
        audio_latent_downsample_factor=int(token_state["audio_latent_downsample_factor"]),
    )
    actual_shape = tuple(int(x) for x in audio_state["base_shape"])
    expected_shape = (actual_shape[0], actual_shape[1], expected_frames, actual_shape[3])
    shape_error = 0.0 if actual_shape == expected_shape else 1.0

    expected_pos = _audio_positions(
        batch=actual_shape[0],
        frames=expected_frames,
        sample_rate=int(token_state["sample_rate"]),
        hop_length=int(token_state["hop_length"]),
        audio_latent_downsample_factor=int(token_state["audio_latent_downsample_factor"]),
        is_causal=bool(token_state["is_causal"]),
        shift=int(token_state["shift"]),
        device=token_state["positions"].device,
    )
    position_error = float((token_state["positions"] - expected_pos).abs().max().item()) if expected_pos.numel() else 0.0
    duration_error = abs(float(audio_state["duration_seconds"]) - expected_duration)
    token_shape_expected = (actual_shape[0], expected_frames, actual_shape[1] * actual_shape[3])
    token_shape_error = 0.0 if tuple(int(x) for x in token_state["latent"].shape) == token_shape_expected else 1.0
    mask_shape_error = 0.0 if tuple(int(x) for x in token_state["denoise_mask"].shape) == (actual_shape[0], expected_frames, 1) else 1.0
    all_one_mask = bool(torch.all(token_state["denoise_mask"] == 1.0).item())

    passed = (
        shape_error == 0.0
        and position_error == 0.0
        and duration_error <= 1e-12
        and token_shape_error == 0.0
        and mask_shape_error == 0.0
        and all_one_mask
    )
    return {
        "passed": passed,
        "shape_error": shape_error,
        "position_error": position_error,
        "duration_error": duration_error,
        "token_shape_error": token_shape_error,
        "mask_shape_error": mask_shape_error,
        "all_one_mask": all_one_mask,
        "audio_shape": actual_shape,
        "expected_shape": expected_shape,
        "token_shape": tuple(int(x) for x in token_state["latent"].shape),
        "expected_token_shape": token_shape_expected,
        "expected_duration": expected_duration,
        "pixel_frames": expected_pixel_frames,
        "fps": expected_fps,
    }


def materialize_audio_gaussian_noised_state(
    audio_official_state: dict[str, Any],
    *,
    seed: int,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    state = clone_audio_official_state(audio_official_state)
    token_state = state["token_state"]
    if token_state.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("Audio official state already contains Gaussian noiser metadata; do not noise it twice.")

    latent0 = token_state["latent"].to(dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = token_state["clean_latent"].to(device=latent0.device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = token_state["denoise_mask"].to(device=latent0.device, dtype=torch.float32)
    noise = _sample_official_noise(latent0, seed=int(seed))
    noised = official_gaussian_noiser_formula(latent0, clean, mask, noise, float(noise_scale))

    token_state["latent"] = noised
    token_state["clean_latent"] = clean
    token_state["denoise_mask"] = mask
    token_state["positions"] = token_state["positions"].to(device=latent0.device, dtype=torch.float32)
    token_state[NOISER_METADATA_KEY] = {
        "seed": int(seed),
        "noise_scale": float(noise_scale),
        "state_dtype": str(OFFICIAL_DFR_STATE_DTYPE).replace("torch.", ""),
        "noise_dtype": str(OFFICIAL_DFR_STATE_DTYPE).replace("torch.", ""),
        "device_type": latent0.device.type,
    }

    out = {"samples": _unpatchify_audio(noised, channels=int(token_state["channels"]), mel_bins=int(token_state["mel_bins"]))}
    out[AUDIO_OFFICIAL_STATE_KEY] = state
    return out


def validate_audio_gaussian_noiser(
    pre_noise_audio_state: dict[str, Any],
    noised_audio_state: dict[str, Any],
    *,
    seed: int,
    noise_scale: float,
) -> dict[str, Any]:
    pre = get_audio_official_state(pre_noise_audio_state)
    post = get_audio_official_state(noised_audio_state)
    pre_tokens = pre["token_state"]
    post_tokens = post["token_state"]
    if pre_tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("pre_noise_audio_state is already noised; connect the state immediately before the audio noiser.")
    metadata = post_tokens.get(NOISER_METADATA_KEY)
    if metadata is None:
        raise ValueError("noised_audio_state has no audio Gaussian noiser metadata. Connect the audio noiser output.")

    latent0 = pre_tokens["latent"].to(device=post_tokens["latent"].device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = pre_tokens["clean_latent"].to(device=post_tokens["latent"].device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = pre_tokens["denoise_mask"].to(device=post_tokens["latent"].device, dtype=torch.float32)
    noise = _sample_official_noise(latent0, seed=int(seed))
    expected = official_gaussian_noiser_formula(latent0, clean, mask, noise, float(noise_scale))

    actual = post_tokens["latent"]
    latent_err = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
    clean_err = float((post_tokens["clean_latent"] - clean).abs().max().item()) if clean.numel() else 0.0
    mask_err = float((post_tokens["denoise_mask"] - mask).abs().max().item()) if mask.numel() else 0.0
    pos_err = float(
        (post_tokens["positions"] - pre_tokens["positions"].to(device=post_tokens["positions"].device, dtype=torch.float32)).abs().max().item()
    ) if post_tokens["positions"].numel() else 0.0
    dtype_ok = (
        actual.dtype == torch.bfloat16
        and post_tokens["clean_latent"].dtype == torch.bfloat16
        and post_tokens["denoise_mask"].dtype == torch.float32
    )
    metadata_ok = (
        int(metadata.get("seed", -1)) == int(seed)
        and abs(float(metadata.get("noise_scale", -999.0)) - float(noise_scale)) <= 1e-12
        and metadata.get("state_dtype") == "bfloat16"
        and metadata.get("noise_dtype") == "bfloat16"
    )
    structure_ok = (
        tuple(int(x) for x in pre["base_shape"]) == tuple(int(x) for x in post["base_shape"])
        and pre.get("audio_config") == post.get("audio_config")
        and pre.get("video_alignment") == post.get("video_alignment")
    )
    metadata_error = 0.0 if (dtype_ok and metadata_ok and structure_ok and pos_err == 0.0) else 1.0
    pure_noise_err = float((actual - noise).abs().max().item()) if actual.numel() else 0.0

    passed = latent_err == 0.0 and clean_err == 0.0 and mask_err == 0.0 and metadata_error == 0.0
    return {
        "passed": passed,
        "latent_error": latent_err,
        "clean_error": clean_err,
        "mask_error": mask_err,
        "position_error": pos_err,
        "metadata_error": metadata_error,
        "pure_noise_error": pure_noise_err,
        "token_shape": tuple(int(x) for x in actual.shape),
        "dtype_ok": dtype_ok,
        "metadata_ok": metadata_ok,
        "structure_ok": structure_ok,
    }
