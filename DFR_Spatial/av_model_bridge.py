"""Joint Stage-1 AV model-input bridge for the strict LTX DFR parity path.

This module is the first half of the audio-video execution bridge:
- accept the already-noised official VIDEO token state;
- accept the already-noised official AUDIO token state;
- verify the two states are mutually compatible;
- build one deterministic joint model-input description in the same ordering used
  upstream (video tokens first, audio tokens second).

It intentionally stops *before* the real transformer call. The returned object keeps
video and audio token tensors separate because their latent feature dimensions are
modality-specific before the joint transformer input projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .audio_state import get_audio_official_state, validate_audio_state_against_video
from .dfr_noiser import NOISER_METADATA_KEY
from .latent_state import get_official_state
from .frame_rate import official_conditioning_fps, model_video_positions


AV_MODEL_INPUT_VERSION = 1


@dataclass
class DFRStage1AVModelInput:
    """Pure-data description of one joint Stage-1 LTXAV model input.

    ``video_latent`` and ``audio_latent`` intentionally remain separate because the
    raw token feature size is modality-specific (e.g. video latent channels vs.
    audio spectrogram channels×mel bins).  The "joint" part of this object is the
    shared ordering/bookkeeping that says how a true multimodal execution should
    treat the two streams as one sequence: video tokens first, audio tokens second.
    """

    video_latent: torch.Tensor                 # [B, Nv, Dv]
    audio_latent: torch.Tensor                 # [B, Na, Da]
    video_denoise_mask: torch.Tensor           # [B, Nv, 1]
    audio_denoise_mask: torch.Tensor           # [B, Na, 1]
    joint_denoise_mask: torch.Tensor           # [B, Nv+Na, 1]
    joint_modality_ids: torch.Tensor           # [B, Nv+Na, 1], 0=video, 1=audio
    video_positions: torch.Tensor              # [B, 3, Nv, 2]
    audio_positions: torch.Tensor              # [B, 1, Na, 2]
    video_keyframes_mask: torch.Tensor | None  # [B, Nv, 1] or None
    video_token_count: int
    audio_token_count: int
    total_token_count: int
    video_feature_dim: int
    audio_feature_dim: int
    batch_size: int
    video_token_range: tuple[int, int]
    audio_token_range: tuple[int, int]
    frame_rate: float
    duration_seconds: float
    video_base_shape: tuple[int, int, int, int, int]
    audio_base_shape: tuple[int, int, int, int]
    video_scale_factors: tuple[int, int, int]
    audio_config: dict[str, Any]
    modality_order: tuple[str, str] = ("video", "audio")
    version: int = AV_MODEL_INPUT_VERSION


def _clone_tensor_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.clone() if torch.is_tensor(value) else None


def _require_noised_video_state(video_official_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = get_official_state(video_official_state)
    token_state = state.get("token_state")
    if token_state is None:
        raise ValueError(
            "Video official state has no token_state. Complete the official video-conditioning path first."
        )
    if token_state.get(NOISER_METADATA_KEY) is None:
        raise ValueError(
            "Joint AV model-input bridging requires the Gaussian-noised VIDEO official state. "
            "Apply 'LTX Official: Stage 1 AV Gaussian Noiser' or 'LTX Official: Gaussian Noiser (DFR)' first."
        )
    return state, token_state


def _require_noised_audio_state(audio_official_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = get_audio_official_state(audio_official_state)
    token_state = state["token_state"]
    if token_state.get(NOISER_METADATA_KEY) is None:
        raise ValueError(
            "Joint AV model-input bridging requires the Gaussian-noised AUDIO official state. "
            "Apply 'LTX Official: Stage 1 AV Gaussian Noiser' or 'LTX Official: Gaussian Noiser (Audio DFR)' first."
        )
    return state, token_state


def _resolve_joint_device(video_tokens: dict[str, Any], audio_tokens: dict[str, Any]) -> torch.device:
    video_device = video_tokens["latent"].device
    audio_device = audio_tokens["latent"].device
    if video_device == audio_device:
        return video_device
    return video_device


def materialize_stage1_av_model_input(
    video_official_state: dict[str, Any],
    audio_official_state: dict[str, Any],
) -> DFRStage1AVModelInput:
    """Build the deterministic joint AV transformer-input description.

    Ordering follows upstream Stage-1 AV execution:
    VIDEO tokens first, AUDIO tokens second.
    """
    v_state, v_tokens = _require_noised_video_state(video_official_state)
    a_state, a_tokens = _require_noised_audio_state(audio_official_state)

    if v_tokens["latent"].ndim != 3:
        raise ValueError(f"Video token latent must be [B,N,D], got {tuple(v_tokens['latent'].shape)}.")
    if a_tokens["latent"].ndim != 3:
        raise ValueError(f"Audio token latent must be [B,N,D], got {tuple(a_tokens['latent'].shape)}.")
    if v_tokens["denoise_mask"].ndim != 3 or v_tokens["denoise_mask"].shape[-1] != 1:
        raise ValueError(f"Video denoise_mask must be [B,N,1], got {tuple(v_tokens['denoise_mask'].shape)}.")
    if a_tokens["denoise_mask"].ndim != 3 or a_tokens["denoise_mask"].shape[-1] != 1:
        raise ValueError(f"Audio denoise_mask must be [B,N,1], got {tuple(a_tokens['denoise_mask'].shape)}.")
    if v_tokens["positions"].ndim != 4 or v_tokens["positions"].shape[1] != 3 or v_tokens["positions"].shape[-1] != 2:
        raise ValueError(f"Video positions must be [B,3,N,2], got {tuple(v_tokens['positions'].shape)}.")
    if a_tokens["positions"].ndim != 4 or a_tokens["positions"].shape[1] != 1 or a_tokens["positions"].shape[-1] != 2:
        raise ValueError(f"Audio positions must be [B,1,N,2], got {tuple(a_tokens['positions'].shape)}.")

    batch = int(v_tokens["latent"].shape[0])
    if int(a_tokens["latent"].shape[0]) != batch:
        raise ValueError(
            f"Video/audio batch mismatch: video batch={batch}, audio batch={int(a_tokens['latent'].shape[0])}."
        )

    alignment = validate_audio_state_against_video(audio_official_state, video_official_state)
    if not bool(alignment["passed"]):
        raise ValueError(
            "Audio official state is not aligned to the connected video state. "
            f"Validation result: {alignment}."
        )

    device = _resolve_joint_device(v_tokens, a_tokens)
    video_latent = v_tokens["latent"].to(device=device).clone()
    audio_latent = a_tokens["latent"].to(device=device).clone()
    video_mask = v_tokens["denoise_mask"].to(device=device, dtype=torch.float32).clone()
    audio_mask = a_tokens["denoise_mask"].to(device=device, dtype=torch.float32).clone()
    video_positions = v_tokens["positions"].to(device=device, dtype=torch.float32).clone()
    audio_positions = a_tokens["positions"].to(device=device, dtype=torch.float32).clone()
    video_keyframes_mask = _clone_tensor_or_none(v_tokens.get("keyframes_mask"))
    if video_keyframes_mask is not None:
        video_keyframes_mask = video_keyframes_mask.to(device=device, dtype=torch.float32)

    video_token_count = int(video_latent.shape[1])
    audio_token_count = int(audio_latent.shape[1])
    total_token_count = video_token_count + audio_token_count
    if int(video_positions.shape[2]) != video_token_count:
        raise ValueError(
            f"Video position count {int(video_positions.shape[2])} does not match token count {video_token_count}."
        )
    if int(audio_positions.shape[2]) != audio_token_count:
        raise ValueError(
            f"Audio position count {int(audio_positions.shape[2])} does not match token count {audio_token_count}."
        )
    if int(video_mask.shape[1]) != video_token_count:
        raise ValueError(
            f"Video mask count {int(video_mask.shape[1])} does not match token count {video_token_count}."
        )
    if int(audio_mask.shape[1]) != audio_token_count:
        raise ValueError(
            f"Audio mask count {int(audio_mask.shape[1])} does not match token count {audio_token_count}."
        )
    if video_keyframes_mask is not None and int(video_keyframes_mask.shape[1]) != video_token_count:
        raise ValueError(
            f"Video keyframes_mask count {int(video_keyframes_mask.shape[1])} does not match token count {video_token_count}."
        )

    joint_denoise_mask = torch.cat([video_mask, audio_mask], dim=1)
    joint_modality_ids = torch.cat(
        [
            torch.zeros((batch, video_token_count, 1), device=device, dtype=torch.int64),
            torch.ones((batch, audio_token_count, 1), device=device, dtype=torch.int64),
        ],
        dim=1,
    )

    fps = float(v_tokens.get("fps") or v_state.get("fps") or alignment["fps"])
    video_positions = model_video_positions(video_positions, fps)
    duration_seconds = float(a_state["duration_seconds"])
    video_range = (0, video_token_count)
    audio_range = (video_token_count, total_token_count)

    return DFRStage1AVModelInput(
        video_latent=video_latent,
        audio_latent=audio_latent,
        video_denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
        joint_denoise_mask=joint_denoise_mask,
        joint_modality_ids=joint_modality_ids,
        video_positions=video_positions,
        audio_positions=audio_positions,
        video_keyframes_mask=video_keyframes_mask,
        video_token_count=video_token_count,
        audio_token_count=audio_token_count,
        total_token_count=total_token_count,
        video_feature_dim=int(video_latent.shape[2]),
        audio_feature_dim=int(audio_latent.shape[2]),
        batch_size=batch,
        video_token_range=video_range,
        audio_token_range=audio_range,
        frame_rate=official_conditioning_fps(fps),
        duration_seconds=duration_seconds,
        video_base_shape=tuple(int(x) for x in v_state["base_shape"]),
        audio_base_shape=tuple(int(x) for x in a_state["base_shape"]),
        video_scale_factors=tuple(int(x) for x in (v_tokens.get("scale_factors") or v_state.get("scale_factors") or (8, 32, 32))),
        audio_config=dict(a_state.get("audio_config", {})),
    )


def av_model_input_stats(model_input: DFRStage1AVModelInput) -> dict[str, Any]:
    return {
        "version": int(model_input.version),
        "total_token_count": int(model_input.total_token_count),
        "joint_mask_shape": tuple(int(x) for x in model_input.joint_denoise_mask.shape),
        "joint_modality_ids_shape": tuple(int(x) for x in model_input.joint_modality_ids.shape),
        "video_token_shape": tuple(int(x) for x in model_input.video_latent.shape),
        "audio_token_shape": tuple(int(x) for x in model_input.audio_latent.shape),
        "video_positions_shape": tuple(int(x) for x in model_input.video_positions.shape),
        "audio_positions_shape": tuple(int(x) for x in model_input.audio_positions.shape),
        "video_keyframes_mask_shape": None if model_input.video_keyframes_mask is None else tuple(int(x) for x in model_input.video_keyframes_mask.shape),
        "video_token_range": tuple(int(x) for x in model_input.video_token_range),
        "audio_token_range": tuple(int(x) for x in model_input.audio_token_range),
        "frame_rate": float(model_input.frame_rate),
        "duration_seconds": float(model_input.duration_seconds),
        "device": str(model_input.video_latent.device),
        "video_dtype": str(model_input.video_latent.dtype).replace("torch.", ""),
        "audio_dtype": str(model_input.audio_latent.dtype).replace("torch.", ""),
        "video_feature_dim": int(model_input.video_feature_dim),
        "audio_feature_dim": int(model_input.audio_feature_dim),
        "modality_order": model_input.modality_order,
    }


def validate_stage1_av_model_input(
    video_official_state: dict[str, Any],
    audio_official_state: dict[str, Any],
    model_input: DFRStage1AVModelInput,
) -> dict[str, Any]:
    """Exact-value regression check for the joint AV input builder."""
    if not isinstance(model_input, DFRStage1AVModelInput):
        raise ValueError(f"model_input has unexpected type {type(model_input)}.")

    v_state, _ = _require_noised_video_state(video_official_state)
    a_state, _ = _require_noised_audio_state(audio_official_state)
    expected = materialize_stage1_av_model_input(video_official_state, audio_official_state)

    def _max_abs(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
        if a is None and b is None:
            return 0.0
        if (a is None) != (b is None):
            return 1.0
        assert a is not None and b is not None
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"Validation shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}.")
        if a.numel() == 0:
            return 0.0
        return float((a.to(device=b.device, dtype=b.dtype) - b).abs().max().item())

    def _equal_int(a: torch.Tensor, b: torch.Tensor) -> float:
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"Validation shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}.")
        return 0.0 if torch.equal(a.to(device=b.device, dtype=b.dtype), b) else 1.0

    video_err = _max_abs(expected.video_latent, model_input.video_latent)
    audio_err = _max_abs(expected.audio_latent, model_input.audio_latent)
    video_mask_err = _max_abs(expected.video_denoise_mask, model_input.video_denoise_mask)
    audio_mask_err = _max_abs(expected.audio_denoise_mask, model_input.audio_denoise_mask)
    joint_mask_err = _max_abs(expected.joint_denoise_mask, model_input.joint_denoise_mask)
    modality_ids_error = _equal_int(expected.joint_modality_ids, model_input.joint_modality_ids)
    video_pos_err = _max_abs(expected.video_positions, model_input.video_positions)
    audio_pos_err = _max_abs(expected.audio_positions, model_input.audio_positions)
    keyframes_mask_err = _max_abs(expected.video_keyframes_mask, model_input.video_keyframes_mask)

    metadata_ok = (
        int(model_input.version) == AV_MODEL_INPUT_VERSION
        and int(model_input.video_token_count) == int(expected.video_token_count)
        and int(model_input.audio_token_count) == int(expected.audio_token_count)
        and int(model_input.total_token_count) == int(expected.total_token_count)
        and int(model_input.video_feature_dim) == int(expected.video_feature_dim)
        and int(model_input.audio_feature_dim) == int(expected.audio_feature_dim)
        and int(model_input.batch_size) == int(expected.batch_size)
        and tuple(int(x) for x in model_input.video_token_range) == tuple(int(x) for x in expected.video_token_range)
        and tuple(int(x) for x in model_input.audio_token_range) == tuple(int(x) for x in expected.audio_token_range)
        and abs(float(model_input.frame_rate) - float(expected.frame_rate)) <= 1e-12
        and abs(float(model_input.duration_seconds) - float(expected.duration_seconds)) <= 1e-12
        and tuple(int(x) for x in model_input.video_base_shape) == tuple(int(x) for x in v_state["base_shape"])
        and tuple(int(x) for x in model_input.audio_base_shape) == tuple(int(x) for x in a_state["base_shape"])
        and tuple(int(x) for x in model_input.video_scale_factors) == tuple(int(x) for x in expected.video_scale_factors)
        and dict(model_input.audio_config) == dict(a_state.get("audio_config", {}))
        and tuple(model_input.modality_order) == ("video", "audio")
    )
    metadata_error = 0.0 if metadata_ok else 1.0

    # Order-specific regression: the advertised ranges must partition the joint
    # bookkeeping exactly into a video prefix and audio suffix.
    vr0, vr1 = model_input.video_token_range
    ar0, ar1 = model_input.audio_token_range
    if vr0 != 0 or vr1 < vr0 or ar0 != vr1 or ar1 < ar0 or ar1 != int(model_input.total_token_count):
        ordering_error = 1.0
    else:
        joint_mask_video_err = _max_abs(model_input.joint_denoise_mask[:, vr0:vr1], model_input.video_denoise_mask)
        joint_mask_audio_err = _max_abs(model_input.joint_denoise_mask[:, ar0:ar1], model_input.audio_denoise_mask)
        joint_ids_video_ok = torch.all(model_input.joint_modality_ids[:, vr0:vr1] == 0).item()
        joint_ids_audio_ok = torch.all(model_input.joint_modality_ids[:, ar0:ar1] == 1).item()
        ordering_error = max(
            joint_mask_video_err,
            joint_mask_audio_err,
            0.0 if joint_ids_video_ok and joint_ids_audio_ok else 1.0,
        )

    passed = all(
        err == 0.0
        for err in (
            video_err,
            audio_err,
            video_mask_err,
            audio_mask_err,
            joint_mask_err,
            modality_ids_error,
            video_pos_err,
            audio_pos_err,
            keyframes_mask_err,
            metadata_error,
            ordering_error,
        )
    )

    return {
        "passed": passed,
        "video_error": video_err,
        "audio_error": audio_err,
        "video_mask_error": video_mask_err,
        "audio_mask_error": audio_mask_err,
        "joint_mask_error": joint_mask_err,
        "modality_ids_error": modality_ids_error,
        "video_position_error": video_pos_err,
        "audio_position_error": audio_pos_err,
        "keyframes_mask_error": keyframes_mask_err,
        "metadata_error": metadata_error,
        "ordering_error": ordering_error,
        "total_token_count": int(model_input.total_token_count),
        "video_token_shape": tuple(int(x) for x in model_input.video_latent.shape),
        "audio_token_shape": tuple(int(x) for x in model_input.audio_latent.shape),
        "joint_mask_shape": tuple(int(x) for x in model_input.joint_denoise_mask.shape),
        "joint_modality_ids_shape": tuple(int(x) for x in model_input.joint_modality_ids.shape),
        "video_positions_shape": tuple(int(x) for x in model_input.video_positions.shape),
        "audio_positions_shape": tuple(int(x) for x in model_input.audio_positions.shape),
        "video_token_range": tuple(int(x) for x in model_input.video_token_range),
        "audio_token_range": tuple(int(x) for x in model_input.audio_token_range),
        "frame_rate": float(model_input.frame_rate),
        "duration_seconds": float(model_input.duration_seconds),
        "device": str(model_input.video_latent.device),
        "audio_alignment_passed": bool(validate_audio_state_against_video(audio_official_state, video_official_state)["passed"]),
    }
