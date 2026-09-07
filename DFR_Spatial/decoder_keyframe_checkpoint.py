"""U3.4b1b: recover the official decoder keyframe tag directly from the VAE checkpoint.

Current Comfy's LTX DiffVAE runtime does not register ``NADiffusionDecoder.type_emb``,
so the runtime module tree cannot tell us whether the safetensors checkpoint carries
the trained keyframe tag. Upstream LTX explicitly checks the raw checkpoint and only
synthesizes a zero ``type_emb`` for checkpoints that genuinely lack it.

This module mirrors that distinction. It reads only the safetensors header/tensor from
the VAE file selected by the user and returns a small package for later U3.4b stages.
No decode and no model mutation happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .decoder_keyframe_substrate import DFRDecoderKeyframeSubstrate, EXPECTED_KEYFRAME_CHANNELS


DECODER_KEYFRAME_CHECKPOINT_VERSION = 1
_TYPE_EMB_STRIPPED_KEY = "type_emb"
_VALID_PREFIXES = ("vae.decoder.", "decoder.")


@dataclass(frozen=True)
class DFRDecoderKeyframeCheckpointWeights:
    version: int
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_has_type_emb: bool
    source_key: str | None
    type_emb: torch.Tensor
    type_emb_shape: tuple[int, ...]
    type_emb_dtype: str
    type_emb_numel: int
    type_emb_is_zero_fallback: bool
    expected_channels: int
    channels_match_expected: bool


def _strip_decoder_prefix(key: str) -> str | None:
    for prefix in _VALID_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _resolve_vae_checkpoint_path(vae_name: str) -> Path:
    try:
        import folder_paths  # type: ignore
    except Exception as exc:  # pragma: no cover - only available inside Comfy runtime
        raise RuntimeError("Comfy folder_paths is unavailable; run this node inside ComfyUI.") from exc

    full = folder_paths.get_full_path("vae", str(vae_name))
    if full is None:
        raise FileNotFoundError(f"Could not resolve VAE checkpoint {vae_name!r} in Comfy's vae folder.")
    path = Path(full)
    if not path.is_file():
        raise FileNotFoundError(f"Resolved VAE checkpoint does not exist: {path}")
    return path


def _find_type_emb_key(keys: list[str]) -> str | None:
    matches: list[str] = []
    for key in keys:
        stripped = _strip_decoder_prefix(key)
        if stripped == _TYPE_EMB_STRIPPED_KEY:
            matches.append(key)
    if len(matches) > 1:
        raise ValueError(f"Checkpoint contains multiple decoder type_emb tensors: {matches}")
    return matches[0] if matches else None


def extract_decoder_keyframe_checkpoint_weights(
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
    vae_name: str,
) -> tuple[DFRDecoderKeyframeCheckpointWeights, str]:
    if not isinstance(decoder_keyframe_substrate, DFRDecoderKeyframeSubstrate):
        raise ValueError(
            f"Expected DFRDecoderKeyframeSubstrate, got {type(decoder_keyframe_substrate).__name__}."
        )
    if not decoder_keyframe_substrate.keyframe_channels_match_expected:
        raise ValueError(
            "U3.4b1 substrate keyframe latent width does not match the expected LTX decoder width; "
            f"got {decoder_keyframe_substrate.keyframe_channels}, expected {EXPECTED_KEYFRAME_CHANNELS}."
        )

    path = _resolve_vae_checkpoint_path(vae_name)
    try:
        import safetensors
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("safetensors is unavailable in the Comfy runtime.") from exc

    source_key: str | None = None
    tensor: torch.Tensor | None = None
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        source_key = _find_type_emb_key(keys)
        if source_key is not None:
            tensor = handle.get_tensor(source_key).detach().cpu().contiguous()

    checkpoint_has_type_emb = tensor is not None
    if tensor is None:
        # Exact upstream compatibility behavior for checkpoints that truly do not
        # carry the keyframe tag: zero means "no trained tag".
        tensor = torch.zeros((EXPECTED_KEYFRAME_CHANNELS,), dtype=torch.bfloat16, device="cpu")
        zero_fallback = True
    else:
        zero_fallback = False

    if tensor.ndim != 1:
        raise ValueError(
            f"Decoder type_emb must be a 1-D latent-channel vector, got shape {tuple(tensor.shape)} "
            f"from key {source_key!r}."
        )
    channels_match_expected = int(tensor.numel()) == EXPECTED_KEYFRAME_CHANNELS
    if not channels_match_expected:
        raise ValueError(
            f"Decoder type_emb width mismatch: checkpoint tensor has {int(tensor.numel())} values, "
            f"expected {EXPECTED_KEYFRAME_CHANNELS}."
        )

    package = DFRDecoderKeyframeCheckpointWeights(
        version=DECODER_KEYFRAME_CHECKPOINT_VERSION,
        checkpoint_name=str(vae_name),
        checkpoint_path=str(path),
        checkpoint_has_type_emb=bool(checkpoint_has_type_emb),
        source_key=source_key,
        type_emb=tensor,
        type_emb_shape=tuple(int(x) for x in tensor.shape),
        type_emb_dtype=str(tensor.dtype),
        type_emb_numel=int(tensor.numel()),
        type_emb_is_zero_fallback=bool(zero_fallback),
        expected_channels=EXPECTED_KEYFRAME_CHANNELS,
        channels_match_expected=bool(channels_match_expected),
    )
    report = checkpoint_weights_report(package)
    return package, report


def validate_decoder_keyframe_checkpoint_weights(
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
    vae_name: str,
    decoder_keyframe_checkpoint_weights: DFRDecoderKeyframeCheckpointWeights,
) -> dict[str, Any]:
    if not isinstance(decoder_keyframe_checkpoint_weights, DFRDecoderKeyframeCheckpointWeights):
        raise ValueError(
            "Expected DFRDecoderKeyframeCheckpointWeights, got "
            f"{type(decoder_keyframe_checkpoint_weights).__name__}."
        )
    expected, _ = extract_decoder_keyframe_checkpoint_weights(decoder_keyframe_substrate, vae_name)
    actual = decoder_keyframe_checkpoint_weights

    metadata_fields = (
        "version",
        "checkpoint_name",
        "checkpoint_path",
        "checkpoint_has_type_emb",
        "source_key",
        "type_emb_shape",
        "type_emb_dtype",
        "type_emb_numel",
        "type_emb_is_zero_fallback",
        "expected_channels",
        "channels_match_expected",
    )
    metadata_error = 0.0 if all(getattr(actual, k) == getattr(expected, k) for k in metadata_fields) else 1.0

    if not torch.is_tensor(actual.type_emb) or tuple(actual.type_emb.shape) != tuple(expected.type_emb.shape):
        tensor_error = 1.0
    elif torch.equal(actual.type_emb.detach().cpu(), expected.type_emb.detach().cpu()):
        tensor_error = 0.0
    else:
        tensor_error = float(
            (actual.type_emb.detach().cpu().to(torch.float32) - expected.type_emb.detach().cpu().to(torch.float32))
            .abs().max().item()
        )

    passed = metadata_error == 0.0 and tensor_error == 0.0
    return {
        "passed": bool(passed),
        "metadata_error": float(metadata_error),
        "tensor_error": float(tensor_error),
        "checkpoint_has_type_emb": bool(expected.checkpoint_has_type_emb),
        "source_key": expected.source_key,
        "type_emb_shape": expected.type_emb_shape,
        "type_emb_dtype": expected.type_emb_dtype,
        "zero_fallback": bool(expected.type_emb_is_zero_fallback),
        "channels_match_expected": bool(expected.channels_match_expected),
    }


def checkpoint_weights_report(package: DFRDecoderKeyframeCheckpointWeights) -> str:
    return (
        f"PASS=True; stage=U3.4b1b_checkpoint_keyframe_weights; checkpoint={package.checkpoint_name}; "
        f"checkpoint_has_type_emb={package.checkpoint_has_type_emb}; source_key={package.source_key}; "
        f"type_emb_shape={package.type_emb_shape}; type_emb_dtype={package.type_emb_dtype}; "
        f"type_emb_numel={package.type_emb_numel}; zero_fallback={package.type_emb_is_zero_fallback}; "
        f"expected_channels={package.expected_channels}; channels_match_expected={package.channels_match_expected}; "
        "model_mutated=False; decoder_called=False."
    )
