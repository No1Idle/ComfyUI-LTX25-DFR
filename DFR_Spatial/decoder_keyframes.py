
"""U3.2 final decoder-keyframe construction for strict Spatial-DFR parity.

This mirrors upstream ``decode_keyframes_from_slots`` for the current
Spatial x2 / temporal-upscalings=0 parity path.  It does not decode anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_handoff import DFRStage2DecoderHandoff
from .dfr_layout import validate_layout


FINAL_DECODE_KEYFRAMES_VERSION = 1


@dataclass(frozen=True)
class DFRFinalDecodeKeyframes:
    """Comfy-side representation of upstream ``DecodeKeyframes | None``.

    ``present=False`` is the explicit Comfy representation of upstream ``None``.
    When present, ``latents`` is [B,C,K,H,W] and ``pixel_frame_indices`` is a
    CPU ``torch.long`` tensor, matching the helper's construction semantics.
    """

    version: int
    present: bool
    num_frames: int
    source_positions: tuple[int, ...]
    kept_source_indices: tuple[int, ...]
    kept_positions: tuple[int, ...]
    dropped_positions: tuple[int, ...]
    latents: torch.Tensor | None
    pixel_frame_indices: torch.Tensor
    source_keyframe_count: int
    kept_keyframe_count: int


def build_final_decode_keyframes(
    decoder_handoff: DFRStage2DecoderHandoff,
    dfr_layout: dict[str, Any],
) -> tuple[DFRFinalDecodeKeyframes, str, str, str]:
    """Mirror upstream ``decode_keyframes_from_slots`` after final canvas trim.

    Current U3 scope is Spatial x2 with temporal-upscalings=0, so the final
    decoder canvas is exactly ``dfr_layout['requested_frames']`` and the Stage-2
    carried positions are exactly ``dfr_layout['pixel_frame_indices']``.
    """
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(
            f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}."
        )
    layout = validate_layout(dfr_layout)

    latents = decoder_handoff.stage_2_generated_keyframes
    if not torch.is_tensor(latents) or latents.ndim != 5:
        raise ValueError(
            "decoder_handoff.stage_2_generated_keyframes must be [B,C,K,H,W]."
        )

    positions = tuple(int(x) for x in layout["pixel_frame_indices"])
    source_count = int(latents.shape[2])
    if source_count != len(positions):
        raise ValueError(
            "Stage-2 generated-slot count does not match the DFR layout positions: "
            f"slots={source_count}, positions={len(positions)}, positions={positions}."
        )
    if int(decoder_handoff.generated_keyframe_count) != source_count:
        raise ValueError(
            "decoder_handoff generated-keyframe metadata disagrees with its latent tensor: "
            f"metadata={decoder_handoff.generated_keyframe_count}, tensor={source_count}."
        )

    num_frames = int(layout["requested_frames"])
    kept = [(index, position) for index, position in enumerate(positions) if 0 <= position < num_frames]
    kept_source_indices = tuple(index for index, _ in kept)
    kept_positions = tuple(position for _, position in kept)
    dropped_positions = tuple(position for position in positions if not (0 <= position < num_frames))

    if kept:
        if kept_source_indices == tuple(range(source_count)):
            final_latents = latents
        else:
            final_latents = torch.cat(
                [latents[:, :, index : index + 1] for index, _ in kept],
                dim=2,
            )
        pixel_frame_indices = torch.tensor(kept_positions, dtype=torch.long)
        present = True
    else:
        final_latents = None
        pixel_frame_indices = torch.empty((0,), dtype=torch.long)
        present = False

    package = DFRFinalDecodeKeyframes(
        version=FINAL_DECODE_KEYFRAMES_VERSION,
        present=present,
        num_frames=num_frames,
        source_positions=positions,
        kept_source_indices=kept_source_indices,
        kept_positions=kept_positions,
        dropped_positions=dropped_positions,
        latents=final_latents,
        pixel_frame_indices=pixel_frame_indices,
        source_keyframe_count=source_count,
        kept_keyframe_count=len(kept),
    )

    kept_text = ",".join(str(x) for x in kept_positions)
    dropped_text = ",".join(str(x) for x in dropped_positions)
    latent_shape = None if final_latents is None else tuple(int(x) for x in final_latents.shape)
    report = (
        f"PASS=True; stage=U3.2_final_decode_keyframes; num_frames={num_frames}; "
        f"source_positions={positions}; kept_positions={kept_positions}; dropped_positions={dropped_positions}; "
        f"source_keyframes={source_count}; kept_keyframes={len(kept)}; present={present}; "
        f"latents_shape={latent_shape}; indices_dtype={pixel_frame_indices.dtype}; "
        "decoder_consumed=False; output_latent_changed=False."
    )
    return package, kept_text, dropped_text, report


def validate_final_decode_keyframes(
    decoder_handoff: DFRStage2DecoderHandoff,
    dfr_layout: dict[str, Any],
    final_decode_keyframes: DFRFinalDecodeKeyframes,
) -> dict[str, Any]:
    """Rebuild U3.2 independently and compare all exact fields/tensors."""
    expected, _, _, _ = build_final_decode_keyframes(decoder_handoff, dfr_layout)
    if not isinstance(final_decode_keyframes, DFRFinalDecodeKeyframes):
        raise ValueError(
            f"Expected DFRFinalDecodeKeyframes, got {type(final_decode_keyframes).__name__}."
        )

    actual = final_decode_keyframes
    metadata_fields = (
        "version",
        "present",
        "num_frames",
        "source_positions",
        "kept_source_indices",
        "kept_positions",
        "dropped_positions",
        "source_keyframe_count",
        "kept_keyframe_count",
    )
    metadata_error = 0.0 if all(getattr(actual, k) == getattr(expected, k) for k in metadata_fields) else 1.0

    indices_error = 0.0 if (
        torch.is_tensor(actual.pixel_frame_indices)
        and actual.pixel_frame_indices.dtype == torch.long
        and actual.pixel_frame_indices.device.type == expected.pixel_frame_indices.device.type
        and torch.equal(actual.pixel_frame_indices.detach().cpu(), expected.pixel_frame_indices.detach().cpu())
    ) else 1.0

    if expected.latents is None:
        latents_error = 0.0 if actual.latents is None else 1.0
    elif not torch.is_tensor(actual.latents) or tuple(actual.latents.shape) != tuple(expected.latents.shape):
        latents_error = 1.0
    elif torch.equal(actual.latents, expected.latents):
        latents_error = 0.0
    else:
        latents_error = float(
            (actual.latents.to(device=expected.latents.device, dtype=torch.float32)
             - expected.latents.to(dtype=torch.float32)).abs().max().item()
        )

    passed = metadata_error == 0.0 and indices_error == 0.0 and latents_error == 0.0
    return {
        "passed": bool(passed),
        "latents_error": float(latents_error),
        "indices_error": float(indices_error),
        "metadata_error": float(metadata_error),
        "present": bool(expected.present),
        "num_frames": int(expected.num_frames),
        "source_positions": expected.source_positions,
        "kept_positions": expected.kept_positions,
        "dropped_positions": expected.dropped_positions,
        "source_keyframe_count": int(expected.source_keyframe_count),
        "kept_keyframe_count": int(expected.kept_keyframe_count),
        "latents_shape": None if expected.latents is None else tuple(int(x) for x in expected.latents.shape),
    }
