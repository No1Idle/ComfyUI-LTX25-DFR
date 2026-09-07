"""U3.4a runtime probe for keyframe-aware final decoder support.

This is the first safe sub-step toward official decoder-keyframe parity.
It does **not** decode. Instead, it verifies three prerequisites:

1. the exact post-Stage2 RNG state from U3.1 can be restored;
2. the U3.2 final decoder-keyframe package is structurally valid;
3. the loaded Comfy/LTX video-decoder runtime exposes any plausible keyframe-
   aware entry point (direct ``keyframes=...`` parameter, explicit
   ``forward_with_keyframes`` helpers, or type-embedding receivers).

The current Comfy bridge already handles continued RNG (U3.3b). U3.4a adds the
introspection needed before attempting a real keyframe-aware decode path.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch

from .decoder_bridge import restore_decoder_generator
from .decoder_handoff import DFRStage2DecoderHandoff
from .decoder_keyframes import DFRFinalDecodeKeyframes


KEYFRAME_RUNTIME_PROBE_VERSION = 1


@dataclass(frozen=True)
class DFRDecoderKeyframeRuntimeProbe:
    version: int
    wrapper_class: str
    first_stage_model_class: str
    decoder_class: str
    generator_state_restored: bool
    rng_device: str
    keyframes_present: bool
    keyframe_count: int
    keyframe_latent_shape: tuple[int, ...] | None
    keyframe_indices_shape: tuple[int, ...]
    decode_video_accepts_keyframes: bool
    decode_accepts_keyframes: bool
    direct_decoder_forward_accepts_keyframes: bool
    direct_decoder_has_forward_with_keyframes: bool
    direct_decoder_has_combined_keyframe_path: bool
    candidate_forward_paths: tuple[str, ...]
    type_embedding_paths: tuple[str, ...]
    signatures: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]


def _class_name(obj: Any) -> str:
    if obj is None:
        return "<none>"
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _signature_text(callable_obj: Any) -> str:
    if callable_obj is None:
        return "<missing>"
    try:
        return str(inspect.signature(callable_obj))
    except (TypeError, ValueError) as exc:
        return f"<signature unavailable: {type(exc).__name__}>"


def _accepts_parameter(callable_obj: Any, name: str) -> bool:
    if callable_obj is None:
        return False
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _named_modules_limited(root: Any, prefix: str, max_depth: int = 5):
    if root is None or not hasattr(root, "named_modules"):
        return
    for path, module in root.named_modules():
        parts = [p for p in path.split(".") if p]
        if len(parts) > max_depth:
            continue
        full_path = prefix if not path else f"{prefix}.{path}"
        yield full_path, module


def _discover_candidate_paths(direct_decoder: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    forward_paths: list[str] = []
    type_emb_paths: list[str] = []
    type_attrs = ("type_emb", "type_embedding", "type_embeddings", "type_token_embedding")

    # Root decoder itself first.
    if callable(getattr(direct_decoder, "forward_with_keyframes", None)):
        forward_paths.append("vae.first_stage_model.decoder.forward_with_keyframes")
    if callable(getattr(direct_decoder, "forward_combined_with_keyframes", None)):
        forward_paths.append("vae.first_stage_model.decoder.forward_combined_with_keyframes")
    for attr in type_attrs:
        if hasattr(direct_decoder, attr):
            type_emb_paths.append(f"vae.first_stage_model.decoder.{attr}")

    for path, module in _named_modules_limited(direct_decoder, "vae.first_stage_model.decoder"):
        if path == "vae.first_stage_model.decoder":
            continue
        if callable(getattr(module, "forward_with_keyframes", None)):
            forward_paths.append(f"{path}.forward_with_keyframes")
        if callable(getattr(module, "forward_combined_with_keyframes", None)):
            forward_paths.append(f"{path}.forward_combined_with_keyframes")
        for attr in type_attrs:
            if hasattr(module, attr):
                type_emb_paths.append(f"{path}.{attr}")

    # Preserve order, drop duplicates.
    dedup_forward = tuple(dict.fromkeys(forward_paths))
    dedup_type = tuple(dict.fromkeys(type_emb_paths))
    return dedup_forward, dedup_type


def probe_decoder_keyframe_runtime(
    vae: Any,
    decoder_handoff: DFRStage2DecoderHandoff,
    final_decode_keyframes: DFRFinalDecodeKeyframes,
) -> tuple[DFRDecoderKeyframeRuntimeProbe, dict[str, Any]]:
    if not isinstance(final_decode_keyframes, DFRFinalDecodeKeyframes):
        raise ValueError(
            f"Expected DFRFinalDecodeKeyframes, got {type(final_decode_keyframes).__name__}."
        )

    if final_decode_keyframes.present:
        if not torch.is_tensor(final_decode_keyframes.latents) or final_decode_keyframes.latents.ndim != 5:
            raise ValueError("final_decode_keyframes.latents must be [B,C,K,H,W] when present=True.")
    if not torch.is_tensor(final_decode_keyframes.pixel_frame_indices):
        raise ValueError("final_decode_keyframes.pixel_frame_indices must be a tensor.")
    if final_decode_keyframes.pixel_frame_indices.dtype != torch.long:
        raise ValueError("final_decode_keyframes.pixel_frame_indices must be torch.long.")

    generator, rng_device = restore_decoder_generator(decoder_handoff)
    expected_state = decoder_handoff.rng_state_after_stage2_av.detach().cpu()
    restored_state = generator.get_state().detach().cpu()
    generator_state_restored = torch.equal(restored_state, expected_state)

    first_stage_model = getattr(vae, "first_stage_model", None)
    direct_decoder = getattr(first_stage_model, "decoder", None)

    objects: tuple[tuple[str, Any], ...] = (
        ("vae.decode_tiled", getattr(vae, "decode_tiled", None)),
        ("vae.decode", getattr(vae, "decode", None)),
        ("first_stage_model.decode_video", getattr(first_stage_model, "decode_video", None)),
        ("first_stage_model.decode_tiled", getattr(first_stage_model, "decode_tiled", None)),
        ("first_stage_model.decode", getattr(first_stage_model, "decode", None)),
        ("first_stage_model.decoder.forward", getattr(direct_decoder, "forward", None)),
        ("first_stage_model.decoder.forward_with_keyframes", getattr(direct_decoder, "forward_with_keyframes", None)),
        ("first_stage_model.decoder.forward_combined_with_keyframes", getattr(direct_decoder, "forward_combined_with_keyframes", None)),
    )
    signatures = tuple((name, _signature_text(obj)) for name, obj in objects)

    candidate_forward_paths, type_embedding_paths = _discover_candidate_paths(direct_decoder)

    decode_video_accepts_keyframes = _accepts_parameter(getattr(first_stage_model, "decode_video", None), "keyframes")
    decode_accepts_keyframes = _accepts_parameter(getattr(first_stage_model, "decode", None), "keyframes")
    direct_decoder_forward_accepts_keyframes = _accepts_parameter(getattr(direct_decoder, "forward", None), "keyframes")
    direct_decoder_has_forward_with_keyframes = callable(getattr(direct_decoder, "forward_with_keyframes", None))
    direct_decoder_has_combined_keyframe_path = callable(getattr(direct_decoder, "forward_combined_with_keyframes", None))

    notes: list[str] = []
    if final_decode_keyframes.present and final_decode_keyframes.kept_keyframe_count == 0:
        notes.append("The keyframe package is present=True but contains zero kept keyframes.")
    if not final_decode_keyframes.present:
        notes.append("No decoder keyframes survive final trim; official decode will behave like U3.3b.")
    if not (decode_video_accepts_keyframes or decode_accepts_keyframes or direct_decoder_forward_accepts_keyframes):
        notes.append("No obvious top-level decode entry point advertises a keyframes parameter.")
    if not (direct_decoder_has_forward_with_keyframes or direct_decoder_has_combined_keyframe_path or candidate_forward_paths):
        notes.append("No explicit forward_with_keyframes-style helper was discovered on the decoder tree.")
    if not type_embedding_paths:
        notes.append("No type-embedding receiver was discovered on the decoder tree within probe depth.")
    if not notes:
        notes.append("Runtime exposes at least one plausible keyframe-aware decoder path for U3.4b.")

    probe = DFRDecoderKeyframeRuntimeProbe(
        version=KEYFRAME_RUNTIME_PROBE_VERSION,
        wrapper_class=_class_name(vae),
        first_stage_model_class=_class_name(first_stage_model),
        decoder_class=_class_name(direct_decoder),
        generator_state_restored=bool(generator_state_restored),
        rng_device=rng_device,
        keyframes_present=bool(final_decode_keyframes.present),
        keyframe_count=int(final_decode_keyframes.kept_keyframe_count),
        keyframe_latent_shape=None if final_decode_keyframes.latents is None else tuple(int(x) for x in final_decode_keyframes.latents.shape),
        keyframe_indices_shape=tuple(int(x) for x in final_decode_keyframes.pixel_frame_indices.shape),
        decode_video_accepts_keyframes=bool(decode_video_accepts_keyframes),
        decode_accepts_keyframes=bool(decode_accepts_keyframes),
        direct_decoder_forward_accepts_keyframes=bool(direct_decoder_forward_accepts_keyframes),
        direct_decoder_has_forward_with_keyframes=bool(direct_decoder_has_forward_with_keyframes),
        direct_decoder_has_combined_keyframe_path=bool(direct_decoder_has_combined_keyframe_path),
        candidate_forward_paths=candidate_forward_paths,
        type_embedding_paths=type_embedding_paths,
        signatures=signatures,
        notes=tuple(notes),
    )

    details = {
        "probe": probe,
        "generator": generator,
        "wrapper": vae,
        "first_stage_model": first_stage_model,
        "direct_decoder": direct_decoder,
    }
    return probe, details


def keyframe_runtime_supported(probe: DFRDecoderKeyframeRuntimeProbe) -> bool:
    return bool(
        probe.decode_video_accepts_keyframes
        or probe.decode_accepts_keyframes
        or probe.direct_decoder_forward_accepts_keyframes
        or probe.direct_decoder_has_forward_with_keyframes
        or probe.direct_decoder_has_combined_keyframe_path
        or len(probe.candidate_forward_paths) > 0
    )


def keyframe_runtime_probe_report(probe: DFRDecoderKeyframeRuntimeProbe) -> str:
    sig_text = "\n".join(f"{name}: {sig}" for name, sig in probe.signatures)
    candidate_text = ", ".join(probe.candidate_forward_paths) if probe.candidate_forward_paths else "<none>"
    type_emb_text = ", ".join(probe.type_embedding_paths) if probe.type_embedding_paths else "<none>"
    notes_text = " | ".join(probe.notes)
    return (
        f"PASS_RNG={probe.generator_state_restored}; stage=U3.4a_decoder_keyframe_runtime_probe; "
        f"rng_device={probe.rng_device}; wrapper={probe.wrapper_class}; "
        f"first_stage_model={probe.first_stage_model_class}; decoder={probe.decoder_class}; "
        f"keyframes_present={probe.keyframes_present}; keyframe_count={probe.keyframe_count}; "
        f"keyframe_latent_shape={probe.keyframe_latent_shape}; keyframe_indices_shape={probe.keyframe_indices_shape}; "
        f"decode_video_accepts_keyframes={probe.decode_video_accepts_keyframes}; "
        f"decode_accepts_keyframes={probe.decode_accepts_keyframes}; "
        f"direct_decoder_forward_accepts_keyframes={probe.direct_decoder_forward_accepts_keyframes}; "
        f"direct_decoder_has_forward_with_keyframes={probe.direct_decoder_has_forward_with_keyframes}; "
        f"direct_decoder_has_combined_keyframe_path={probe.direct_decoder_has_combined_keyframe_path}; "
        f"runtime_support_found={keyframe_runtime_supported(probe)};\n"
        f"KEYFRAME_FORWARD_CANDIDATES: {candidate_text}\n"
        f"TYPE_EMBEDDING_PATHS: {type_emb_text}\n"
        f"SIGNATURES:\n{sig_text}\n"
        f"NOTES: {notes_text}\n"
        "decoder_called=False; keyframes_consumed=False; output_changed=False."
    )
