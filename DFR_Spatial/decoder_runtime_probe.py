"""U3.3a runtime probe for the loaded Comfy LTX video decoder.

This deliberately performs no decode.  The official DFR source defines the
required final call (latent + continued generator, then keyframes in U3.4), but
Comfy's VAE wrapper is a separate runtime interface.  U3.3a discovers the
concrete loaded wrapper/model signatures and proves that the U3.1 generator
state can be reconstructed exactly before U3.3b calls any decoder path.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch

from .decoder_handoff import DFRStage2DecoderHandoff


RUNTIME_PROBE_VERSION = 1


@dataclass(frozen=True)
class DFRDecoderRuntimeProbe:
    version: int
    wrapper_class: str
    first_stage_model_class: str
    generator_state_restored: bool
    rng_device: str
    candidate_path: str | None
    candidate_accepts_generator: bool
    candidate_accepts_keyframes: bool
    signatures: tuple[tuple[str, str], ...]


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


def _restore_generator(handoff: DFRStage2DecoderHandoff) -> tuple[torch.Generator, str]:
    if not isinstance(handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(handoff).__name__}.")
    state = handoff.rng_state_after_stage2_av
    if not torch.is_tensor(state):
        raise ValueError("decoder_handoff.rng_state_after_stage2_av is not a tensor.")

    device_type = str(handoff.rng_device_type).strip().lower()
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(
            "U3.3a currently supports the single-device DFR path on CPU/CUDA; "
            f"handoff reports rng_device_type={handoff.rng_device_type!r}."
        )
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The handoff requires a CUDA generator, but CUDA is unavailable in this runtime.")

    generator = torch.Generator(device=device_type)
    generator.set_state(state.detach().cpu())
    if not torch.equal(generator.get_state().detach().cpu(), state.detach().cpu()):
        raise RuntimeError("Reconstructed decoder generator state does not exactly match the U3.1 handoff state.")
    return generator, device_type


def probe_decoder_runtime(vae: Any, decoder_handoff: DFRStage2DecoderHandoff) -> tuple[DFRDecoderRuntimeProbe, dict[str, Any]]:
    """Inspect concrete Comfy VAE/first-stage decoder signatures without decoding."""
    generator, rng_device = _restore_generator(decoder_handoff)
    first_stage_model = getattr(vae, "first_stage_model", None)

    objects: tuple[tuple[str, Any], ...] = (
        ("vae.decode_tiled", getattr(vae, "decode_tiled", None)),
        ("vae.decode", getattr(vae, "decode", None)),
        ("vae._decode_tiled_owned", getattr(vae, "_decode_tiled_owned", None)),
        ("vae._owned_tiled_args", getattr(vae, "_owned_tiled_args", None)),
        ("first_stage_model.decode_video", getattr(first_stage_model, "decode_video", None)),
        ("first_stage_model.decode_tiled", getattr(first_stage_model, "decode_tiled", None)),
        ("first_stage_model.decode", getattr(first_stage_model, "decode", None)),
    )
    signatures = tuple((name, _signature_text(obj)) for name, obj in objects)

    # Prefer a model-native decode_video path because that is the official decoder API.
    # If absent, report the next concrete callable that can accept an explicit generator;
    # U3.3b will still validate wrapper scaling/output semantics before using it.
    priority = (
        "first_stage_model.decode_video",
        "first_stage_model.decode_tiled",
        "first_stage_model.decode",
        "vae._decode_tiled_owned",
        "vae.decode_tiled",
        "vae.decode",
    )
    by_name = dict(objects)
    candidate_path = None
    candidate_accepts_generator = False
    candidate_accepts_keyframes = False
    for name in priority:
        obj = by_name.get(name)
        if callable(obj) and _accepts_parameter(obj, "generator"):
            candidate_path = name
            candidate_accepts_generator = True
            candidate_accepts_keyframes = _accepts_parameter(obj, "keyframes")
            break

    result = DFRDecoderRuntimeProbe(
        version=RUNTIME_PROBE_VERSION,
        wrapper_class=_class_name(vae),
        first_stage_model_class=_class_name(first_stage_model),
        generator_state_restored=True,
        rng_device=rng_device,
        candidate_path=candidate_path,
        candidate_accepts_generator=candidate_accepts_generator,
        candidate_accepts_keyframes=candidate_accepts_keyframes,
        signatures=signatures,
    )

    details = {
        "probe": result,
        "generator": generator,  # runtime-only; never serialized or consumed in U3.3a
        "wrapper": vae,
        "first_stage_model": first_stage_model,
    }
    return result, details


def runtime_probe_report(probe: DFRDecoderRuntimeProbe) -> str:
    sig_text = "\n".join(f"{name}: {sig}" for name, sig in probe.signatures)
    return (
        f"PASS_RNG={probe.generator_state_restored}; stage=U3.3a_decoder_runtime_probe; "
        f"rng_device={probe.rng_device}; wrapper={probe.wrapper_class}; "
        f"first_stage_model={probe.first_stage_model_class}; "
        f"generator_path={probe.candidate_path or '<none>'}; "
        f"generator_supported={probe.candidate_accepts_generator}; "
        f"keyframes_supported_on_same_path={probe.candidate_accepts_keyframes};\n"
        f"SIGNATURES:\n{sig_text}\n"
        "decoder_called=False; keyframes_consumed=False; output_changed=False."
    )
