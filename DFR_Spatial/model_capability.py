"""Generated-keyframe capability preflight for the strict DFR path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_CAPABILITY_KEY = "use_keyframes_abs_pos_embedding"


def _mapping_flag(value: Any) -> bool | None:
    if not isinstance(value, Mapping):
        return None
    if _CAPABILITY_KEY in value:
        return bool(value[_CAPABILITY_KEY])
    for key in ("transformer", "unet_config", "config", "model_config"):
        nested = value.get(key)
        found = _mapping_flag(nested)
        if found is not None:
            return found
    return None


def _real_attr(value: Any, name: str) -> Any:
    """Read only a genuinely stored attribute, never Comfy's dynamic __getattr__ fallback.

    Some Comfy model-config objects deliberately return a false-y placeholder for unknown
    attributes and emit a warning. Using hasattr/getattr on those objects made C12 invent an
    explicit ``use_keyframes_abs_pos_embedding=False`` declaration that was not actually there.
    """
    if value is None:
        return None
    try:
        dct = object.__getattribute__(value, "__dict__")
    except Exception:
        dct = None
    if isinstance(dct, Mapping) and name in dct:
        return dct[name]
    try:
        return object.__getattribute__(value, name)
    except Exception:
        return None


def _explicit_flag(model: Any) -> tuple[bool | None, str]:
    """Find a real checkpoint/config declaration without triggering dynamic config fallbacks."""
    base = _real_attr(model, "model")
    model_config = _real_attr(base, "model_config")

    # Comfy's supported-model config keeps the transformer constructor dictionary here.
    unet_config = _real_attr(model_config, "unet_config")
    found = _mapping_flag(unet_config)
    if found is not None:
        return found, "MODEL.model.model_config.unet_config"

    # Also inspect real stored dictionaries on the relevant wrapper objects. Do not use
    # hasattr/getattr for the capability key: Comfy's config __getattr__ can fabricate False.
    for label, candidate in (
        ("MODEL.model.model_config", model_config),
        ("MODEL.model", base),
        ("MODEL", model),
    ):
        if candidate is None:
            continue
        try:
            dct = object.__getattribute__(candidate, "__dict__")
        except Exception:
            dct = None
        found = _mapping_flag(dct)
        if found is not None:
            return found, label

    diffusion_model = _real_attr(base, "diffusion_model")
    if diffusion_model is not None:
        for label, candidate in (
            ("MODEL.model.diffusion_model.config", _real_attr(diffusion_model, "config")),
            ("MODEL.model.diffusion_model", diffusion_model),
        ):
            if isinstance(candidate, Mapping):
                found = _mapping_flag(candidate)
            else:
                try:
                    found = _mapping_flag(object.__getattribute__(candidate, "__dict__"))
                except Exception:
                    found = None
            if found is not None:
                return found, label

    return None, ""


def _module_evidence(model: Any) -> str | None:
    """Runtime proof for Comfy builds that do not retain the original config mapping."""
    base = _real_attr(model, "model")
    diffusion_model = _real_attr(base, "diffusion_model")
    if diffusion_model is None:
        return None

    # This only enumerates the already-constructed module tree; it does not materialize
    # offloaded/staged parameter storage onto the GPU.
    for iterator_name in ("named_modules", "named_parameters", "named_buffers"):
        iterator = _real_attr(diffusion_model, iterator_name)
        if not callable(iterator):
            continue
        try:
            for name, _value in iterator():
                lowered = str(name).lower()
                if "keyframe" in lowered and ("pos" in lowered or "embed" in lowered):
                    return f"diffusion_model.{iterator_name}:{name}"
        except Exception:
            continue
    return None


def generated_keyframe_capability(model: Any) -> tuple[bool, str]:
    """Return capability plus the evidence used to establish it.

    Prefer positive runtime module evidence over a stored negative flag, because a Comfy
    checkpoint wrapper can retain a generic constructor config while the instantiated LTX
    module itself exposes the generated-keyframe embedding. A real explicit False is used only
    when there is no positive runtime evidence.
    """
    declared, source = _explicit_flag(model)
    if declared is True:
        return True, f"config:{source}=True"

    evidence = _module_evidence(model)
    if evidence is not None:
        return True, f"module:{evidence}"

    if declared is False:
        return False, f"config:{source}=False"

    return False, "capability flag/module evidence not found"


def assert_generated_keyframes_supported(model: Any) -> str:
    supported, evidence = generated_keyframe_capability(model)
    if not supported:
        raise ValueError(
            "DFR generated keyframe slots require a transformer trained with the keyframe absolute-position "
            "embedding (use_keyframes_abs_pos_embedding=True). The connected MODEL does not expose that "
            f"capability: {evidence}. Use the current LTX-2.5 distilled generated-keyframe checkpoint."
        )
    return evidence
