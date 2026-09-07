"""Stage-2D detailing IC-LoRA integration for strict Spatial-DFR parity.

This phase is intentionally model-only. It does not no-op or reassemble Stage-2
conditioning, does not add noise, and does not execute the transformer.

Current upstream DFR builds a separate Stage-2 diffusion stage by adding the
required detailing IC-LoRA on top of the same Stage-1 model/LoRA stack. For the
LTX-2.5 Pixel-Spatial x2 adapter current upstream hardcodes model strength 0.5;
this Comfy node exposes that value as an input for controlled comparisons while
defaulting to strict parity. The reference_downscale_factor metadata is 2.

Inside ComfyUI we deliberately delegate the actual LoRA patching to the native
``LoraLoaderModelOnly`` node, then validate the resulting ModelPatcher and its
metadata against the already-built Stage-2C reference conditioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

from .latent_state import get_official_state


OFFICIAL_DETAILING_LORA_BASENAME = (
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
)
OFFICIAL_DETAILING_STRENGTH = 0.5
OFFICIAL_REFERENCE_DOWNSCALE_FACTOR = 2
DETAILING_CONTRACT_ATTACHMENT = "ltx25_dfr_stage2_detailing_contract"


@dataclass(frozen=True)
class DFRDetailingSpec:
    """Resolved detailing adapter identity + metadata used by both Stage 2C and 2D."""

    lora_name: str
    lora_path: str
    reference_downscale_factor: int


def _resolve_lora_path(lora_name: str) -> str:
    try:
        import folder_paths  # type: ignore
    except Exception as exc:  # pragma: no cover - outside Comfy
        raise ValueError("Could not import ComfyUI folder_paths while resolving the detailing LoRA.") from exc

    getter = getattr(folder_paths, "get_full_path_or_raise", None)
    if callable(getter):
        return str(getter("loras", str(lora_name)))
    getter = getattr(folder_paths, "get_full_path", None)
    if callable(getter):
        resolved = getter("loras", str(lora_name))
        if resolved:
            return str(resolved)
    raise ValueError(f"Could not resolve LoRA path for {lora_name!r} under ComfyUI/models/loras.")


def resolve_stage2_detailing_spec(lora_name: str) -> tuple[DFRDetailingSpec, int, str]:
    """Read the selected IC-LoRA metadata before Stage 2C constructs its reference."""
    if Path(str(lora_name)).name != OFFICIAL_DETAILING_LORA_BASENAME:
        raise ValueError(
            "Strict Stage-2 DFR parity requires the official LTX-2.5 x2 Pixel-Spatial IC-LoRA: "
            f"{OFFICIAL_DETAILING_LORA_BASENAME}. Selected: {lora_name!r}."
        )
    lora_path = _resolve_lora_path(str(lora_name))
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ValueError(f"Could not read safetensors metadata from detailing LoRA {lora_path!r}.") from exc

    factor = _metadata_factor(metadata)
    if factor != OFFICIAL_REFERENCE_DOWNSCALE_FACTOR:
        raise ValueError(
            "The selected detailing LoRA metadata does not advertise the required x2 reference factor. "
            f"Resolved reference_downscale_factor={factor}; expected 2."
        )
    spec = DFRDetailingSpec(str(lora_name), lora_path, factor)
    report = (
        f"PASS=True; stage=2_preflight_detailing_metadata; lora={Path(str(lora_name)).name}; "
        f"reference_downscale_factor={factor}; source=safetensors_metadata."
    )
    return spec, factor, report


def lora_choices() -> list[str]:
    """Return Comfy LoRA choices with the official detailing adapter first when available."""
    try:
        import folder_paths  # type: ignore

        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        names = []

    preferred = [name for name in names if Path(name).name == OFFICIAL_DETAILING_LORA_BASENAME]
    others = [name for name in names if Path(name).name != OFFICIAL_DETAILING_LORA_BASENAME]
    # Keep the required official basename visible even before the file is installed;
    # native Comfy loading will then give the normal missing-file error instead of an
    # invalid combo/default mismatch.
    ordered = preferred or [OFFICIAL_DETAILING_LORA_BASENAME]
    return ordered + sorted(others)


def _extract_node_result(result: Any, node_name: str) -> Any:
    candidate = result

    # Current Comfy V3 nodes may return IO.NodeOutput; legacy nodes often return tuples.
    if hasattr(candidate, "result"):
        candidate = candidate.result
    elif isinstance(candidate, dict) and "result" in candidate:
        candidate = candidate["result"]

    if isinstance(candidate, (tuple, list)):
        if not candidate:
            raise ValueError(f"{node_name} returned an empty result sequence.")
        candidate = candidate[0]
    return candidate


def _resolve_native_lora_loader_class():
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception as exc:  # pragma: no cover - only expected outside Comfy
        raise ValueError(
            "Could not import ComfyUI's global nodes registry. Stage 2D must run inside ComfyUI."
        ) from exc

    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise ValueError("ComfyUI nodes registry has no NODE_CLASS_MAPPINGS dict.")

    node_cls = mappings.get("LoraLoaderModelOnly")
    if node_cls is None:
        # Some Comfy builds expose the class directly even if a mapping has not been materialized yet.
        node_cls = getattr(comfy_nodes, "LoraLoaderModelOnly", None)
    if node_cls is None:
        raise ValueError("Native Comfy node 'LoraLoaderModelOnly' is not registered.")
    return node_cls


def _metadata_factor(metadata: Any) -> int:
    """Match Comfy's GetICLoRAParameters semantics for reference_downscale_factor."""
    if not isinstance(metadata, Mapping):
        return 1
    try:
        raw = next(v for k, v in metadata.items() if str(k).endswith("reference_downscale_factor"))
        return max(1, round(float(raw)))
    except (StopIteration, TypeError, ValueError):
        return 1


def _get_attachment(model: Any, key: str) -> Any:
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        return getter(key)
    attachments = getattr(model, "attachments", None)
    if isinstance(attachments, Mapping):
        return attachments.get(key)
    return None


def _set_attachment(model: Any, key: str, value: Any) -> None:
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter(key, value)
        return
    attachments = getattr(model, "attachments", None)
    if attachments is None:
        attachments = {}
        setattr(model, "attachments", attachments)
    if not isinstance(attachments, dict):
        raise ValueError("MODEL does not expose a writable attachment store.")
    attachments[key] = value


def _patch_count(model: Any) -> int | None:
    patches = getattr(model, "patches", None)
    if isinstance(patches, Mapping):
        return len(patches)
    return None


def _patch_uuid(model: Any) -> str:
    value = getattr(model, "patches_uuid", "")
    return str(value) if value is not None else ""


def _reference_factor_from_stage2_state(stage_2_video_state: dict[str, Any]) -> int:
    state = get_official_state(stage_2_video_state)
    operations = list(state.get("operations", []))
    if not operations or operations[-1].get("type") != "VideoConditionByReferenceLatent":
        raise ValueError(
            "Stage 2D requires the validated Stage-2C state ending in VideoConditionByReferenceLatent."
        )
    return int(operations[-1].get("downscale_factor", -1))


def apply_stage2_detailing_lora(
    model: Any,
    stage_2_video_state: dict[str, Any] | None,
    detailing_spec: DFRDetailingSpec,
    strength_model: float = OFFICIAL_DETAILING_STRENGTH,
) -> tuple[Any, int, str]:
    """Apply the x2 Pixel-Spatial IC-LoRA through native Comfy LoRA loading.

    ``strength_model`` defaults to current upstream DFR parity (0.5) but remains
    user-adjustable so the same workflow can make controlled strength comparisons.
    """
    if not isinstance(detailing_spec, DFRDetailingSpec):
        raise ValueError(f"Stage 2D expected DFRDetailingSpec, got {type(detailing_spec).__name__}.")
    lora_name = detailing_spec.lora_name
    if Path(str(lora_name)).name != OFFICIAL_DETAILING_LORA_BASENAME:
        raise ValueError(
            "Strict Stage-2D parity requires the official LTX-2.5 x2 Pixel-Spatial IC-LoRA: "
            f"{OFFICIAL_DETAILING_LORA_BASENAME}. Selected: {lora_name!r}."
        )

    if stage_2_video_state is None:
        # The lean production path prepares the model independently of video
        # conditioning. Stage 2C validates this same factor when it appends the
        # reference latent; the atomic validation path can still supply a state
        # here for an explicit cross-check.
        stage2_reference_factor = int(detailing_spec.reference_downscale_factor)
    else:
        stage2_reference_factor = _reference_factor_from_stage2_state(stage_2_video_state)
        if stage2_reference_factor != detailing_spec.reference_downscale_factor:
            raise ValueError(
                "Stage-2C reference conditioning factor no longer matches the metadata resolved before conditioning. "
                f"Stage2C={stage2_reference_factor}; metadata={detailing_spec.reference_downscale_factor}."
            )

    node_cls = _resolve_native_lora_loader_class()
    node = node_cls()
    function_name = (
        getattr(node_cls, "FUNCTION", None)
        or getattr(node, "FUNCTION", None)
        or "load_lora_model_only"
    )
    func = getattr(node, function_name, None)
    if not callable(func):
        raise ValueError(
            f"Native node '{node_cls.__name__}' does not expose callable function '{function_name}'."
        )

    strength_model = float(strength_model)
    if not math.isfinite(strength_model):
        raise ValueError(f"strength_model must be finite, got {strength_model!r}.")

    result = func(model, str(lora_name), strength_model)
    detailing_model = _extract_node_result(result, node_cls.__name__)
    if detailing_model is None:
        raise ValueError("Native LoraLoaderModelOnly returned no MODEL.")
    if detailing_model is model:
        raise ValueError(
            "Native LoraLoaderModelOnly returned the original MODEL object. "
            "Stage 2 must receive a cloned model patcher with the detailing LoRA applied."
        )

    lora_metadata = _get_attachment(detailing_model, "lora_metadata")
    metadata_factor = _metadata_factor(lora_metadata)
    if metadata_factor != detailing_spec.reference_downscale_factor:
        raise ValueError(
            "Native LoRA loader metadata disagrees with the pre-resolved detailing metadata. "
            f"native={metadata_factor}; pre_resolved={detailing_spec.reference_downscale_factor}."
        )

    base_uuid = _patch_uuid(model)
    detailing_uuid = _patch_uuid(detailing_model)
    if base_uuid and detailing_uuid and base_uuid == detailing_uuid:
        raise ValueError(
            "Detailing MODEL has the same patches_uuid as the input MODEL; the LoRA patch set did not change."
        )

    base_patch_count = _patch_count(model)
    detailing_patch_count = _patch_count(detailing_model)
    if (
        base_patch_count is not None
        and detailing_patch_count is not None
        and detailing_patch_count <= base_patch_count
    ):
        raise ValueError(
            "Detailing MODEL did not gain LoRA patch keys. "
            f"base_patch_count={base_patch_count}; detailing_patch_count={detailing_patch_count}."
        )

    raw_base_model = getattr(model, "model", None)
    raw_detailing_model = getattr(detailing_model, "model", None)
    if raw_base_model is not None and raw_detailing_model is not None and raw_base_model is not raw_detailing_model:
        raise ValueError(
            "LoraLoaderModelOnly unexpectedly replaced the underlying diffusion checkpoint. "
            "Stage 2D must clone/patch the same base model, not load a different model."
        )

    contract = {
        "phase": "2D_detailing_iclora",
        "lora_name": str(lora_name),
        "lora_basename": Path(str(lora_name)).name,
        "strength_model": strength_model,
        "reference_downscale_factor": metadata_factor,
        "native_node": str(node_cls.__name__),
        "base_patches_uuid": base_uuid,
        "detailing_patches_uuid": detailing_uuid,
        "base_patch_count": base_patch_count,
        "detailing_patch_count": detailing_patch_count,
    }
    _set_attachment(detailing_model, DETAILING_CONTRACT_ATTACHMENT, contract)

    report = (
        f"PASS=True; stage=2D_detailing_iclora; native_node={node_cls.__name__}; "
        f"lora={Path(str(lora_name)).name}; strength_model={strength_model:.9g}; "
        f"reference_downscale_factor={metadata_factor}; stage2_reference_factor={stage2_reference_factor}; "
        f"base_patch_count={base_patch_count}; detailing_patch_count={detailing_patch_count}; "
        f"patch_uuid_changed={bool(not base_uuid or not detailing_uuid or base_uuid != detailing_uuid)}."
    )
    return detailing_model, metadata_factor, report


def prepare_stage2_detailing_model(
    model: Any,
    lora_name: str,
    strength_model: float = OFFICIAL_DETAILING_STRENGTH,
) -> tuple[Any, DFRDetailingSpec]:
    """Resolve the official adapter metadata and apply it as one model-only operation."""
    detailing_spec, _factor, _metadata_report = resolve_stage2_detailing_spec(lora_name)
    detailing_model, _factor, _apply_report = apply_stage2_detailing_lora(
        model=model,
        stage_2_video_state=None,
        detailing_spec=detailing_spec,
        strength_model=strength_model,
    )
    return detailing_model, detailing_spec


def validate_stage2_detailing_lora(
    base_model: Any,
    detailing_model: Any,
    stage_2_video_state: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
) -> dict[str, Any]:
    """Validate the Stage-2 detailing model without executing a transformer call."""
    if not isinstance(detailing_spec, DFRDetailingSpec):
        raise ValueError(f"Expected DFRDetailingSpec, got {type(detailing_spec).__name__}.")
    lora_name = detailing_spec.lora_name
    contract = _get_attachment(detailing_model, DETAILING_CONTRACT_ATTACHMENT)
    native_metadata = _get_attachment(detailing_model, "lora_metadata")
    native_factor = _metadata_factor(native_metadata)
    stage2_factor = _reference_factor_from_stage2_state(stage_2_video_state)

    expected_basename = OFFICIAL_DETAILING_LORA_BASENAME
    actual_basename = Path(str(lora_name)).name

    contract_error = 0.0
    if not isinstance(contract, Mapping):
        contract_error = 1.0
        contract = {}

    lora_name_error = 0.0 if actual_basename == expected_basename else 1.0
    contract_lora_error = 0.0 if contract.get("lora_basename") == expected_basename else 1.0
    strength_error = (
        0.0
        if float(contract.get("strength_model", float("nan"))) == OFFICIAL_DETAILING_STRENGTH
        else 1.0
    )
    metadata_error = 0.0 if native_factor == detailing_spec.reference_downscale_factor else 1.0
    reference_factor_error = (
        0.0
        if stage2_factor == native_factor == detailing_spec.reference_downscale_factor
        else 1.0
    )

    same_object_error = 0.0 if detailing_model is not base_model else 1.0

    raw_base = getattr(base_model, "model", None)
    raw_detail = getattr(detailing_model, "model", None)
    underlying_model_error = (
        0.0
        if raw_base is None or raw_detail is None or raw_base is raw_detail
        else 1.0
    )

    base_uuid = _patch_uuid(base_model)
    detail_uuid = _patch_uuid(detailing_model)
    uuid_error = 0.0 if not base_uuid or not detail_uuid or base_uuid != detail_uuid else 1.0

    base_patch_count = _patch_count(base_model)
    detail_patch_count = _patch_count(detailing_model)
    patch_count_error = 0.0
    if base_patch_count is not None and detail_patch_count is not None:
        patch_count_error = 0.0 if detail_patch_count > base_patch_count else 1.0

    passed = bool(
        contract_error == 0.0
        and lora_name_error == 0.0
        and contract_lora_error == 0.0
        and strength_error == 0.0
        and metadata_error == 0.0
        and reference_factor_error == 0.0
        and same_object_error == 0.0
        and underlying_model_error == 0.0
        and uuid_error == 0.0
        and patch_count_error == 0.0
    )

    return {
        "passed": passed,
        "contract_error": contract_error,
        "lora_name_error": lora_name_error,
        "contract_lora_error": contract_lora_error,
        "strength_error": strength_error,
        "metadata_error": metadata_error,
        "reference_factor_error": reference_factor_error,
        "same_object_error": same_object_error,
        "underlying_model_error": underlying_model_error,
        "uuid_error": uuid_error,
        "patch_count_error": patch_count_error,
        "reference_downscale_factor": native_factor,
        "stage2_reference_factor": stage2_factor,
        "base_patch_count": base_patch_count,
        "detailing_patch_count": detail_patch_count,
        "base_patches_uuid": base_uuid,
        "detailing_patches_uuid": detail_uuid,
        "lora_basename": actual_basename,
        "strength_model": contract.get("strength_model"),
        "native_node": contract.get("native_node", "unknown"),
    }
