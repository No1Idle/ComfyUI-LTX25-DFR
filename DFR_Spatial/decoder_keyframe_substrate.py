"""U3.4b1 decoder-keyframe substrate preflight for the Spatial-DFR parity path.

This stage still does **not** run the keyframe-aware decoder. Instead it builds
and validates the structural object that later U3.4b substeps will consume:

- exact post-Stage2 RNG restoration (same U3.1 source as U3.3/U3.4a)
- verified U3.2 final decoder keyframes
- canonical keyframe timing metadata on the latent-border timeline
- inventory of the loaded decoder modules/parameters relevant to the official
  keyframe-aware path (type embeddings, conv_in, conv_in_x_t, upsamplers,
  timestep receivers, output head, generic blocks)

The main purpose is to confirm what is already present in the loaded Comfy VAE
runtime and what will still need to be ported in later U3.4b steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .decoder_bridge import restore_decoder_generator
from .decoder_handoff import DFRStage2DecoderHandoff
from .decoder_keyframes import DFRFinalDecodeKeyframes, validate_final_decode_keyframes
from .dfr_layout import pixel_to_latent_index, validate_layout


DECODER_KEYFRAME_SUBSTRATE_VERSION = 1
EXPECTED_KEYFRAME_CHANNELS = 128


@dataclass(frozen=True)
class DFRDecoderKeyframeSubstrate:
    version: int
    rng_device_type: str
    generator_state_restored: bool
    wrapper_class: str
    first_stage_model_class: str
    decoder_class: str
    temporal_scale: int
    keyframes_present: bool
    keyframe_count: int
    keyframe_channels: int
    expected_keyframe_channels: int
    keyframe_channels_match_expected: bool
    keyframe_spatial_shape: tuple[int, int] | None
    keyframe_pixel_frame_indices: tuple[int, ...]
    keyframe_latent_frame_indices: tuple[int, ...]
    keyframe_valid_mask: tuple[bool, ...]
    conv_in_paths: tuple[str, ...]
    conv_in_weight_shapes: tuple[tuple[int, ...], ...]
    conv_in_x_t_paths: tuple[str, ...]
    conv_in_x_t_weight_shapes: tuple[tuple[int, ...], ...]
    type_emb_paths: tuple[str, ...]
    type_emb_shapes: tuple[tuple[int, ...], ...]
    timestep_receiver_paths: tuple[str, ...]
    upsample_paths: tuple[str, ...]
    block_paths: tuple[str, ...]
    output_head_paths: tuple[str, ...]
    has_conv_in: bool
    has_conv_in_x_t: bool
    has_type_emb: bool
    has_timestep_receiver: bool
    has_upsamplers: bool
    has_output_head: bool
    notes: tuple[str, ...]


def _class_name(obj: Any) -> str:
    if obj is None:
        return "<none>"
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_module(obj: Any) -> bool:
    return hasattr(obj, "named_modules") and hasattr(obj, "named_parameters")


def _dedupe_keep_order(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(x) for x in items))


def _weight_shape_from_module(module: Any) -> tuple[int, ...] | None:
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return tuple(int(x) for x in weight.shape)
    return None


def _collect_named_modules(root: Any):
    if not _is_module(root):
        return []
    return list(root.named_modules())


def _collect_named_parameters(root: Any):
    if not _is_module(root):
        return []
    return list(root.named_parameters())


def _match_any(path: str, *tokens: str) -> bool:
    lower = path.lower()
    return any(token in lower for token in tokens)


def _collect_module_paths(root: Any, predicate) -> tuple[str, ...]:
    paths: list[str] = []
    for path, module in _collect_named_modules(root):
        if not path:
            continue
        if predicate(path, module):
            paths.append(f"vae.first_stage_model.decoder.{path}")
    return _dedupe_keep_order(paths)


def _collect_module_weight_shapes(root: Any, predicate) -> tuple[tuple[str, tuple[int, ...]], ...]:
    items: list[tuple[str, tuple[int, ...]]] = []
    for path, module in _collect_named_modules(root):
        if not path:
            continue
        if predicate(path, module):
            shape = _weight_shape_from_module(module)
            if shape is not None:
                items.append((f"vae.first_stage_model.decoder.{path}", shape))
    return tuple(items)


def _collect_type_emb_details(root: Any) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    paths: list[str] = []
    shapes: list[tuple[int, ...]] = []

    # Parameters first: type_emb may be a bare Parameter rather than a module.
    for path, param in _collect_named_parameters(root):
        if not path or "type_emb" not in path.lower():
            continue
        paths.append(f"vae.first_stage_model.decoder.{path}")
        shapes.append(tuple(int(x) for x in param.shape))

    # Then module names with weight if they weren't already represented.
    existing = set(paths)
    for path, module in _collect_named_modules(root):
        if not path or "type_emb" not in path.lower():
            continue
        full_path = f"vae.first_stage_model.decoder.{path}"
        if full_path in existing:
            continue
        shape = _weight_shape_from_module(module)
        if shape is not None:
            paths.append(full_path)
            shapes.append(shape)

    return _dedupe_keep_order(paths), tuple(shapes)


def build_decoder_keyframe_substrate(
    vae: Any,
    decoder_handoff: DFRStage2DecoderHandoff,
    final_decode_keyframes: DFRFinalDecodeKeyframes,
    dfr_layout: dict[str, Any],
    *,
    collect_diagnostics: bool = True,
) -> tuple[DFRDecoderKeyframeSubstrate, str]:
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(
            f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}."
        )
    if not isinstance(final_decode_keyframes, DFRFinalDecodeKeyframes):
        raise ValueError(
            f"Expected DFRFinalDecodeKeyframes, got {type(final_decode_keyframes).__name__}."
        )

    layout = validate_layout(dfr_layout)
    if collect_diagnostics:
        validation = validate_final_decode_keyframes(decoder_handoff, layout, final_decode_keyframes)
        if not validation["passed"]:
            raise ValueError(
                "final_decode_keyframes failed U3.2 validation before U3.4b1 substrate preparation: "
                f"latents_error={validation['latents_error']}, indices_error={validation['indices_error']}, "
                f"metadata_error={validation['metadata_error']}."
            )

    generator, rng_device_type = restore_decoder_generator(decoder_handoff)
    expected_state = decoder_handoff.rng_state_after_stage2_av.detach().cpu()
    restored_state = generator.get_state().detach().cpu()
    generator_state_restored = bool(torch.equal(restored_state, expected_state))

    first_stage_model = getattr(vae, "first_stage_model", None)
    direct_decoder = getattr(first_stage_model, "decoder", None)
    if first_stage_model is None or direct_decoder is None:
        raise ValueError("Connected VAE does not expose vae.first_stage_model.decoder.")

    temporal_scale = int(layout["temporal_scale"])
    keyframes_present = bool(final_decode_keyframes.present)
    pixel_indices = tuple(int(x) for x in final_decode_keyframes.pixel_frame_indices.detach().cpu().tolist())
    latent_indices = tuple(pixel_to_latent_index(idx, temporal_scale) for idx in pixel_indices)
    valid_mask = tuple(True for _ in pixel_indices)

    keyframe_count = int(final_decode_keyframes.kept_keyframe_count)
    if final_decode_keyframes.latents is None:
        keyframe_channels = 0
        keyframe_spatial_shape = None
    else:
        latents = final_decode_keyframes.latents
        if not torch.is_tensor(latents) or latents.ndim != 5:
            raise ValueError("final_decode_keyframes.latents must be [B,C,K,H,W] when present=True.")
        keyframe_channels = int(latents.shape[1])
        keyframe_spatial_shape = (int(latents.shape[-2]), int(latents.shape[-1]))
        if int(latents.shape[2]) != keyframe_count:
            raise ValueError(
                "final_decode_keyframes.kept_keyframe_count does not match the latent tensor depth: "
                f"metadata={keyframe_count}, tensor={int(latents.shape[2])}."
            )

    keyframe_channels_match_expected = keyframe_channels == EXPECTED_KEYFRAME_CHANNELS or (not keyframes_present and keyframe_channels == 0)

    type_emb_paths, type_emb_shapes = _collect_type_emb_details(direct_decoder)

    conv_in_items = _collect_module_weight_shapes(
        direct_decoder,
        lambda path, module: _match_any(path, "conv_in") and "conv_in_x_t" not in path.lower(),
    )
    conv_in_paths = tuple(path for path, _ in conv_in_items)
    conv_in_weight_shapes = tuple(shape for _, shape in conv_in_items)

    conv_in_x_t_items = _collect_module_weight_shapes(
        direct_decoder,
        lambda path, module: "conv_in_x_t" in path.lower(),
    )
    conv_in_x_t_paths = tuple(path for path, _ in conv_in_x_t_items)
    conv_in_x_t_weight_shapes = tuple(shape for _, shape in conv_in_x_t_items)

    timestep_receiver_paths = _collect_module_paths(
        direct_decoder,
        lambda path, module: _match_any(path, "time_embed", "timestep", "temb", "adaln", "ada_ln", "time_mlp"),
    )
    upsample_paths = _collect_module_paths(
        direct_decoder,
        lambda path, module: _match_any(path, "upsample", "upsamples", "up_blocks", "up_block", ".up.", "upsampler"),
    )
    block_paths = _collect_module_paths(
        direct_decoder,
        lambda path, module: ("block" in path.lower()) or ("block" in type(module).__name__.lower()),
    )
    output_head_paths = _collect_module_paths(
        direct_decoder,
        lambda path, module: _match_any(path, "conv_out", "proj_out", "output", "out_proj", "head"),
    )

    notes: list[str] = []
    if not generator_state_restored:
        notes.append("Failed to reconstruct the exact U3.1 post-Stage2 RNG state.")
    if keyframes_present and keyframe_count != len(pixel_indices):
        notes.append(
            "The number of kept keyframes disagrees with the supplied final pixel-frame indices."
        )
    if keyframes_present and not keyframe_channels_match_expected:
        notes.append(
            f"Expected decoder keyframe latent channels={EXPECTED_KEYFRAME_CHANNELS}, got {keyframe_channels}."
        )
    if not type_emb_paths:
        notes.append("No decoder type_emb receiver is present in the current loaded decoder tree.")
    if not conv_in_paths:
        notes.append("No conv_in module was discovered on the current loaded decoder tree.")
    if not conv_in_x_t_paths:
        notes.append("No conv_in_x_t module was discovered on the current loaded decoder tree.")
    if not upsample_paths:
        notes.append("No explicit upsampler modules were discovered; later U3.4b stages may need broader mapping.")
    if not timestep_receiver_paths:
        notes.append("No explicit timestep/AdaLN receiver modules were discovered.")
    if not output_head_paths:
        notes.append("No explicit output-head module was discovered.")
    if not notes:
        notes.append("Decoder substrate preflight looks structurally complete for the current runtime inventory.")

    substrate = DFRDecoderKeyframeSubstrate(
        version=DECODER_KEYFRAME_SUBSTRATE_VERSION,
        rng_device_type=rng_device_type,
        generator_state_restored=generator_state_restored,
        wrapper_class=_class_name(vae),
        first_stage_model_class=_class_name(first_stage_model),
        decoder_class=_class_name(direct_decoder),
        temporal_scale=temporal_scale,
        keyframes_present=keyframes_present,
        keyframe_count=keyframe_count,
        keyframe_channels=keyframe_channels,
        expected_keyframe_channels=EXPECTED_KEYFRAME_CHANNELS,
        keyframe_channels_match_expected=bool(keyframe_channels_match_expected),
        keyframe_spatial_shape=keyframe_spatial_shape,
        keyframe_pixel_frame_indices=pixel_indices,
        keyframe_latent_frame_indices=latent_indices,
        keyframe_valid_mask=valid_mask,
        conv_in_paths=conv_in_paths,
        conv_in_weight_shapes=conv_in_weight_shapes,
        conv_in_x_t_paths=conv_in_x_t_paths,
        conv_in_x_t_weight_shapes=conv_in_x_t_weight_shapes,
        type_emb_paths=type_emb_paths,
        type_emb_shapes=type_emb_shapes,
        timestep_receiver_paths=timestep_receiver_paths,
        upsample_paths=upsample_paths,
        block_paths=block_paths,
        output_head_paths=output_head_paths,
        has_conv_in=bool(conv_in_paths),
        has_conv_in_x_t=bool(conv_in_x_t_paths),
        has_type_emb=bool(type_emb_paths),
        has_timestep_receiver=bool(timestep_receiver_paths),
        has_upsamplers=bool(upsample_paths),
        has_output_head=bool(output_head_paths),
        notes=tuple(notes),
    )

    report = substrate_report(substrate) if collect_diagnostics else ""
    return substrate, report


def validate_decoder_keyframe_substrate(
    vae: Any,
    decoder_handoff: DFRStage2DecoderHandoff,
    final_decode_keyframes: DFRFinalDecodeKeyframes,
    dfr_layout: dict[str, Any],
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
) -> dict[str, Any]:
    if not isinstance(decoder_keyframe_substrate, DFRDecoderKeyframeSubstrate):
        raise ValueError(
            f"Expected DFRDecoderKeyframeSubstrate, got {type(decoder_keyframe_substrate).__name__}."
        )
    expected, _report = build_decoder_keyframe_substrate(vae, decoder_handoff, final_decode_keyframes, dfr_layout)
    actual = decoder_keyframe_substrate
    metadata_fields = (
        "version",
        "rng_device_type",
        "generator_state_restored",
        "wrapper_class",
        "first_stage_model_class",
        "decoder_class",
        "temporal_scale",
        "keyframes_present",
        "keyframe_count",
        "keyframe_channels",
        "expected_keyframe_channels",
        "keyframe_channels_match_expected",
        "keyframe_spatial_shape",
        "keyframe_pixel_frame_indices",
        "keyframe_latent_frame_indices",
        "keyframe_valid_mask",
        "conv_in_paths",
        "conv_in_weight_shapes",
        "conv_in_x_t_paths",
        "conv_in_x_t_weight_shapes",
        "type_emb_paths",
        "type_emb_shapes",
        "timestep_receiver_paths",
        "upsample_paths",
        "block_paths",
        "output_head_paths",
        "has_conv_in",
        "has_conv_in_x_t",
        "has_type_emb",
        "has_timestep_receiver",
        "has_upsamplers",
        "has_output_head",
        "notes",
    )
    metadata_error = 0.0 if all(getattr(actual, k) == getattr(expected, k) for k in metadata_fields) else 1.0
    passed = metadata_error == 0.0
    return {
        "passed": bool(passed),
        "metadata_error": float(metadata_error),
        "has_type_emb": bool(expected.has_type_emb),
        "has_conv_in": bool(expected.has_conv_in),
        "has_conv_in_x_t": bool(expected.has_conv_in_x_t),
        "channels_match_expected": bool(expected.keyframe_channels_match_expected),
        "keyframe_count": int(expected.keyframe_count),
        "keyframe_latent_indices": expected.keyframe_latent_frame_indices,
        "type_emb_paths": expected.type_emb_paths,
        "conv_in_paths": expected.conv_in_paths,
        "conv_in_x_t_paths": expected.conv_in_x_t_paths,
    }


def substrate_report(substrate: DFRDecoderKeyframeSubstrate) -> str:
    notes_text = " | ".join(substrate.notes)
    return (
        f"PASS_RNG={substrate.generator_state_restored}; stage=U3.4b1_decoder_keyframe_substrate; "
        f"rng_device={substrate.rng_device_type}; wrapper={substrate.wrapper_class}; "
        f"first_stage_model={substrate.first_stage_model_class}; decoder={substrate.decoder_class}; "
        f"temporal_scale={substrate.temporal_scale}; keyframes_present={substrate.keyframes_present}; "
        f"keyframe_count={substrate.keyframe_count}; keyframe_channels={substrate.keyframe_channels}; "
        f"expected_keyframe_channels={substrate.expected_keyframe_channels}; "
        f"channels_match_expected={substrate.keyframe_channels_match_expected}; "
        f"keyframe_spatial_shape={substrate.keyframe_spatial_shape}; "
        f"pixel_indices={substrate.keyframe_pixel_frame_indices}; latent_indices={substrate.keyframe_latent_frame_indices}; "
        f"has_type_emb={substrate.has_type_emb}; has_conv_in={substrate.has_conv_in}; "
        f"has_conv_in_x_t={substrate.has_conv_in_x_t}; has_timestep_receiver={substrate.has_timestep_receiver}; "
        f"has_upsamplers={substrate.has_upsamplers}; has_output_head={substrate.has_output_head};\n"
        f"TYPE_EMB_PATHS: {substrate.type_emb_paths if substrate.type_emb_paths else '<none>'}; shapes={substrate.type_emb_shapes}\n"
        f"CONV_IN_PATHS: {substrate.conv_in_paths if substrate.conv_in_paths else '<none>'}; shapes={substrate.conv_in_weight_shapes}\n"
        f"CONV_IN_X_T_PATHS: {substrate.conv_in_x_t_paths if substrate.conv_in_x_t_paths else '<none>'}; shapes={substrate.conv_in_x_t_weight_shapes}\n"
        f"TIMESTEP_RECEIVERS: {substrate.timestep_receiver_paths if substrate.timestep_receiver_paths else '<none>'}\n"
        f"UPSAMPLERS: {substrate.upsample_paths if substrate.upsample_paths else '<none>'}\n"
        f"BLOCKS: {substrate.block_paths if substrate.block_paths else '<none>'}\n"
        f"OUTPUT_HEADS: {substrate.output_head_paths if substrate.output_head_paths else '<none>'}\n"
        f"NOTES: {notes_text}"
    )
