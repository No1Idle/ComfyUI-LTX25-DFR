"""U3.4b3b1 runtime probe for deterministic decoder-stage integration.

This is the first safe split of U3.4b3b. It still does **not** run the full
keyframe-aware deterministic decoder stages on the real Stage-2 output, because
that would require wiring the exact per-block dual-stream/joint-attention path
and would be too expensive to validate blindly.

Instead, U3.4b3b1 proves the next integration facts against the *real loaded
Comfy decoder runtime*:

1. the shared ``conv_in`` accepts both the video latent stream and the prepared
   typed keyframe stream;
2. the top-level deterministic upsamplers can be executed on small synthetic
   tensors and their output geometry matches the validated stride schedule from
   U3.4b3a;
3. the decoder tree exposes the expected deterministic stage groups, together
   with block counts and forward signatures needed for the later full U3.4b3b
   block/joint-attention wiring.

So this is a real runtime probe for the deterministic-stage path, not the final
full execution yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch

from .decoder_det_stage_context import DFRDeterministicDecoderKeyframeContext


DET_STAGE_RUNTIME_PROBE_VERSION = 1


@dataclass(frozen=True)
class DFRDeterministicDecoderStageRuntimeProbe:
    version: int
    probe_mode: str
    decoder_class: str
    conv_in_signature: str
    conv_in_shared_weights_used: bool
    conv_in_output_channels: int
    conv_in_video_shape: tuple[int, ...]
    conv_in_keyframe_shape: tuple[int, ...]
    probe_input_video_shape: tuple[int, ...]
    probe_input_keyframe_shape: tuple[int, ...]
    upsample_count: int
    upsample_strides: tuple[tuple[int, int, int], ...]
    upsample_signatures: tuple[str, ...]
    upsample_video_shapes: tuple[tuple[int, ...], ...]
    upsample_keyframe_shapes: tuple[tuple[int, ...], ...]
    det_stage_group_count: int
    det_stage_block_counts: tuple[int, ...]
    det_stage_group_signatures: tuple[tuple[str, ...], ...]
    conv_in_x_t_signature: str
    shared_adaln_signature: str
    stage_count_matches_context: bool
    upsample_schedule_matches_context: bool
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


def _ensure_module_list(name: str, value: Any) -> list[Any]:
    if value is None:
        raise ValueError(f"Loaded decoder does not expose {name}.")
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__"):
        return list(value)
    raise ValueError(f"Loaded decoder field {name} is not iterable: {type(value).__name__}.")


def _parameter_anchor(module: Any) -> tuple[torch.device, torch.dtype]:
    try:
        param = next(module.parameters())
        return param.device, param.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


def _expected_shape_after_strides(
    start_shape: tuple[int, int, int, int, int],
    strides: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    b, c, t, h, w = map(int, start_shape)
    out: list[tuple[int, ...]] = []
    for stride_t, stride_h, stride_w in strides:
        t = t * int(stride_t) - (1 if int(stride_t) == 2 else 0)
        h = h * int(stride_h)
        w = w * int(stride_w)
        out.append((b, c, t, h, w))
    return tuple(out)


def _shapes_match(actual: tuple[tuple[int, ...], ...], expected: tuple[tuple[int, ...], ...]) -> bool:
    return len(actual) == len(expected) and all(tuple(a) == tuple(e) for a, e in zip(actual, expected))


def _stage_group_signatures(stage_groups: list[Any]) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for group in stage_groups:
        blocks = _ensure_module_list("det_stages[*]", group)
        out.append(tuple(_signature_text(getattr(block, "forward", block)) for block in blocks))
    return tuple(out)


def _resolve_runtime(vae: Any) -> tuple[Any, Any, Any, list[Any], list[Any], Any, Any]:
    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if first_stage_model is None or decoder is None:
        raise ValueError("Connected VAE does not expose vae.first_stage_model.decoder.")
    conv_in = getattr(decoder, "conv_in", None)
    if not callable(conv_in):
        raise ValueError("Loaded decoder does not expose callable decoder.conv_in.")
    upsamples = _ensure_module_list("decoder.upsamples", getattr(decoder, "upsamples", None))
    det_stages = _ensure_module_list("decoder.det_stages", getattr(decoder, "det_stages", None))
    conv_in_x_t = getattr(decoder, "conv_in_x_t", None)
    shared_adaln = getattr(decoder, "shared_adaln", None)
    return first_stage_model, decoder, conv_in, upsamples, det_stages, conv_in_x_t, shared_adaln


def probe_deterministic_decoder_stage_runtime(
    vae: Any,
    deterministic_decoder_keyframe_context: DFRDeterministicDecoderKeyframeContext,
    probe_video_frames: int = 3,
    probe_height: int = 2,
    probe_width: int = 2,
) -> tuple[DFRDeterministicDecoderStageRuntimeProbe, str]:
    if not isinstance(
        deterministic_decoder_keyframe_context, DFRDeterministicDecoderKeyframeContext
    ):
        raise ValueError(
            "Expected DFRDeterministicDecoderKeyframeContext, got "
            f"{type(deterministic_decoder_keyframe_context).__name__}."
        )
    if probe_video_frames < 1 or probe_height < 1 or probe_width < 1:
        raise ValueError("Probe geometry must be strictly positive.")

    _, decoder, conv_in, upsamples, det_stages, conv_in_x_t, shared_adaln = _resolve_runtime(vae)
    device, dtype = _parameter_anchor(decoder)

    ctx = deterministic_decoder_keyframe_context
    keyframe_planes = max(1, min(int(ctx.keyframe_count), 2))

    video = torch.zeros(
        (1, int(ctx.keyframe_channels), int(probe_video_frames), int(probe_height), int(probe_width)),
        device=device,
        dtype=dtype,
    )
    keyframes = torch.zeros(
        (1, int(ctx.keyframe_channels), keyframe_planes, int(probe_height), int(probe_width)),
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        video_conv = conv_in(video)
        keyframe_conv = conv_in(keyframes)
        video_shapes: list[tuple[int, ...]] = []
        keyframe_shapes: list[tuple[int, ...]] = []
        running_video = video_conv
        running_keyframes = keyframe_conv
        for upsample in upsamples:
            running_video = upsample(running_video)
            running_keyframes = upsample(running_keyframes)
            video_shapes.append(tuple(int(x) for x in running_video.shape))
            keyframe_shapes.append(tuple(int(x) for x in running_keyframes.shape))

    strides = tuple(tuple(int(x) for x in stage.upsample_stride_from_previous) for stage in ctx.stage_descriptors[1:])
    expected_video_shapes = _expected_shape_after_strides(tuple(int(x) for x in video_conv.shape), strides)
    expected_keyframe_shapes = _expected_shape_after_strides(tuple(int(x) for x in keyframe_conv.shape), strides)

    det_stage_groups = [
        _ensure_module_list(f"decoder.det_stages[{idx}]", group)
        for idx, group in enumerate(det_stages)
    ]
    det_stage_block_counts = tuple(len(group) for group in det_stage_groups)
    det_stage_group_signatures = _stage_group_signatures(det_stages)

    notes: list[str] = []
    if len(upsamples) != int(ctx.deterministic_stage_count):
        notes.append(
            "Top-level upsample count disagrees with U3.4b3a deterministic_stage_count."
        )
    if len(det_stages) != int(ctx.deterministic_stage_count):
        notes.append(
            "det_stages group count disagrees with U3.4b3a deterministic_stage_count."
        )
    if not _shapes_match(tuple(video_shapes), expected_video_shapes):
        notes.append(
            f"Video upsample geometry mismatch: actual={tuple(video_shapes)} expected={expected_video_shapes}."
        )
    if not _shapes_match(tuple(keyframe_shapes), expected_keyframe_shapes):
        notes.append(
            f"Keyframe upsample geometry mismatch: actual={tuple(keyframe_shapes)} expected={expected_keyframe_shapes}."
        )
    if not notes:
        notes.append(
            "Shared conv_in and all deterministic upsamplers executed successfully on synthetic video/keyframe tensors; det_stage block groups were structurally discovered for later full U3.4b3b wiring."
        )
        notes.append(
            "This split intentionally does not execute det_stage blocks or Stage-5 diffusion blocks yet."
        )

    probe = DFRDeterministicDecoderStageRuntimeProbe(
        version=DET_STAGE_RUNTIME_PROBE_VERSION,
        probe_mode="runtime_small_tensor_probe_pre_block_integration",
        decoder_class=_class_name(decoder),
        conv_in_signature=_signature_text(getattr(conv_in, "forward", conv_in)),
        conv_in_shared_weights_used=True,
        conv_in_output_channels=int(video_conv.shape[1]),
        conv_in_video_shape=tuple(int(x) for x in video_conv.shape),
        conv_in_keyframe_shape=tuple(int(x) for x in keyframe_conv.shape),
        probe_input_video_shape=tuple(int(x) for x in video.shape),
        probe_input_keyframe_shape=tuple(int(x) for x in keyframes.shape),
        upsample_count=len(upsamples),
        upsample_strides=tuple(
            tuple(int(x) for x in getattr(upsample, "stride", ())) for upsample in upsamples
        ),
        upsample_signatures=tuple(_signature_text(getattr(upsample, "forward", upsample)) for upsample in upsamples),
        upsample_video_shapes=tuple(video_shapes),
        upsample_keyframe_shapes=tuple(keyframe_shapes),
        det_stage_group_count=len(det_stages),
        det_stage_block_counts=det_stage_block_counts,
        det_stage_group_signatures=det_stage_group_signatures,
        conv_in_x_t_signature=_signature_text(getattr(conv_in_x_t, "forward", conv_in_x_t)),
        shared_adaln_signature=_signature_text(getattr(shared_adaln, "forward", shared_adaln)),
        stage_count_matches_context=(len(det_stages) == int(ctx.deterministic_stage_count) and len(upsamples) == int(ctx.deterministic_stage_count)),
        upsample_schedule_matches_context=(tuple(tuple(int(x) for x in getattr(upsample, "stride", ())) for upsample in upsamples) == tuple(ctx.upsample_strides) and _shapes_match(tuple(video_shapes), expected_video_shapes) and _shapes_match(tuple(keyframe_shapes), expected_keyframe_shapes)),
        notes=tuple(notes),
    )
    return probe, deterministic_stage_runtime_probe_report(probe)


def validate_deterministic_decoder_stage_runtime(
    vae: Any,
    deterministic_decoder_keyframe_context: DFRDeterministicDecoderKeyframeContext,
    runtime_probe: DFRDeterministicDecoderStageRuntimeProbe,
    probe_video_frames: int = 3,
    probe_height: int = 2,
    probe_width: int = 2,
) -> dict[str, Any]:
    if not isinstance(runtime_probe, DFRDeterministicDecoderStageRuntimeProbe):
        raise ValueError(
            "Expected DFRDeterministicDecoderStageRuntimeProbe, got "
            f"{type(runtime_probe).__name__}."
        )
    expected, _ = probe_deterministic_decoder_stage_runtime(
        vae,
        deterministic_decoder_keyframe_context,
        probe_video_frames=probe_video_frames,
        probe_height=probe_height,
        probe_width=probe_width,
    )

    metadata_fields = (
        "version",
        "probe_mode",
        "decoder_class",
        "conv_in_signature",
        "conv_in_shared_weights_used",
        "conv_in_output_channels",
        "conv_in_video_shape",
        "conv_in_keyframe_shape",
        "probe_input_video_shape",
        "probe_input_keyframe_shape",
        "upsample_count",
        "upsample_strides",
        "upsample_signatures",
        "upsample_video_shapes",
        "upsample_keyframe_shapes",
        "det_stage_group_count",
        "det_stage_block_counts",
        "det_stage_group_signatures",
        "conv_in_x_t_signature",
        "shared_adaln_signature",
        "stage_count_matches_context",
        "upsample_schedule_matches_context",
        "notes",
    )
    metadata_error = 0.0 if all(getattr(runtime_probe, k) == getattr(expected, k) for k in metadata_fields) else 1.0
    passed = metadata_error == 0.0 and expected.stage_count_matches_context and expected.upsample_schedule_matches_context
    return {
        "passed": bool(passed),
        "metadata_error": float(metadata_error),
        "stage_count_matches_context": bool(expected.stage_count_matches_context),
        "upsample_schedule_matches_context": bool(expected.upsample_schedule_matches_context),
        "upsample_count": int(expected.upsample_count),
        "upsample_strides": expected.upsample_strides,
        "det_stage_group_count": int(expected.det_stage_group_count),
        "det_stage_block_counts": expected.det_stage_block_counts,
        "conv_in_video_shape": expected.conv_in_video_shape,
        "conv_in_keyframe_shape": expected.conv_in_keyframe_shape,
        "upsample_video_shapes": expected.upsample_video_shapes,
        "upsample_keyframe_shapes": expected.upsample_keyframe_shapes,
    }


def deterministic_stage_runtime_probe_report(
    probe: DFRDeterministicDecoderStageRuntimeProbe,
) -> str:
    return (
        f"PASS={probe.stage_count_matches_context and probe.upsample_schedule_matches_context}; "
        f"stage=U3.4b3b1_det_stage_runtime_probe; mode={probe.probe_mode}; "
        f"decoder={probe.decoder_class}; conv_in_signature={probe.conv_in_signature}; "
        f"conv_in_shared_weights_used={probe.conv_in_shared_weights_used}; "
        f"conv_in_output_channels={probe.conv_in_output_channels}; "
        f"probe_input_video_shape={probe.probe_input_video_shape}; "
        f"probe_input_keyframe_shape={probe.probe_input_keyframe_shape}; "
        f"conv_in_video_shape={probe.conv_in_video_shape}; conv_in_keyframe_shape={probe.conv_in_keyframe_shape}; "
        f"upsample_count={probe.upsample_count}; upsample_strides={probe.upsample_strides}; "
        f"det_stage_group_count={probe.det_stage_group_count}; det_stage_block_counts={probe.det_stage_block_counts}; "
        f"stage_count_matches_context={probe.stage_count_matches_context}; "
        f"upsample_schedule_matches_context={probe.upsample_schedule_matches_context}.\n"
        f"UPSAMPLE_VIDEO_SHAPES: {probe.upsample_video_shapes}\n"
        f"UPSAMPLE_KEYFRAME_SHAPES: {probe.upsample_keyframe_shapes}\n"
        f"UPSAMPLE_SIGNATURES: {probe.upsample_signatures}\n"
        f"DET_STAGE_GROUP_SIGNATURES: {probe.det_stage_group_signatures}\n"
        f"CONV_IN_X_T_SIGNATURE: {probe.conv_in_x_t_signature}\n"
        f"SHARED_ADALN_SIGNATURE: {probe.shared_adaln_signature}\n"
        f"NOTES: {' | '.join(probe.notes)}"
    )
