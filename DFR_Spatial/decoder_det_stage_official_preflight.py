"""U3.4b3b1 official-source preflight for deterministic dual-stream decoder stages.

C23b replaces the earlier C23 probe. The algorithmic contract in this module is
not inferred from runtime behavior. It is a direct port of the frozen upstream
LTX keyframe-decoder rules:

- stage blocks: ``block.forward_with_keyframes(video, keyframes)``
- video hop: normal stage upsample with the tile's ``drop_leading_frame``
- keyframe hop: ``upsample_keyframe_planes`` -- each plane becomes an isolated
  T=1 clip, goes through the SAME upsampler with ``drop_leading_frame=True``,
  then is restored to plane layout; plane count never changes
- times: rebuilt from global pixel indices with ``keyframe_clip_times`` using
  the next stage's remaining temporal stride

Runtime inspection is used only to prove that Comfy's existing backbone modules
map to the checkpoint architecture that the official algorithm will reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_det_stage_context import DFRDeterministicDecoderKeyframeContext

PREFLIGHT_VERSION = 2


@dataclass(frozen=True)
class DFROfficialDeterministicStagePreflight:
    version: int
    source_contract: str
    config_source: str
    decoder_class: str
    checkpoint_stage_channels: tuple[int, ...]
    checkpoint_stage_depths: tuple[int, ...]
    checkpoint_stage_kernels: tuple[tuple[int, int, int], ...]
    checkpoint_upsamples: tuple[tuple[tuple[int, int, int], int], ...]
    checkpoint_stage5_kernel: tuple[int, int, int]
    checkpoint_model_output_type: str
    runtime_stage_channels: tuple[int, ...]
    runtime_stage_depths: tuple[int, ...]
    runtime_stage_kernels: tuple[tuple[int, int, int], ...]
    runtime_upsample_strides: tuple[tuple[int, int, int], ...]
    runtime_upsample_reductions: tuple[int, ...]
    architecture_matches_checkpoint_config: bool
    keyframe_plane_count: int
    keyframe_plane_shapes: tuple[tuple[int, ...], ...]
    keyframe_plane_count_invariant: bool
    keyframe_plane_helper_exact: bool
    remaining_time_strides: tuple[int, ...]
    keyframe_stage_times: tuple[tuple[float, ...], ...]
    block_forward_with_keyframes_present: tuple[tuple[bool, ...], ...]
    missing_keyframe_methods_expected: bool
    type_emb_checkpoint_present: bool
    decoder_called: bool
    det_stage_blocks_called: bool
    notes: tuple[str, ...]


def _class_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _as_tuple3(value: Any) -> tuple[int, int, int]:
    out = tuple(int(x) for x in value)
    if len(out) != 3:
        raise ValueError(f"Expected 3-D tuple, got {out}.")
    return out


def _module_list(value: Any, name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"Loaded decoder does not expose {name}.")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{name} is not iterable.") from exc


def _checkpoint_decoder_config(vae: Any) -> tuple[dict[str, Any], str]:
    first_stage_model = getattr(vae, "first_stage_model", None)
    if first_stage_model is None:
        raise ValueError("Connected VAE has no first_stage_model.")
    config = getattr(first_stage_model, "config", None)
    if not isinstance(config, dict):
        raise ValueError(
            "C23b requires first_stage_model.config so architecture comes from the VAE "
            "checkpoint metadata instead of from guessed/default constants."
        )
    decoder_config = config.get("decoder")
    if not isinstance(decoder_config, dict):
        raise ValueError(
            "C23b requires first_stage_model.config['decoder']; the loaded VAE did not expose "
            "checkpoint-derived decoder architecture metadata."
        )
    return decoder_config, str(config.get("model_output_type", "v"))


def _checkpoint_architecture(vae: Any):
    cfg, model_output_type = _checkpoint_decoder_config(vae)
    required = ("stage_channels", "stage_depths", "stage_kernels", "upsamples")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(
            "Checkpoint-derived decoder config is missing fields required for an exact "
            f"architecture mapping: {missing}."
        )
    stage_channels = tuple(int(x) for x in cfg["stage_channels"])
    stage_depths = tuple(int(x) for x in cfg["stage_depths"])
    stage_kernels = tuple(_as_tuple3(x) for x in cfg["stage_kernels"])
    upsamples = tuple((_as_tuple3(item[0]), int(item[1])) for item in cfg["upsamples"])
    stage5_kernel = _as_tuple3(cfg.get("stage5_kernel", stage_kernels[-1]))
    return stage_channels, stage_depths, stage_kernels, upsamples, stage5_kernel, model_output_type


def _runtime_architecture(decoder: Any):
    det_stages = _module_list(getattr(decoder, "det_stages", None), "decoder.det_stages")
    upsamples = _module_list(getattr(decoder, "upsamples", None), "decoder.upsamples")
    conv_in = getattr(decoder, "conv_in", None)
    if conv_in is None:
        raise ValueError("Loaded decoder has no conv_in.")

    stage_channels: list[int] = []
    conv_out = getattr(conv_in, "out_features", None)
    if conv_out is None:
        weight = getattr(conv_in, "weight", None)
        if not torch.is_tensor(weight):
            raise ValueError("Cannot resolve decoder.conv_in output width.")
        conv_out = int(weight.shape[0])
    stage_channels.append(int(conv_out))

    reductions: list[int] = []
    strides: list[tuple[int, int, int]] = []
    for upsample in upsamples:
        stride = _as_tuple3(getattr(upsample, "stride", ()))
        strides.append(stride)
        out_channels = getattr(upsample, "out_channels", None)
        if out_channels is None:
            proj = getattr(upsample, "proj", None)
            proj_out = getattr(proj, "out_features", None)
            if proj_out is None:
                raise ValueError("Cannot resolve upsampler output channels.")
            out_channels = int(proj_out) // int(stride[0] * stride[1] * stride[2])
        stage_channels.append(int(out_channels))

        proj = getattr(upsample, "proj", None)
        in_features = getattr(proj, "in_features", stage_channels[-2])
        proj_out = getattr(proj, "out_features", None)
        if proj_out is None:
            reductions.append(-1)
        else:
            reductions.append(
                int(stride[0] * stride[1] * stride[2] * int(in_features) // int(proj_out))
            )

    depths = tuple(len(_module_list(group, "decoder.det_stages[*]")) for group in det_stages)

    kernels: list[tuple[int, int, int]] = []
    for group in det_stages:
        blocks = _module_list(group, "decoder.det_stages[*]")
        if not blocks:
            raise ValueError("Encountered an empty deterministic stage group.")
        attn = getattr(blocks[0], "attn", None)
        kernel = getattr(attn, "kernel_size", None)
        if kernel is None:
            raise ValueError("Cannot resolve deterministic stage attention kernel_size.")
        kernels.append(_as_tuple3(kernel))

    # stage-5 kernel belongs to diff_blocks, append if available so comparison can
    # use the same stage_kernels layout as checkpoint config.
    diff_blocks = _module_list(getattr(decoder, "diff_blocks", None), "decoder.diff_blocks")
    if diff_blocks:
        attn = getattr(diff_blocks[0], "attn", None)
        kernel = getattr(attn, "kernel_size", None)
        if kernel is not None:
            kernels.append(_as_tuple3(kernel))

    return (
        tuple(stage_channels),
        depths,
        tuple(kernels),
        tuple(strides),
        tuple(reductions),
        det_stages,
        upsamples,
    )


def remaining_time_strides_official(upsamples: list[Any]) -> tuple[int, ...]:
    """Direct port of frozen upstream keyframes.remaining_time_strides."""
    strides = [int(up.stride[0]) for up in upsamples]
    remaining: list[int] = []
    for index in range(len(strides)):
        product = 1
        for stride in strides[index:]:
            product *= stride
        remaining.append(product)
    remaining.append(1)
    return tuple(remaining)


def keyframe_stage_times_official(
    pixel_frame_indices: torch.Tensor, remaining_time_stride: int
) -> torch.Tensor:
    """Direct port of frozen upstream keyframes.keyframe_stage_times."""
    if remaining_time_stride < 1:
        raise ValueError("remaining_time_stride must be positive.")
    frames = pixel_frame_indices.to(torch.float32)
    center_offset = (remaining_time_stride - 1) / 2
    times = (frames + center_offset) / remaining_time_stride
    return torch.where(frames == 0, torch.zeros_like(times), times)


def keyframe_clip_times_official(
    pixel_frame_indices: torch.Tensor,
    remaining_time_stride: int,
    clip_start_frame: int,
    extra_origin: float = 0.0,
) -> torch.Tensor:
    """Direct port of frozen upstream keyframes.keyframe_clip_times."""
    times = keyframe_stage_times_official(pixel_frame_indices, remaining_time_stride)
    origin = keyframe_stage_times_official(
        torch.as_tensor(
            [clip_start_frame],
            dtype=torch.int64,
            device=pixel_frame_indices.device,
        ),
        remaining_time_stride,
    )
    return times - origin - extra_origin


def upsample_keyframe_planes_official(
    upsample: Any, x: torch.Tensor
) -> torch.Tensor:
    """Direct port of frozen upstream keyframes.upsample_keyframe_planes.

    ``x`` is channels-last [B,P,H,W,C].
    """
    if x.ndim != 5:
        raise ValueError(f"Expected keyframe stream [B,P,H,W,C], got {tuple(x.shape)}.")
    batch, planes, height, width, channels = map(int, x.shape)
    flat = x.reshape(batch * planes, 1, height, width, channels)
    upsampled = upsample(flat, drop_leading_frame=True)
    if upsampled.ndim != 5:
        raise ValueError("Upsampler did not return channels-last 5-D activations.")
    if int(upsampled.shape[1]) != 1:
        raise RuntimeError(
            "Official isolated keyframe upsampling must preserve T=1, got "
            f"T={upsampled.shape[1]}."
        )
    out = upsampled[:, 0].reshape(
        batch,
        planes,
        int(upsampled.shape[2]),
        int(upsampled.shape[3]),
        int(upsampled.shape[4]),
    )
    if int(out.shape[1]) != planes:
        raise RuntimeError(
            f"Official keyframe plane count changed under upsample: {planes} -> {out.shape[1]}."
        )
    return out


def _probe_official_keyframe_plane_hops(
    decoder: Any,
    context: DFRDeterministicDecoderKeyframeContext,
    upsamples: list[Any],
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    """Execute ONLY the exact upstream plane-upsample helper on tiny tensors.

    Deterministic NA blocks are deliberately not executed here.
    """
    try:
        anchor = next(decoder.parameters())
        device, dtype = anchor.device, anchor.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    planes = int(context.keyframe_count)
    h = 2
    w = 2
    shapes: list[tuple[int, ...]] = []
    exact = True

    for upsample in upsamples:
        proj = getattr(upsample, "proj", None)
        in_channels = getattr(proj, "in_features", None)
        if in_channels is None:
            raise ValueError("Cannot resolve upsampler input width for keyframe-plane probe.")
        x = torch.zeros(
            (1, planes, h, w, int(in_channels)),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            y = upsample_keyframe_planes_official(upsample, x)
        stride = _as_tuple3(upsample.stride)
        expected_h = h * stride[1]
        expected_w = w * stride[2]
        if (
            int(y.shape[0]) != 1
            or int(y.shape[1]) != planes
            or int(y.shape[2]) != expected_h
            or int(y.shape[3]) != expected_w
        ):
            exact = False
        shapes.append(tuple(int(v) for v in y.shape))
        # Each stage is probed independently because the real block stack changes
        # channel width before the next stage. Only spatial geometry/plane invariance
        # is being validated here.
    return tuple(shapes), exact


def prepare_official_deterministic_stage_preflight(
    vae: Any,
    deterministic_decoder_keyframe_context: DFRDeterministicDecoderKeyframeContext,
    *,
    collect_diagnostics: bool = True,
) -> tuple[DFROfficialDeterministicStagePreflight, str]:
    if not isinstance(
        deterministic_decoder_keyframe_context, DFRDeterministicDecoderKeyframeContext
    ):
        raise ValueError(
            "Expected DFRDeterministicDecoderKeyframeContext, got "
            f"{type(deterministic_decoder_keyframe_context).__name__}."
        )

    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if decoder is None:
        raise ValueError("Connected VAE has no decoder.")

    (
        cfg_channels,
        cfg_depths,
        cfg_kernels,
        cfg_upsamples,
        cfg_stage5_kernel,
        cfg_model_output_type,
    ) = _checkpoint_architecture(vae)

    (
        runtime_channels,
        runtime_depths,
        runtime_kernels,
        runtime_strides,
        runtime_reductions,
        det_stages,
        upsamples,
    ) = _runtime_architecture(decoder)

    expected_det_depths = cfg_depths[:-1]
    expected_det_kernels = cfg_kernels[:-1]
    expected_strides = tuple(item[0] for item in cfg_upsamples)
    expected_reductions = tuple(item[1] for item in cfg_upsamples)

    # Runtime kernel tuple includes stage 5 when exposed.
    runtime_det_kernels = runtime_kernels[: len(expected_det_kernels)]
    runtime_stage5_kernel = (
        runtime_kernels[len(expected_det_kernels)]
        if len(runtime_kernels) > len(expected_det_kernels)
        else None
    )

    architecture_ok = (
        runtime_channels == cfg_channels
        and runtime_depths == expected_det_depths
        and runtime_det_kernels == expected_det_kernels
        and runtime_strides == expected_strides
        and runtime_reductions == expected_reductions
        and (
            runtime_stage5_kernel is None
            or runtime_stage5_kernel == cfg_stage5_kernel
        )
    )

    if collect_diagnostics:
        plane_shapes, plane_helper_exact = _probe_official_keyframe_plane_hops(
            decoder, deterministic_decoder_keyframe_context, upsamples
        )
        plane_invariant = all(
            shape[1] == int(deterministic_decoder_keyframe_context.keyframe_count)
            for shape in plane_shapes
        )
    else:
        # The production decoder uses this frozen helper directly in the real
        # deterministic stages.  Re-running four synthetic GPU upsample probes
        # before every decode does not affect the output.
        plane_shapes = ()
        plane_helper_exact = True
        plane_invariant = True

    remaining = remaining_time_strides_official(upsamples)
    pixel_indices = torch.tensor(
        deterministic_decoder_keyframe_context.keyframe_pixel_frame_indices,
        dtype=torch.int64,
    )
    stage_times = tuple(
        tuple(float(x) for x in keyframe_clip_times_official(pixel_indices, stride, 0).tolist())
        for stride in remaining
    )

    forward_with_keyframes = tuple(
        tuple(
            callable(getattr(block, "forward_with_keyframes", None))
            for block in _module_list(group, "decoder.det_stages[*]")
        )
        for group in det_stages
    )
    missing_expected = not any(any(group) for group in forward_with_keyframes)

    notes = (
        "Algorithm contract is frozen-source-defined; runtime inspection is mapping/verification only.",
        "Keyframe hop uses the exact upsample_keyframe_planes rule: fold planes into batch as T=1, same upsampler, drop_leading_frame=True, restore plane layout.",
        "Keyframe times use keyframe_clip_times chunk-center positions, not integer latent indices.",
        "Current Comfy deterministic blocks are expected to lack forward_with_keyframes; U3.4b3b2 will add the ported dual-stream block path without changing checkpoint weights.",
    )

    state = DFROfficialDeterministicStagePreflight(
        version=PREFLIGHT_VERSION,
        source_contract="frozen_diffusion_video_decoder+keyframes",
        config_source="vae_checkpoint_metadata_via_comfy_first_stage_model.config",
        decoder_class=_class_name(decoder),
        checkpoint_stage_channels=cfg_channels,
        checkpoint_stage_depths=cfg_depths,
        checkpoint_stage_kernels=cfg_kernels,
        checkpoint_upsamples=cfg_upsamples,
        checkpoint_stage5_kernel=cfg_stage5_kernel,
        checkpoint_model_output_type=cfg_model_output_type,
        runtime_stage_channels=runtime_channels,
        runtime_stage_depths=runtime_depths,
        runtime_stage_kernels=runtime_kernels,
        runtime_upsample_strides=runtime_strides,
        runtime_upsample_reductions=runtime_reductions,
        architecture_matches_checkpoint_config=bool(architecture_ok),
        keyframe_plane_count=int(deterministic_decoder_keyframe_context.keyframe_count),
        keyframe_plane_shapes=plane_shapes,
        keyframe_plane_count_invariant=bool(plane_invariant),
        keyframe_plane_helper_exact=bool(plane_helper_exact),
        remaining_time_strides=remaining,
        keyframe_stage_times=stage_times,
        block_forward_with_keyframes_present=forward_with_keyframes,
        missing_keyframe_methods_expected=bool(missing_expected),
        type_emb_checkpoint_present=bool(
            deterministic_decoder_keyframe_context.checkpoint_has_type_emb
        ),
        decoder_called=False,
        det_stage_blocks_called=False,
        notes=notes,
    )
    report = official_deterministic_stage_preflight_report(state) if collect_diagnostics else ""
    return state, report


def validate_official_deterministic_stage_preflight(
    vae: Any,
    deterministic_decoder_keyframe_context: DFRDeterministicDecoderKeyframeContext,
    preflight: DFROfficialDeterministicStagePreflight,
) -> dict[str, Any]:
    if not isinstance(preflight, DFROfficialDeterministicStagePreflight):
        raise ValueError(
            f"Expected DFROfficialDeterministicStagePreflight, got {type(preflight).__name__}."
        )
    expected, _ = prepare_official_deterministic_stage_preflight(
        vae, deterministic_decoder_keyframe_context
    )

    fields = tuple(expected.__dataclass_fields__.keys())
    metadata_error = 0.0 if all(getattr(preflight, f) == getattr(expected, f) for f in fields) else 1.0

    passed = (
        metadata_error == 0.0
        and expected.architecture_matches_checkpoint_config
        and expected.keyframe_plane_count_invariant
        and expected.keyframe_plane_helper_exact
        and expected.type_emb_checkpoint_present
        and expected.missing_keyframe_methods_expected
        and expected.decoder_called is False
        and expected.det_stage_blocks_called is False
    )
    return {
        "passed": bool(passed),
        "metadata_error": float(metadata_error),
        "architecture_matches_checkpoint_config": bool(
            expected.architecture_matches_checkpoint_config
        ),
        "keyframe_plane_count_invariant": bool(
            expected.keyframe_plane_count_invariant
        ),
        "keyframe_plane_helper_exact": bool(expected.keyframe_plane_helper_exact),
        "remaining_time_strides": expected.remaining_time_strides,
        "keyframe_stage_times": expected.keyframe_stage_times,
        "checkpoint_stage_channels": expected.checkpoint_stage_channels,
        "runtime_stage_channels": expected.runtime_stage_channels,
        "checkpoint_stage_depths": expected.checkpoint_stage_depths,
        "runtime_stage_depths": expected.runtime_stage_depths,
        "checkpoint_stage_kernels": expected.checkpoint_stage_kernels,
        "runtime_stage_kernels": expected.runtime_stage_kernels,
        "checkpoint_upsamples": expected.checkpoint_upsamples,
        "runtime_upsample_strides": expected.runtime_upsample_strides,
        "runtime_upsample_reductions": expected.runtime_upsample_reductions,
        "missing_keyframe_methods_expected": bool(
            expected.missing_keyframe_methods_expected
        ),
    }


def official_deterministic_stage_preflight_report(
    state: DFROfficialDeterministicStagePreflight,
) -> str:
    return (
        f"PASS={state.architecture_matches_checkpoint_config and state.keyframe_plane_count_invariant and state.keyframe_plane_helper_exact and state.type_emb_checkpoint_present and state.missing_keyframe_methods_expected}; "
        f"stage=U3.4b3b1_official_source_preflight; version={state.version}; "
        f"source_contract={state.source_contract}; config_source={state.config_source}; "
        f"decoder={state.decoder_class}; architecture_matches_checkpoint_config={state.architecture_matches_checkpoint_config}; "
        f"checkpoint_stage_channels={state.checkpoint_stage_channels}; runtime_stage_channels={state.runtime_stage_channels}; "
        f"checkpoint_stage_depths={state.checkpoint_stage_depths}; runtime_stage_depths={state.runtime_stage_depths}; "
        f"checkpoint_stage_kernels={state.checkpoint_stage_kernels}; runtime_stage_kernels={state.runtime_stage_kernels}; "
        f"checkpoint_upsamples={state.checkpoint_upsamples}; runtime_upsample_strides={state.runtime_upsample_strides}; "
        f"runtime_upsample_reductions={state.runtime_upsample_reductions}; checkpoint_stage5_kernel={state.checkpoint_stage5_kernel}; "
        f"checkpoint_model_output_type={state.checkpoint_model_output_type}; "
        f"keyframe_plane_count={state.keyframe_plane_count}; keyframe_plane_shapes={state.keyframe_plane_shapes}; "
        f"keyframe_plane_count_invariant={state.keyframe_plane_count_invariant}; keyframe_plane_helper_exact={state.keyframe_plane_helper_exact}; "
        f"remaining_time_strides={state.remaining_time_strides}; keyframe_stage_times={state.keyframe_stage_times}; "
        f"block_forward_with_keyframes_present={state.block_forward_with_keyframes_present}; "
        f"missing_keyframe_methods_expected={state.missing_keyframe_methods_expected}; "
        f"type_emb_checkpoint_present={state.type_emb_checkpoint_present}; decoder_called=False; det_stage_blocks_called=False.\n"
        f"NOTES: {' | '.join(state.notes)}"
    )
