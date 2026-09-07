"""U3.4b3a geometry-correct deterministic decoder stage/keyframe context.

C22b fixes the C22 schedule model to use the *actual loaded decoder upsample
strides* instead of assuming every deterministic upsampler is spatial x2.

It also fixes the keyframe preparation order before U3.4b3b:
    decoder latent -> per-channel un-normalize -> add trained decoder.type_emb

No deterministic decoder block is executed here yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_keyframe_substrate import DFRDecoderKeyframeSubstrate
from .decoder_keyframe_checkpoint import DFRDecoderKeyframeCheckpointWeights
from .decoder_joint_attention import DFRJointDecoderAttention, KEYFRAME_CONTEXT_SLOTS
from .decoder_keyframes import DFRFinalDecodeKeyframes
from .dfr_layout import validate_layout

DET_STAGE_CONTEXT_VERSION = 2


@dataclass(frozen=True)
class DFRDeterministicDecoderStage:
    stage_index: int
    stage_name: str
    spatial_shape: tuple[int, int]
    spatial_scale_from_latent: tuple[int, int]
    temporal_unit_pixels: int
    expected_video_frames: int
    keyframe_frame_indices: tuple[int, ...]
    keyframe_valid_mask: tuple[bool, ...]
    upsample_stride_from_previous: tuple[int, int, int]


@dataclass(frozen=True)
class DFRDeterministicDecoderKeyframeContext:
    version: int
    stage_mode: str
    stage_count: int
    deterministic_stage_count: int
    top_level_upsample_count: int
    checkpoint_has_type_emb: bool
    type_emb_source_key: str | None
    type_emb_shape: tuple[int, ...]
    type_emb_dtype: str
    joint_attention_backend: str
    joint_attention_num_slots: int
    keyframe_channels: int
    keyframe_count: int
    keyframe_pixel_frame_indices: tuple[int, ...]
    keyframe_latent_frame_indices: tuple[int, ...]
    un_normalize_applied: bool
    typed_keyframe_latents: torch.Tensor
    typed_keyframe_shape: tuple[int, ...]
    typed_keyframe_dtype: str
    upsample_strides: tuple[tuple[int, int, int], ...]
    temporal_scale_schedule: tuple[int, ...]
    spatial_scale_schedule: tuple[tuple[int, int], ...]
    patch_size: int
    deterministic_output_spatial_shape: tuple[int, int]
    final_pixel_spatial_shape: tuple[int, int]
    stage_descriptors: tuple[DFRDeterministicDecoderStage, ...]
    notes: tuple[str, ...]


_OFFICIAL_PRODUCTION_STRIDES = (
    (1, 2, 2),
    (2, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
)


def _resolve_runtime_geometry(vae: Any):
    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if first_stage_model is None or decoder is None:
        raise ValueError("Connected VAE does not expose vae.first_stage_model.decoder.")

    upsamplers = getattr(decoder, "upsamples", None)
    if upsamplers is None:
        raise ValueError("Loaded decoder does not expose decoder.upsamples.")

    strides: list[tuple[int, int, int]] = []
    for idx, upsample in enumerate(upsamplers):
        stride = getattr(upsample, "stride", None)
        if stride is None:
            raise ValueError(f"decoder.upsamples[{idx}] does not expose a stride attribute.")
        stride = tuple(int(x) for x in stride)
        if len(stride) != 3 or any(x < 1 for x in stride):
            raise ValueError(f"Invalid decoder upsample stride at index {idx}: {stride}.")
        strides.append(stride)

    patch_size = int(getattr(decoder, "patch_size", 0))
    if patch_size < 1:
        raise ValueError(f"Loaded decoder has invalid patch_size={patch_size}.")

    stats = getattr(first_stage_model, "per_channel_statistics", None)
    un_normalize = getattr(stats, "un_normalize", None)
    if not callable(un_normalize):
        raise ValueError(
            "Loaded CausalDiffusionVAE does not expose per_channel_statistics.un_normalize(...)."
        )
    return first_stage_model, decoder, tuple(strides), patch_size, un_normalize


def _remaining_temporal_strides(
    strides: tuple[tuple[int, int, int], ...]
) -> tuple[int, ...]:
    # Same meaning as upstream remaining_time_strides(self.upsamples):
    # divisor for keyframe pixel positions at each deterministic-stage input,
    # plus stage-5 input.
    out: list[int] = []
    for start in range(len(strides) + 1):
        product = 1
        for stride in strides[start:]:
            product *= int(stride[0])
        out.append(product)
    return tuple(out)


def _build_stage_descriptors(
    requested_frames: int,
    keyframe_pixel_indices: tuple[int, ...],
    base_shape: tuple[int, int],
    temporal_scale: int,
    strides: tuple[tuple[int, int, int], ...],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[DFRDeterministicDecoderStage, ...],
]:
    temporal_units = _remaining_temporal_strides(strides)
    if temporal_units[0] != temporal_scale:
        raise ValueError(
            "Loaded decoder temporal upsample ladder disagrees with the DFR layout: "
            f"decoder={temporal_units[0]}, layout={temporal_scale}."
        )

    h, w = int(base_shape[0]), int(base_shape[1])
    scale_h = scale_w = 1
    t_frames = int((requested_frames - 1) // temporal_scale + 1)

    shapes = [(h, w)]
    spatial_scales = [(1, 1)]
    video_lengths = [t_frames]

    for p_t, p_h, p_w in strides:
        scale_h *= p_h
        scale_w *= p_w
        h *= p_h
        w *= p_w
        # Comfy LinearPixelShuffleUpsample drops the duplicated leading frame
        # whenever p_t == 2 on the full causal decode.
        t_frames = t_frames * p_t - (1 if p_t == 2 else 0)
        shapes.append((h, w))
        spatial_scales.append((scale_h, scale_w))
        video_lengths.append(t_frames)

    valid_mask = tuple(True for _ in keyframe_pixel_indices)
    descriptors: list[DFRDeterministicDecoderStage] = []
    names = ["latent_input"] + [
        f"det_stage_{idx}_out" for idx in range(1, len(strides) + 1)
    ]
    hop_strides = [(1, 1, 1)] + list(strides)

    for idx, (name, shape, spatial_scale, unit, video_frames, hop_stride) in enumerate(
        zip(
            names,
            shapes,
            spatial_scales,
            temporal_units,
            video_lengths,
            hop_strides,
        )
    ):
        if any(frame % unit != 0 for frame in keyframe_pixel_indices):
            raise ValueError(
                f"Keyframe pixel indices {keyframe_pixel_indices} are not divisible by "
                f"stage temporal unit {unit}."
            )
        descriptors.append(
            DFRDeterministicDecoderStage(
                stage_index=idx,
                stage_name=name,
                spatial_shape=shape,
                spatial_scale_from_latent=spatial_scale,
                temporal_unit_pixels=int(unit),
                expected_video_frames=int(video_frames),
                keyframe_frame_indices=tuple(
                    int(frame // unit) for frame in keyframe_pixel_indices
                ),
                keyframe_valid_mask=valid_mask,
                upsample_stride_from_previous=tuple(int(x) for x in hop_stride),
            )
        )

    return temporal_units, tuple(spatial_scales), tuple(descriptors)


def prepare_deterministic_decoder_keyframe_context(
    vae: Any,
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
    decoder_keyframe_checkpoint_weights: DFRDecoderKeyframeCheckpointWeights,
    joint_decoder_attention: DFRJointDecoderAttention,
    final_decode_keyframes: DFRFinalDecodeKeyframes,
    dfr_layout: dict[str, Any],
) -> tuple[DFRDeterministicDecoderKeyframeContext, str]:
    if not isinstance(decoder_keyframe_substrate, DFRDecoderKeyframeSubstrate):
        raise ValueError(
            f"Expected DFRDecoderKeyframeSubstrate, got "
            f"{type(decoder_keyframe_substrate).__name__}."
        )
    if not isinstance(
        decoder_keyframe_checkpoint_weights, DFRDecoderKeyframeCheckpointWeights
    ):
        raise ValueError(
            "Expected DFRDecoderKeyframeCheckpointWeights, got "
            f"{type(decoder_keyframe_checkpoint_weights).__name__}."
        )
    if not isinstance(joint_decoder_attention, DFRJointDecoderAttention):
        raise ValueError(
            f"Expected DFRJointDecoderAttention, got "
            f"{type(joint_decoder_attention).__name__}."
        )
    if not isinstance(final_decode_keyframes, DFRFinalDecodeKeyframes):
        raise ValueError(
            f"Expected DFRFinalDecodeKeyframes, got "
            f"{type(final_decode_keyframes).__name__}."
        )

    layout = validate_layout(dfr_layout)
    if not final_decode_keyframes.present or final_decode_keyframes.latents is None:
        raise ValueError(
            "U3.4b3a requires a present U3.2 final decoder-keyframe package."
        )
    if not decoder_keyframe_substrate.keyframe_channels_match_expected:
        raise ValueError("U3.4b3a requires the validated U3.4b1 substrate.")
    if not decoder_keyframe_checkpoint_weights.channels_match_expected:
        raise ValueError("U3.4b3a requires the validated U3.4b1b type_emb.")
    if not joint_decoder_attention.channels_match_expected:
        raise ValueError("U3.4b3a requires the validated U3.4b2 attention package.")
    if int(joint_decoder_attention.num_slots) != KEYFRAME_CONTEXT_SLOTS:
        raise ValueError(
            f"Expected {KEYFRAME_CONTEXT_SLOTS} joint-attention keyframe slots, "
            f"got {joint_decoder_attention.num_slots}."
        )

    _, _, strides, patch_size, un_normalize = _resolve_runtime_geometry(vae)
    if strides != _OFFICIAL_PRODUCTION_STRIDES:
        raise ValueError(
            "Loaded Comfy decoder upsample strides do not match the frozen official "
            f"LTX-2.5 production ladder: runtime={strides}, "
            f"official={_OFFICIAL_PRODUCTION_STRIDES}."
        )

    latents = final_decode_keyframes.latents
    # Exact intended ordering for the current Comfy decoder bridge:
    # un-normalize decoder latents, then add the trained keyframe tag,
    # immediately before the shared conv_in used by U3.4b3b.
    with torch.no_grad():
        unnormalized = un_normalize(latents)
        type_emb = decoder_keyframe_checkpoint_weights.type_emb.to(
            device=unnormalized.device, dtype=unnormalized.dtype
        )
        if type_emb.ndim != 1 or int(type_emb.shape[0]) != int(unnormalized.shape[1]):
            raise ValueError(
                f"decoder type_emb shape {tuple(type_emb.shape)} is incompatible with "
                f"keyframe channels {unnormalized.shape[1]}."
            )
        typed_keyframe_latents = (
            unnormalized + type_emb.view(1, -1, 1, 1, 1)
        ).contiguous()

    pixel_indices = tuple(
        int(x)
        for x in final_decode_keyframes.pixel_frame_indices.detach().cpu().tolist()
    )
    latent_indices = tuple(
        int(x) for x in decoder_keyframe_substrate.keyframe_latent_frame_indices
    )
    keyframe_count = int(final_decode_keyframes.kept_keyframe_count)

    temporal_schedule, spatial_schedule, stage_descriptors = _build_stage_descriptors(
        requested_frames=int(layout["requested_frames"]),
        keyframe_pixel_indices=pixel_indices,
        base_shape=decoder_keyframe_substrate.keyframe_spatial_shape,
        temporal_scale=int(layout["temporal_scale"]),
        strides=strides,
    )

    deterministic_output_shape = stage_descriptors[-1].spatial_shape
    final_pixel_shape = (
        deterministic_output_shape[0] * patch_size,
        deterministic_output_shape[1] * patch_size,
    )

    context = DFRDeterministicDecoderKeyframeContext(
        version=DET_STAGE_CONTEXT_VERSION,
        stage_mode="geometry_fixed_schedule_only_pre_integration",
        stage_count=len(stage_descriptors),
        deterministic_stage_count=len(strides),
        top_level_upsample_count=len(strides),
        checkpoint_has_type_emb=bool(
            decoder_keyframe_checkpoint_weights.checkpoint_has_type_emb
        ),
        type_emb_source_key=decoder_keyframe_checkpoint_weights.source_key,
        type_emb_shape=tuple(
            int(x) for x in decoder_keyframe_checkpoint_weights.type_emb_shape
        ),
        type_emb_dtype=str(decoder_keyframe_checkpoint_weights.type_emb_dtype),
        joint_attention_backend=str(joint_decoder_attention.source_backend),
        joint_attention_num_slots=int(joint_decoder_attention.num_slots),
        keyframe_channels=int(latents.shape[1]),
        keyframe_count=keyframe_count,
        keyframe_pixel_frame_indices=pixel_indices,
        keyframe_latent_frame_indices=latent_indices,
        un_normalize_applied=True,
        typed_keyframe_latents=typed_keyframe_latents,
        typed_keyframe_shape=tuple(int(x) for x in typed_keyframe_latents.shape),
        typed_keyframe_dtype=str(typed_keyframe_latents.dtype),
        upsample_strides=strides,
        temporal_scale_schedule=temporal_schedule,
        spatial_scale_schedule=spatial_schedule,
        patch_size=patch_size,
        deterministic_output_spatial_shape=deterministic_output_shape,
        final_pixel_spatial_shape=final_pixel_shape,
        stage_descriptors=stage_descriptors,
        notes=(
            "C22b uses the actual loaded decoder upsample strides; no per-stage x2 assumption remains.",
            "Keyframe preparation order is un-normalize -> trained decoder.type_emb; conv_in is intentionally deferred to U3.4b3b.",
            "No deterministic decoder block or Stage-5 diffusion block is executed in U3.4b3a.",
        ),
    )
    return context, deterministic_context_report(context)


def validate_deterministic_decoder_keyframe_context(
    vae: Any,
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
    decoder_keyframe_checkpoint_weights: DFRDecoderKeyframeCheckpointWeights,
    joint_decoder_attention: DFRJointDecoderAttention,
    final_decode_keyframes: DFRFinalDecodeKeyframes,
    dfr_layout: dict[str, Any],
    deterministic_decoder_keyframe_context: DFRDeterministicDecoderKeyframeContext,
) -> dict[str, Any]:
    if not isinstance(
        deterministic_decoder_keyframe_context,
        DFRDeterministicDecoderKeyframeContext,
    ):
        raise ValueError(
            "Expected DFRDeterministicDecoderKeyframeContext, got "
            f"{type(deterministic_decoder_keyframe_context).__name__}."
        )

    expected, _ = prepare_deterministic_decoder_keyframe_context(
        vae,
        decoder_keyframe_substrate,
        decoder_keyframe_checkpoint_weights,
        joint_decoder_attention,
        final_decode_keyframes,
        dfr_layout,
    )
    actual = deterministic_decoder_keyframe_context

    metadata_fields = (
        "version",
        "stage_mode",
        "stage_count",
        "deterministic_stage_count",
        "top_level_upsample_count",
        "checkpoint_has_type_emb",
        "type_emb_source_key",
        "type_emb_shape",
        "type_emb_dtype",
        "joint_attention_backend",
        "joint_attention_num_slots",
        "keyframe_channels",
        "keyframe_count",
        "keyframe_pixel_frame_indices",
        "keyframe_latent_frame_indices",
        "un_normalize_applied",
        "typed_keyframe_shape",
        "typed_keyframe_dtype",
        "upsample_strides",
        "temporal_scale_schedule",
        "spatial_scale_schedule",
        "patch_size",
        "deterministic_output_spatial_shape",
        "final_pixel_spatial_shape",
        "stage_descriptors",
        "notes",
    )
    metadata_error = (
        0.0
        if all(getattr(actual, k) == getattr(expected, k) for k in metadata_fields)
        else 1.0
    )

    if (
        not torch.is_tensor(actual.typed_keyframe_latents)
        or tuple(actual.typed_keyframe_latents.shape)
        != tuple(expected.typed_keyframe_latents.shape)
    ):
        tensor_error = 1.0
    else:
        delta = (
            actual.typed_keyframe_latents.detach().to(torch.float32).cpu()
            - expected.typed_keyframe_latents.detach().to(torch.float32).cpu()
        ).abs()
        tensor_error = float(delta.max().item()) if delta.numel() else 0.0

    # Prove type_emb is applied after un-normalization, not to the normalized input.
    _, _, _, _, un_normalize = _resolve_runtime_geometry(vae)
    with torch.no_grad():
        unnormalized = un_normalize(final_decode_keyframes.latents)
        type_emb = decoder_keyframe_checkpoint_weights.type_emb.to(
            device=unnormalized.device, dtype=unnormalized.dtype
        )
        expected_typed = unnormalized + type_emb.view(1, -1, 1, 1, 1)
        order_error = float(
            (
                actual.typed_keyframe_latents.to(expected_typed.device, torch.float32)
                - expected_typed.to(torch.float32)
            )
            .abs()
            .max()
            .item()
        )

    passed = metadata_error == 0.0 and tensor_error == 0.0 and order_error == 0.0
    return {
        "passed": bool(passed),
        "metadata_error": float(metadata_error),
        "tensor_error": float(tensor_error),
        "unnormalize_then_type_emb_error": float(order_error),
        "stage_count": int(expected.stage_count),
        "deterministic_stage_count": int(expected.deterministic_stage_count),
        "upsample_strides": expected.upsample_strides,
        "temporal_scale_schedule": expected.temporal_scale_schedule,
        "spatial_scale_schedule": expected.spatial_scale_schedule,
        "deterministic_output_spatial_shape": expected.deterministic_output_spatial_shape,
        "final_pixel_spatial_shape": expected.final_pixel_spatial_shape,
        "stage_names": tuple(stage.stage_name for stage in expected.stage_descriptors),
        "final_stage_keyframe_indices": expected.stage_descriptors[-1].keyframe_frame_indices,
    }


def deterministic_context_report(
    context: DFRDeterministicDecoderKeyframeContext,
) -> str:
    stage_lines = [
        (
            f"stage_index={stage.stage_index}, name={stage.stage_name}, "
            f"spatial_shape={stage.spatial_shape}, "
            f"spatial_scale={stage.spatial_scale_from_latent}, "
            f"temporal_unit_pixels={stage.temporal_unit_pixels}, "
            f"video_frames={stage.expected_video_frames}, "
            f"keyframe_indices={stage.keyframe_frame_indices}, "
            f"upsample_stride_from_previous={stage.upsample_stride_from_previous}"
        )
        for stage in context.stage_descriptors
    ]
    return (
        f"PASS=True; stage=U3.4b3a_geometry_fixed; mode={context.stage_mode}; "
        f"stage_count={context.stage_count}; "
        f"deterministic_stage_count={context.deterministic_stage_count}; "
        f"checkpoint_has_type_emb={context.checkpoint_has_type_emb}; "
        f"type_emb_source_key={context.type_emb_source_key}; "
        f"un_normalize_applied={context.un_normalize_applied}; "
        f"joint_attention_backend={context.joint_attention_backend}; "
        f"joint_attention_num_slots={context.joint_attention_num_slots}; "
        f"keyframe_count={context.keyframe_count}; "
        f"typed_keyframe_shape={context.typed_keyframe_shape}; "
        f"upsample_strides={context.upsample_strides}; "
        f"temporal_scale_schedule={context.temporal_scale_schedule}; "
        f"spatial_scale_schedule={context.spatial_scale_schedule}; "
        f"patch_size={context.patch_size}; "
        f"deterministic_output_spatial_shape={context.deterministic_output_spatial_shape}; "
        f"final_pixel_spatial_shape={context.final_pixel_spatial_shape}.\n"
        f"STAGES:\n- " + "\n- ".join(stage_lines) + "\n"
        f"NOTES: {' | '.join(context.notes)}"
    )
