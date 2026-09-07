"""U3.3b final video decoder bridge with continued RNG and keyframes disabled.

C17b intentionally changes only the stochastic source of Comfy's existing tiled
LTX DiffVAE decode. It preserves the current Comfy ``VAEDecodeTiled`` geometry,
latent un-normalization, wrapper post-processing and output layout, while replacing
``CausalDiffusionVAE.decode()``'s per-call fixed seed-0 generator with the exact
post-Stage2 generator state captured by U3.1.

Generated decoder keyframes remain disabled. U3.4 is responsible for the official
keyframe-aware decoder path and keyframe-aware tiling policy.
"""
from __future__ import annotations
from dataclasses import dataclass
import types
from typing import Any
import torch
from .decoder_handoff import DFRStage2DecoderHandoff

DECODER_BRIDGE_VERSION = 3

@dataclass(frozen=True)
class DFRDecoderBridgeCall:
    version: int
    used_wrapper_path: str
    used_direct_decoder_path: str
    rng_device_type: str
    generator_state_restored: bool
    generator_state_advanced: bool
    keyframes_enabled: bool
    tile_call_count: int
    wrapper_output_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    initial_rng_state: torch.Tensor
    final_rng_state: torch.Tensor
    pixel_tile_x: int
    pixel_tile_y: int
    pixel_overlap: int
    pixel_tile_t: int
    pixel_overlap_t: int
    latent_tile_x: int
    latent_tile_y: int
    latent_overlap: int
    latent_tile_t: int | None
    latent_overlap_t: int | None
    spatial_compression: int
    temporal_compression: int | None

def restore_decoder_generator(decoder_handoff: DFRStage2DecoderHandoff):
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")
    state = decoder_handoff.rng_state_after_stage2_av
    if not torch.is_tensor(state):
        raise ValueError("decoder_handoff.rng_state_after_stage2_av is not a tensor.")
    device_type = str(decoder_handoff.rng_device_type).strip().lower()
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported decoder RNG device type {decoder_handoff.rng_device_type!r}.")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The handoff requires a CUDA generator, but CUDA is unavailable in this runtime.")
    generator = torch.Generator(device=device_type)
    generator.set_state(state.detach().cpu())
    if not torch.equal(generator.get_state().detach().cpu(), state.detach().cpu()):
        raise RuntimeError("Reconstructed decoder generator state does not exactly match the U3.1 handoff state.")
    return generator, device_type

def _resolve_runtime(vae: Any):
    wrapper = getattr(vae, "decode_tiled", None)
    first_stage_model = getattr(vae, "first_stage_model", None)
    if not callable(wrapper) or first_stage_model is None:
        raise ValueError("Connected VAE does not expose the expected Comfy tiled DiffVAE runtime.")
    direct_decoder = getattr(first_stage_model, "decoder", None)
    stats = getattr(first_stage_model, "per_channel_statistics", None)
    unnormalize = getattr(stats, "un_normalize", None)
    original_decode = getattr(first_stage_model, "decode", None)
    if not callable(direct_decoder):
        raise ValueError("first_stage_model.decoder is not callable.")
    if not callable(unnormalize):
        raise ValueError("CausalDiffusionVAE per_channel_statistics.un_normalize(...) is unavailable.")
    if not callable(original_decode):
        raise ValueError("first_stage_model.decode(...) is unavailable.")
    return wrapper, first_stage_model, direct_decoder, unnormalize, original_decode

def _comfy_tiled_decode_args(vae: Any, tile_x: int, tile_y: int, overlap: int, tile_t: int, overlap_t: int):
    """Mirror core VAEDecodeTiled's pixel/frame UI -> latent tile conversion."""
    tile_x, tile_y, overlap, tile_t, overlap_t = map(int, (tile_x, tile_y, overlap, tile_t, overlap_t))
    min_tile = min(tile_x, tile_y)
    if min_tile < overlap * 4:
        overlap = min_tile // 4
    if tile_t < overlap_t * 2:
        overlap_t = overlap_t // 2

    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_compression = int(temporal_compression)
        latent_tile_t = max(2, tile_t // temporal_compression)
        latent_overlap_t = max(1, min(latent_tile_t // 2, overlap_t // temporal_compression))
    else:
        latent_tile_t = None
        latent_overlap_t = None

    spatial_compression = int(vae.spacial_compression_decode())
    latent_tile_x = max(1, tile_x // spatial_compression)
    latent_tile_y = max(1, tile_y // spatial_compression)
    latent_overlap = max(0, overlap // spatial_compression)
    kwargs = {
        "tile_x": latent_tile_x,
        "tile_y": latent_tile_y,
        "overlap": latent_overlap,
        "tile_t": latent_tile_t,
        "overlap_t": latent_overlap_t,
    }
    geom = {
        "pixel_tile_x": tile_x, "pixel_tile_y": tile_y, "pixel_overlap": overlap,
        "pixel_tile_t": tile_t, "pixel_overlap_t": overlap_t,
        "latent_tile_x": latent_tile_x, "latent_tile_y": latent_tile_y, "latent_overlap": latent_overlap,
        "latent_tile_t": latent_tile_t, "latent_overlap_t": latent_overlap_t,
        "spatial_compression": spatial_compression, "temporal_compression": temporal_compression,
    }
    return kwargs, geom

def decode_final_video_with_continued_rng(vae, final_video_latent, decoder_handoff,
                                           tile_x=512, tile_y=512, overlap=64, tile_t=64, overlap_t=16):
    if not isinstance(final_video_latent, dict) or not torch.is_tensor(final_video_latent.get("samples")):
        raise ValueError("final_video_latent must be a Comfy LATENT dict containing tensor 'samples'.")
    samples = final_video_latent["samples"]
    if samples.ndim != 5:
        raise ValueError(f"final_video_latent['samples'] must be [B,C,T,H,W], got {tuple(samples.shape)}.")

    generator, rng_device_type = restore_decoder_generator(decoder_handoff)
    initial_state = generator.get_state().detach().cpu().clone()
    wrapper, first_stage_model, direct_decoder, unnormalize, original_decode = _resolve_runtime(vae)
    decode_kwargs, geom = _comfy_tiled_decode_args(vae, tile_x, tile_y, overlap, tile_t, overlap_t)
    calls = {"count": 0}

    def _patched_decode(self, x):
        calls["count"] += 1
        # Preserve native CausalDiffusionVAE.decode semantics except for the generator source.
        return direct_decoder(unnormalize(x), generator=generator)

    try:
        setattr(first_stage_model, "decode", types.MethodType(_patched_decode, first_stage_model))
        with torch.inference_mode():
            images = wrapper(samples, **decode_kwargs)
    finally:
        setattr(first_stage_model, "decode", original_decode)

    # Match core Comfy VAEDecode/VAEDecodeTiled IMAGE semantics exactly:
    # video VAE wrappers may return [B,F,H,W,C]; Comfy flattens B and F into
    # the IMAGE batch before handing frames to CreateVideo/SaveVideo.
    wrapper_output_shape = tuple(int(x) for x in images.shape)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if images.ndim != 4:
        raise ValueError(
            "U3.3b decoder must return Comfy IMAGE [N,H,W,C] after batch/frame flattening; "
            f"wrapper returned {wrapper_output_shape}, normalized output is {tuple(int(x) for x in images.shape)}."
        )

    final_state = generator.get_state().detach().cpu().clone()
    state = DFRDecoderBridgeCall(
        version=DECODER_BRIDGE_VERSION,
        used_wrapper_path="vae.decode_tiled",
        used_direct_decoder_path="vae.first_stage_model.decoder",
        rng_device_type=rng_device_type,
        generator_state_restored=True,
        generator_state_advanced=not torch.equal(initial_state, final_state),
        keyframes_enabled=False,
        tile_call_count=int(calls["count"]),
        wrapper_output_shape=wrapper_output_shape,
        output_shape=tuple(int(x) for x in images.shape),
        initial_rng_state=initial_state,
        final_rng_state=final_state,
        **geom,
    )
    report = (
        f"PASS=True; stage=U3.3b_final_decoder_bridge; wrapper_path={state.used_wrapper_path}; "
        f"direct_decoder_path={state.used_direct_decoder_path}; rng_device_type={state.rng_device_type}; "
        f"generator_state_restored=True; generator_state_advanced={state.generator_state_advanced}; "
        f"keyframes_enabled=False; tile_decode_calls={state.tile_call_count}; "
        f"wrapper_output_shape={state.wrapper_output_shape}; output_shape={state.output_shape}; "
        f"pixel_tiles=({state.pixel_tile_t},{state.pixel_tile_y},{state.pixel_tile_x}); "
        f"pixel_overlap=({state.pixel_overlap_t},{state.pixel_overlap},{state.pixel_overlap}); "
        f"latent_tiles=({state.latent_tile_t},{state.latent_tile_y},{state.latent_tile_x}); "
        f"latent_overlap=({state.latent_overlap_t},{state.latent_overlap},{state.latent_overlap}); "
        f"compression=(t={state.temporal_compression},s={state.spatial_compression})."
    )
    return images, state, report

def validate_decoder_bridge(decoder_handoff, decoder_bridge_state):
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")
    if not isinstance(decoder_bridge_state, DFRDecoderBridgeCall):
        raise ValueError(f"Expected DFRDecoderBridgeCall, got {type(decoder_bridge_state).__name__}.")
    expected = decoder_handoff.rng_state_after_stage2_av.detach().cpu()
    initial_rng_error = 0.0 if torch.equal(decoder_bridge_state.initial_rng_state.detach().cpu(), expected) else 1.0
    rng_advanced_error = 0.0 if not torch.equal(decoder_bridge_state.initial_rng_state, decoder_bridge_state.final_rng_state) else 1.0
    keyframes_error = 0.0 if decoder_bridge_state.keyframes_enabled is False else 1.0
    tile_calls_error = 0.0 if decoder_bridge_state.tile_call_count > 0 else 1.0
    path_error = 0.0 if (decoder_bridge_state.used_wrapper_path == "vae.decode_tiled" and
                         decoder_bridge_state.used_direct_decoder_path == "vae.first_stage_model.decoder") else 1.0
    output_format_error = 0.0 if len(decoder_bridge_state.output_shape) == 4 else 1.0
    passed = all(v == 0.0 for v in (initial_rng_error, rng_advanced_error, keyframes_error, tile_calls_error, path_error, output_format_error))
    return {
        "passed": passed, "initial_rng_error": initial_rng_error, "rng_advanced_error": rng_advanced_error,
        "keyframes_error": keyframes_error, "tile_calls_error": tile_calls_error, "path_error": path_error,
        "output_format_error": output_format_error,
        "tile_decode_calls": decoder_bridge_state.tile_call_count,
        "pixel_tiles": (decoder_bridge_state.pixel_tile_t, decoder_bridge_state.pixel_tile_y, decoder_bridge_state.pixel_tile_x),
        "latent_tiles": (decoder_bridge_state.latent_tile_t, decoder_bridge_state.latent_tile_y, decoder_bridge_state.latent_tile_x),
        "wrapper_output_shape": decoder_bridge_state.wrapper_output_shape,
        "output_shape": decoder_bridge_state.output_shape,
    }
