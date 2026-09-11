"""Scoped, token-chunked LTXAV output modulation and projection."""
from contextlib import contextmanager
import logging
import math

import torch


def timestep_slice(timestep, start, stop):
    if callable(getattr(timestep, "token_slice", None)):
        return timestep.token_slice(start, stop)
    if torch.is_tensor(timestep):
        return timestep if timestep.shape[1] == 1 else timestep[:, start:stop]
    # Native CompressedTimestep: expand only the requested token window.
    if hasattr(timestep, "data") and hasattr(timestep, "patches_per_frame"):
        if timestep.data.shape[1] == 1 and timestep.patches_per_frame == 1:
            return timestep.data
        indices = torch.arange(start, stop, device=timestep.data.device) // timestep.patches_per_frame
        return timestep.data.index_select(1, indices)
    raise TypeError("Unsupported LTX output timestep representation.")


def chunked_output(model, x, embedded_timestep, keyframe_idxs, chunk_tokens, **kwargs):
    vx, ax = x
    video_timestep, audio_timestep = embedded_timestep
    video = None
    table = model.scale_shift_table[None, None].to(device=vx.device, dtype=vx.dtype)
    for start in range(0, vx.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, vx.shape[1])
        values = table + timestep_slice(video_timestep, start, stop)[:, :, None]
        shift, scale = values[:, :, 0], values[:, :, 1]
        part = model.norm_out(vx[:, start:stop])
        part = part * (1 + scale) + shift
        part = model.proj_out(part)
        if video is None:
            video = part.new_empty((part.shape[0], vx.shape[1], part.shape[2]))
        video[:, start:stop].copy_(part)
        del values, shift, scale, part

    # Match native guide-hole restoration and video unpatchification exactly.
    if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
        full = video.new_zeros(kwargs["orig_patchified_shape"])
        full[:, kwargs["grid_mask"], :] = video
        video = full
    shape = kwargs["orig_shape"]
    video = model.patchifier.unpatchify(
        latents=video, output_height=shape[3], output_width=shape[4],
        output_num_frames=shape[2], out_channels=shape[1] // math.prod(model.patchifier.patch_size),
    )

    # Audio is small: retain the native calculation and reference trimming.
    ref = kwargs.get("ref_audio_seq_len", 0)
    if ref > 0:
        ax = ax[:, ref:]
        if audio_timestep.shape[1] > 1:
            audio_timestep = audio_timestep[:, ref:]
    values = model.audio_scale_shift_table[None, None].to(device=audio_timestep.device, dtype=audio_timestep.dtype) + audio_timestep[:, :, None]
    shift, scale = values[:, :, 0], values[:, :, 1]
    ax = model.audio_norm_out(ax)
    ax = ax * (1 + scale) + shift
    ax = model.audio_proj_out(ax)
    ax = model.a_patchifier.unpatchify(ax, channels=model.num_audio_channels, freq=model.audio_frequency_bins)
    return model.recombine_audio_and_video_latents(video, ax, kwargs.get("av_orig_shape"))


@contextmanager
def stage2_output_chunking(model, chunk_tokens=4096):
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens < 0:
        raise ValueError("output_chunk_tokens must be nonnegative; 0 disables.")
    if not chunk_tokens:
        yield
        return
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", model)
    required = ("_process_output", "norm_out", "proj_out", "scale_shift_table", "audio_scale_shift_table",
                "audio_norm_out", "audio_proj_out", "patchifier", "a_patchifier", "recombine_audio_and_video_latents")
    if any(not hasattr(diffusion, name) for name in required):
        raise ValueError("Unsupported LTXAV output implementation; set output_chunk_tokens=0.")
    had_override = "_process_output" in diffusion.__dict__
    override = diffusion.__dict__.get("_process_output")
    def forward(x, embedded_timestep, keyframe_idxs, **kwargs):
        return chunked_output(diffusion, x, embedded_timestep, keyframe_idxs, chunk_tokens, **kwargs)
    try:
        diffusion._process_output = forward
        logging.info("[LTX DFR optimization] output chunking: %d tokens", chunk_tokens)
        yield
    finally:
        if had_override:
            diffusion._process_output = override
        else:
            del diffusion._process_output
