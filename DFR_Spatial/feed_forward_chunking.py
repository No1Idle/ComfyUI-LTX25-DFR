"""Scoped video feed-forward chunking for the spatial detailing pass."""
from contextlib import contextmanager
import logging

import torch
from .operation_profile import OperationProfile, ACTIVE_OPERATION_PROFILE


def chunked_forward(original, chunk_tokens, diagnostics=None):
    """Keep native projections/quantization/LoRA; bound their token dimension."""
    def forward(x, *args, **kwargs):
        if x.ndim != 3 or x.shape[1] <= chunk_tokens or (torch.is_grad_enabled() and x.requires_grad):
            return original(x, *args, **kwargs)
        result = None
        if diagnostics is not None:
            diagnostics.split_calls += 1
        for start in range(0, x.shape[1], chunk_tokens):
            stop = min(start + chunk_tokens, x.shape[1])
            if diagnostics is not None:
                diagnostics.chunk(stop - start)
            part = original(x[:, start:stop], *args, **kwargs)
            if result is None:
                result = part.new_empty((part.shape[0], x.shape[1], part.shape[2]))
            result[:, start:stop].copy_(part)
            del part
        return result
    return forward


@contextmanager
def stage2_feed_forward_chunking(model, chunk_tokens=4096, diagnose=False):
    """Restore instance methods even if loading, sampling or cleanup fails."""
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens < 0:
        raise ValueError("ff_chunk_tokens must be nonnegative (0 disables chunking).")
    if chunk_tokens == 0 and not diagnose:
        yield
        return
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None:
        diffusion = model
    blocks = getattr(diffusion, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Stage 2 feed-forward chunking requires LTX transformer_blocks; set ff_chunk_tokens=0 for other model wrappers.")
    saved = []
    saved_phases = []
    diagnostics = OperationProfile() if diagnose else None
    token = ACTIVE_OPERATION_PROFILE.set(diagnostics)
    try:
        for index, block in enumerate(blocks):
            ff = getattr(block, "ff", None)
            if ff is None or not callable(getattr(ff, "forward", None)):
                raise ValueError("Unsupported LTX block: missing video feed-forward module.")
            saved.append((ff, "forward" in ff.__dict__, ff.__dict__.get("forward")))
            forward = chunked_forward(ff.forward, chunk_tokens, diagnostics) if chunk_tokens else ff.forward
            ff.forward = diagnostics.wrap(forward, "video_ff", index) if diagnostics else forward
            if diagnostics:
                for name in ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn", "audio_ff"):
                    module = getattr(block, name, None)
                    if module is not None:
                        saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
                        module.forward = diagnostics.wrap(module.forward, name, index)
        if not saved:
            raise ValueError("No video feed-forward blocks found.")
        if diagnostics:
            # These phases are siblings of the block loop, not nested monitors;
            # each peak-reset window therefore has one unambiguous owner.
            for name in ("_process_input", "_prepare_timestep", "_prepare_context", "_prepare_positional_embeddings", "_process_output"):
                original = getattr(diffusion, name, None)
                if callable(original):
                    saved_phases.append((name, name in diffusion.__dict__, diffusion.__dict__.get(name)))
                    setattr(diffusion, name, diagnostics.wrap(original, name.lstrip("_"), -1))
                else:
                    logging.warning("[LTX DFR Stage 2 ops] phase unavailable: %s", name)
        logging.info("[LTX DFR optimization] feed-forward chunking: %d tokens, %d video blocks", chunk_tokens, len(blocks))
        if diagnostics:
            logging.info("[LTX DFR Stage 2 ops] diagnostics enabled: eager module boundaries and CUDA synchronization; timings are not comparable to normal runs.")
        yield
    finally:
        ACTIVE_OPERATION_PROFILE.reset(token)
        for name, had_override, override in reversed(saved_phases):
            if had_override:
                setattr(diffusion, name, override)
            else:
                delattr(diffusion, name)
        for ff, had_override, override in reversed(saved):
            if had_override:
                ff.forward = override
            else:
                del ff.forward
