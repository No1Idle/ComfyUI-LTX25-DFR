"""Choose attention slab width from current CUDA workspace headroom."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import torch


_DECODE_CHUNKS = ContextVar('ltx_decode_attention_chunks', default=None)

@contextmanager
def decoder_attention_chunks(chunks=1):
    if chunks not in (1, 2, 4):
        raise ValueError('attention_chunks must be 1, 2, or 4.')
    token = _DECODE_CHUNKS.set(chunks)
    try:
        yield
    finally:
        _DECODE_CHUNKS.reset(token)


def _attention_workspace_estimate_bytes(video, keyframes, halo, chunks):
    """Conservative estimate for one width-slab attention workspace.

    The executor materializes one fixed-extent slab at a time. Fewer chunks
    reduce duplicated halo work but increase the peak width of that slab.
    """
    width = int(video.shape[3])
    chunk_w = (width + int(chunks) - 1) // int(chunks)
    extent = chunk_w + 2 * int(halo)
    slab_bytes = (
        int(video.shape[0])
        * (int(video.shape[1]) + int(keyframes.shape[1]))
        * int(video.shape[2])
        * extent
        * int(video.shape[4])
        * video.element_size()
    )
    # Conservative working estimate for slabs, normalized inputs, Q/K/V, RoPE,
    # attention outputs and projection temporaries. Leave another 512 MiB free.
    return 24 * slab_bytes + 512 * 1024**2


def choose_attention_chunks(video, keyframes, halo, default=4):
    selected = _DECODE_CHUNKS.get()
    if selected is not None:
        return selected
    mode = os.getenv('LTX_DFR_DECODE_ATTENTION_CHUNKS', 'auto').lower()
    if mode not in ('auto', '1', '2', '4'):
        raise ValueError('LTX_DFR_DECODE_ATTENTION_CHUNKS must be auto, 1, 2, or 4.')
    if mode != 'auto':
        return int(mode)
    if video.device.type != 'cuda' or int(default) != 4:
        return int(default)

    free, _ = torch.cuda.mem_get_info(video.device)
    # Reusable PyTorch cache is available too; other allocations stay charged.
    reusable = max(
        0,
        torch.cuda.memory_reserved(video.device)
        - torch.cuda.memory_allocated(video.device),
    )
    available = int(free) + int(reusable)

    # Prefer the smallest chunk count that fits. This minimizes repeated halo
    # normalization/QKV/RoPE/attention work while preserving the exact slab path.
    for chunks in (1, 2):
        if _attention_workspace_estimate_bytes(video, keyframes, halo, chunks) <= available:
            return chunks
    return 4
