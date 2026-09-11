"""Decode-scoped, bounded cache for immutable attention geometry only."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os

_ACTIVE = ContextVar('ltx_decoder_geometry_cache', default=None)

@contextmanager
def decoder_geometry_cache():
    # No full-volume features or expanded RoPE matrices belong in this cache.
    state = {'entries': {}, 'bytes': 0, 'hits': 0, 'misses': 0}
    token = _ACTIVE.set(None if os.getenv('LTX_DFR_DECODE_CACHE', '1') == '0' else state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)
        state['entries'].clear()

def cached_geometry(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        state = _ACTIVE.get()
        if state is None:
            return fn(*args, **kwargs)
        # Tensor objects are retained as identity keys, preventing allocator/id reuse.
        # Callers only supply geometry tensors that remain immutable during decode.
        key = (fn, args, tuple(sorted(kwargs.items())))
        if key in state['entries']:
            state['hits'] += 1
            return state['entries'][key]
        state['misses'] += 1
        value = fn(*args, **kwargs)
        size = value.numel() * value.element_size()
        if len(state['entries']) < 128 and state['bytes'] + size <= 4 * 1024**2:
            state['entries'][key] = value
            state['bytes'] += size
        return value
    return wrapped
