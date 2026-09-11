"""Store repeated video timestep embeddings once, with exact token indexing."""
from contextlib import contextmanager
from functools import lru_cache
import logging

import torch


@lru_cache(maxsize=None)
def indexed_timestep_type(native_type):
    class IndexedTimestep(native_type):
        def __init__(self, data, inverse):
            super().__init__(data, None)
            self.inverse = inverse

        def expand(self):
            return self.data.index_select(1, self.inverse)

        def token_slice(self, start, stop):
            return self.data.index_select(1, self.inverse[start:stop])

        def expand_for_computation(self, table, batch_size, indices=slice(None)):
            # Perform the same addition before indexing. No approximation of
            # timestep values, masks, or modulation parameters is involved.
            values = super().expand_for_computation(table, batch_size, indices)
            return tuple(value.index_select(1, self.inverse) for value in values)
    return IndexedTimestep


def compact_prepare(original, indexed_type, timestep, batch_size, hidden_dtype, **kwargs):
    # The normal DFR workflow uses batch 1. Preserve native semantics for other
    # batches and unusual timestep inputs rather than merging batch maxima.
    if batch_size != 1 or timestep.ndim not in (2, 3) or kwargs.get("a_timestep") is None:
        return original(timestep, batch_size, hidden_dtype, **kwargs)
    mask = kwargs.get("grid_mask")
    selected = timestep[:, mask] if mask is not None else timestep
    if selected.numel() == 0 or (selected.ndim == 3 and selected.shape[-1] != 1):
        return original(timestep, batch_size, hidden_dtype, **kwargs)
    values, inverse = torch.unique(selected.reshape(-1), sorted=True, return_inverse=True)
    if values.numel() > 64 or values.numel() == selected.numel():
        return original(timestep, batch_size, hidden_dtype, **kwargs)
    reduced_kwargs = dict(kwargs)
    # The surviving values already account for the sparse grid. Force native
    # per-value preparation, never pretend a spatial mask is a temporal mask.
    reduced_kwargs.update(grid_mask=None, has_spatial_mask=True, orig_shape=None)
    result, embedded, extra = original(values.reshape(1, -1, 1), batch_size, hidden_dtype, **reduced_kwargs)
    result, embedded = list(result), list(embedded)
    def indexed(value):
        return indexed_type(value.expand(), inverse)
    result[0] = indexed(result[0])
    embedded[0] = indexed(embedded[0])
    if result[2]:
        cross = list(result[2])
        cross[1], cross[2] = indexed(cross[1]), indexed(cross[2])
        result[2] = cross
    logging.info("[LTX DFR optimization] compact timesteps: %d surviving video tokens -> %d unique values",
                 selected.numel(), values.numel())
    return result, embedded, extra


@contextmanager
def stage2_compact_timestep(model, enabled=True):
    if not enabled:
        yield
        return
    from comfy.ldm.lightricks.av_model import CompressedTimestep
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", model)
    original = diffusion._prepare_timestep
    had_override = "_prepare_timestep" in diffusion.__dict__
    override = diffusion.__dict__.get("_prepare_timestep")
    indexed_type = indexed_timestep_type(CompressedTimestep)
    # Unique-value selection and custom indexed containers stay outside tracing;
    # native projection modules and downstream block execution are retained.
    @torch.compiler.disable
    def prepare(timestep, batch_size, hidden_dtype, **kwargs):
        return compact_prepare(original, indexed_type, timestep, batch_size, hidden_dtype, **kwargs)
    try:
        diffusion._prepare_timestep = prepare
        yield
    finally:
        if had_override:
            diffusion._prepare_timestep = override
        else:
            del diffusion._prepare_timestep
