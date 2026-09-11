"""Diagnostic-only eager boundaries with per-operation CUDA peak accounting."""
from contextvars import ContextVar
import logging

import torch

ACTIVE_OPERATION_PROFILE = ContextVar("ltx_dfr_operation_profile", default=None)
log = logging.getLogger(__name__)


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class OperationProfile:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows = {}
        self.chunks = 0
        self.max_chunk = 0
        self.split_calls = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.gap_peak = 0

    def capture(self, device, gap=False):
        allocated = torch.cuda.max_memory_allocated(device)
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(device))
        if gap:
            self.gap_peak = max(self.gap_peak, allocated)
        return allocated

    def chunk(self, tokens):
        self.chunks += 1
        self.max_chunk = max(self.max_chunk, tokens)

    def wrap(self, original, label, block):
        # These are real runtime counters, not Python side effects during tracing.
        # Deliberately changes compilation boundaries in diagnostic mode only.
        @torch.compiler.disable
        def forward(*args, **kwargs):
            x = _first_tensor((args, kwargs))
            cuda = x is not None and x.device.type == "cuda"
            if cuda:
                torch.cuda.synchronize(x.device)
                self.capture(x.device, gap=True)
                entry = torch.cuda.memory_allocated(x.device)
                torch.cuda.reset_peak_memory_stats(x.device)
            row = self.rows.setdefault(label, {"calls": 0, "tokens": 0, "peak": 0, "extra": 0, "block": -1, "entry": 0, "exit": 0, "shape": ()})
            row["calls"] += 1
            if x is not None:
                row["shape"] = tuple(x.shape)
                row["tokens"] = max(row["tokens"], int(x.shape[1]) if x.ndim > 1 else x.numel())
            if label == "prepare_timestep":
                row["timestep_policy"] = "has_spatial_mask=%s; orig_shape=%s; grid_mask_present=%s" % (
                    kwargs.get("has_spatial_mask"), kwargs.get("orig_shape"), kwargs.get("grid_mask") is not None,
                )
            try:
                return original(*args, **kwargs)
            finally:
                if cuda:
                    torch.cuda.synchronize(x.device)
                    peak = self.capture(x.device)
                    if peak >= row["peak"]:
                        row["peak"], row["block"] = peak, block
                    row["extra"] = max(row["extra"], peak - entry)
                    row["entry"] = max(row["entry"], entry)
                    row["exit"] = max(row["exit"], torch.cuda.memory_allocated(x.device))
                    torch.cuda.reset_peak_memory_stats(x.device)
        return forward

    def report(self, step):
        log.info("[LTX DFR Stage 2 ops] step %d: split_ff_calls=%d; executed_chunks=%d; max_chunk_tokens=%d",
                 step + 1, self.split_calls, self.chunks, self.max_chunk)
        if not self.rows:
            log.warning("[LTX DFR Stage 2 ops] NO instrumented calls observed; execution verification failed.")
        elif "video_ff" not in self.rows:
            log.warning("[LTX DFR Stage 2 ops] No video feed-forward calls observed; chunk execution is NOT verified.")
        for label, row in self.rows.items():
            log.info("[LTX DFR Stage 2 ops] %s: calls=%d; max_tokens=%d; peak_total=%.0f MiB; extra_over_entry=%.0f MiB; peak_block=%d",
                     label, row["calls"], row["tokens"], row["peak"] / 1024**2,
                     row["extra"] / 1024**2, row["block"])
            if block_is_phase(label):
                log.info("[LTX DFR Stage 2 ops] %s: input_shape=%s; max_entry=%.0f MiB; max_exit=%.0f MiB%s",
                         label, row["shape"], row["entry"] / 1024**2, row["exit"] / 1024**2,
                         "; " + row["timestep_policy"] if "timestep_policy" in row else "")
        log.info("[LTX DFR Stage 2 ops] outside monitored calls: peak_total=%.0f MiB", self.gap_peak / 1024**2)


def block_is_phase(label):
    return label in ("process_input", "prepare_timestep", "prepare_context", "prepare_positional_embeddings", "process_output")
