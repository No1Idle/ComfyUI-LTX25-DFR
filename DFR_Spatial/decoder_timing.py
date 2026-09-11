"""Coarse CUDA-event decode timing without per-phase synchronization."""
from contextlib import contextmanager
from collections import defaultdict
import logging
import time
import torch

class DecodeTimings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.records = []
        self.attention_chunk_counts = defaultdict(int)
        self.started = time.perf_counter()

    @contextmanager
    def phase(self, name):
        cuda = self.device.type == 'cuda'
        if cuda:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(torch.cuda.current_stream(self.device))
        else:
            start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                end.record(torch.cuda.current_stream(self.device))
            else:
                end = time.perf_counter()
            self.records.append((name, start, end))

    def report(self, label, cache):
        if self.device.type == 'cuda' and self.records:
            self.records[-1][2].synchronize()
        totals = defaultdict(float)
        for name, start, end in self.records:
            totals[name] += start.elapsed_time(end) / 1000 if self.device.type == 'cuda' else end-start
        logging.getLogger(__name__).info(
            '[LTX DFR decode timing] %s; wall=%.2fs; %s; geometry_cache_hits=%d; misses=%d; cached=%.2f MiB',
            label, time.perf_counter()-self.started,
            '; '.join(f'{name}={seconds:.2f}s' for name, seconds in totals.items() if not name.startswith('stage5.')),
            cache['hits'], cache['misses'], cache['bytes']/1024**2,
        )

        details = {name.removeprefix('stage5.'): seconds for name, seconds in totals.items()
                   if name.startswith('stage5.') and name.count('.') == 1}
        if details:
            logging.getLogger(__name__).info(
                '[LTX DFR decode Stage 5 breakdown] %s; %s; other=%.2fs (included in stage5 total)',
                label, '; '.join(f'{name}={seconds:.2f}s' for name, seconds in details.items()),
                max(0.0, totals.get('stage5', 0.0) - sum(details.values())),
            )

        attention = {name.removeprefix('stage5.attention.'): seconds for name, seconds in totals.items()
                     if name.startswith('stage5.attention.')}
        if attention:
            logging.getLogger(__name__).info(
                '[LTX DFR decode attention breakdown] %s; %s; slab_and_other=%.2fs (included in attention total); tiles_1_chunk=%d; tiles_2_chunks=%d; tiles_4_chunks=%d',
                label, '; '.join(f'{name}={seconds:.2f}s' for name, seconds in attention.items()),
                max(0.0, totals.get('stage5.attention', 0.0) - sum(attention.values())),
                self.attention_chunk_counts[1], self.attention_chunk_counts[2], self.attention_chunk_counts[4],
            )
