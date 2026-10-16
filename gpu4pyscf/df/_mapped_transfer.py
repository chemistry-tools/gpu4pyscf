"""Bounded asynchronous copies of fitted integrals into mapped host storage."""

from __future__ import annotations

import operator

_MAX_WORKSPACE_BYTES = 2 * 1024**3
_MIN_ELEMENTS = 1024 * 1024


def _workspace_bytes(naux, pairs):
    if naux <= 0 or pairs <= 0 or max(naux, pairs) > 2**31-1:
        return None
    size = naux*pairs*8
    return size if _MIN_ELEMENTS*8 <= size <= _MAX_WORKSPACE_BYTES else None


def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


def _available_memory():
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


class _MappedCderiWriter:
    """Own one GPU transpose buffer and its pending host transfer during a build."""

    @classmethod
    def create(cls, host, max_pairs):
        import cupy as cp
        import numpy as np

        if (not isinstance(host, np.ndarray) or host.ndim != 2 or host.dtype != np.float64
                or not host.flags.c_contiguous or not host.flags.writeable
                or not isinstance(host.base, cp.cuda.PinnedMemoryPointer)
                or not 0 < max_pairs <= host.shape[1]):
            return None
        size = _workspace_bytes(host.shape[0], max_pairs)
        device = cp.cuda.device.get_device_id()
        if size is None or not _supported_device(device) or size > _available_memory() // 2:
            return None
        self = cls()
        self.host, self.device = host, device
        self.producer = cp.cuda.get_current_stream()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.ready = cp.cuda.Event(disable_timing=True)
        self.done = cp.cuda.Event(disable_timing=True)
        try:
            self.buffer = cp.empty(size//8, dtype=cp.float64)
        except cp.cuda.memory.OutOfMemoryError:
            return None
        self.pending = False
        return self

    @property
    def workspace_bytes(self):
        return self.buffer.nbytes if self.buffer is not None else 0

    def enqueue(self, source, p0, p1):
        """Schedule an exact write, or return False without scheduling work."""
        import cupy as cp

        try:
            p0, p1 = operator.index(p0), operator.index(p1)
        except TypeError:
            return False
        if (self.buffer is None or not isinstance(source, cp.ndarray)
                or source.ndim != 2 or source.dtype != cp.float64
                or not source.flags.c_contiguous or source.device.id != self.device
                or cp.cuda.device.get_device_id() != self.device
                or cp.cuda.get_current_stream().ptr != self.producer.ptr
                or not 0 <= p0 < p1 <= self.host.shape[1]
                or source.shape != (p1-p0, self.host.shape[0])
                or source.nbytes > self.buffer.nbytes):
            return False
        s0, b0 = source.data.ptr, self.buffer.data.ptr
        if s0 < b0+self.buffer.nbytes and b0 < s0+source.nbytes:
            return False
        if self.pending:
            # The next integral/GEMM has already been queued on the producer.
            # Wait only before overwriting the transpose scratch, allowing that
            # computation to overlap the previous DMA transfer.
            self.producer.wait_event(self.done)
        scratch = self.buffer[:source.size].reshape(self.host.shape[0], p1-p0)
        cp.copyto(scratch, source.T)
        self.ready.record(self.producer)
        self.stream.wait_event(self.ready)
        self.pending = True
        cp.cuda.runtime.memcpy2DAsync(
            self.host.ctypes.data+p0*8, self.host.shape[1]*8,
            scratch.data.ptr, (p1-p0)*8, (p1-p0)*8, self.host.shape[0],
            cp.cuda.runtime.memcpyDeviceToHost, self.stream.ptr)
        self.done.record(self.stream)
        return True

    def finish(self):
        """Complete pending writes before host access or the native fallback."""
        if self.pending:
            self.stream.synchronize()
            self.pending = False

    def close(self):
        try:
            self.finish()
        finally:
            self.buffer = None
