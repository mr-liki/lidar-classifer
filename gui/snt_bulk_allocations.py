"""Bound collection overhead while constructing large, acyclic SNT models."""
from contextlib import contextmanager
import gc
import threading


_lock = threading.RLock()


@contextmanager
def bulk_snt_allocations():
    # GC thresholds are process-wide. Serialize SNT builders and restore the
    # caller's policy even on cancellation/errors. Collection stays enabled;
    # batching generation-0 collections avoids repeatedly walking millions of
    # live geometry containers. Refcount cleanup and explicit GC still work.
    with _lock:
        previous = gc.get_threshold()
        active = (max(previous[0], 50_000), previous[1], previous[2])
        if previous[0] == 0:  # respect an explicitly disabled automatic GC
            active = previous
        gc.set_threshold(*active)
        try:
            yield
        finally:
            if gc.get_threshold() == active:
                gc.set_threshold(*previous)
