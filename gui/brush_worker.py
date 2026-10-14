"""
BrushQueryWorker — background QThread for brush spatial queries.

The scipy cKDTree.query_ball_point() releases the GIL in its C layer.
Running it on a QThread means the main thread's mouse handler stays at
144 Hz (< 7 ms budget) while classification indices are computed in parallel.

Worker-Watcher pattern
----------------------
Main thread              Worker thread
─────────────────────    ─────────────────────────────────────
on_mouse_move()          run()  (blocked on queue.get)
  worker.post_query()  ──► queue.put_nowait(item)
                         ◄── spatial_query + filter (GIL released in C)
  _on_worker_result()  ◄── Signal result_ready.emit(indices, to_class)
    cpu_classify()
    gpu_poke()

Usage
-----
    worker = BrushQueryWorker()
    worker.result_ready.connect(self._on_worker_result, Qt.QueuedConnection)
    worker.start()

    # on mouse-move:
    worker.post_query(center_xy, radius, xyz, classes, to_class, from_classes)

    # on shutdown:
    worker.stop()
    worker.wait(2000)
"""

import queue
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt


class BrushQueryWorker(QThread):
    """
    Background thread for brush KDTree spatial queries.

    Signals
    -------
    result_ready(indices: np.ndarray, to_class: int)
        Emitted from the worker thread; Qt auto-routes it to the main
        thread via a QueuedConnection so the slot runs on the UI thread.
        `indices` is a 1-D int64 array of matching point indices
        (already filtered by `from_classes`).
    """

    result_ready = Signal(object, int)   # (np.ndarray, to_class)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Capacity-2 queue: if the worker is slower than the mouse,
        # the oldest stale query is dropped so latency stays bounded.
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop  = False

    # ──────────────────────────────────────────────────────────────────────
    def post_query(
        self,
        center_xy: tuple,
        radius: float,
        xyz: np.ndarray,
        classes: np.ndarray,
        to_class: int,
        from_classes=None,
    ) -> None:
        """
        Post a spatial query from the main thread.

        Non-blocking: if the queue is full the oldest item is evicted first.
        The caller (mouse handler) is NEVER blocked.
        """
        item = (center_xy, radius, xyz, classes, to_class, from_classes)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()   # evict stale query
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass   # ultra-rare; just skip

    def stop(self) -> None:
        """Signal the worker to exit after finishing its current query."""
        self._stop = True
        try:
            self._queue.put_nowait(None)   # unblock queue.get()
        except queue.Full:
            pass

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Worker loop — executed on the background thread."""
        from gui.spatial_index import get_or_build_index

        while not self._stop:
            # Block until a query arrives (timeout allows stop() to be seen)
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if item is None or self._stop:
                break

            center_xy, radius, xyz, classes, to_class, from_classes = item

            try:
                hit = self._query(center_xy, radius, xyz, classes, from_classes)
            except Exception:
                import traceback
                traceback.print_exc()
                hit = np.empty(0, dtype=np.int64)

            # Emit on this thread; Qt auto-queues delivery to main thread
            self.result_ready.emit(hit, int(to_class))

    # ──────────────────────────────────────────────────────────────────────
    def _query(
        self,
        center_xy: tuple,
        radius: float,
        xyz: np.ndarray,
        classes: np.ndarray,
        from_classes,
    ) -> np.ndarray:
        """
        Spatial query + from_classes filter.

        Uses the persistent shared Phase-2 XY grid built once per dataset.
        The exact circle test runs only on points from intersecting cells.

        Returns 1-D int64 array of matching global point indices.
        """
        from gui.spatial_index import get_or_build_index

        idx_obj = get_or_build_index(xyz)   # O(1) cache hit after first load
        cx, cy  = float(center_xy[0]), float(center_xy[1])

        # Phase 2: the brush is an XY editing tool. Query the shared XY grid
        # directly instead of creating a huge 3-D bounding sphere and forcing
        # a KD-tree search across the cloud's complete Z range.
        hit = np.asarray(
            idx_obj.query_circle_2d((cx, cy), float(radius)),
            dtype=np.int64,
        )

        if hit.size == 0:
            return hit

        # Filter by from_classes (only reclassify allowed source classes)
        if from_classes and len(from_classes) > 0 and classes is not None:
            fc_mask = np.isin(classes[hit], list(from_classes),
                              assume_unique=False)
            hit = hit[fc_mask]

        return hit
