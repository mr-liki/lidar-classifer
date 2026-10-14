"""Sparse classification transaction primitives for NakshaAI LiDAR.

Phase 1 rule: a classification edit is represented by the K canonical point
indices that changed, never by an N-point boolean mask unless a legacy consumer
explicitly requests one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass(slots=True)
class SparseClassificationDelta:
    """Exact canonical classification transition for one transaction."""

    indices: np.ndarray
    old_classes: Optional[np.ndarray] = None
    new_classes: Optional[np.ndarray] = None
    dataset_size: Optional[int] = None
    operation: str = "classification"
    origin_view: str = "unknown"
    tx_id: int = 0
    edit_bbox: Optional[Tuple[float, float, float, float, float, float]] = None

    def __post_init__(self) -> None:
        raw_idx = np.asarray(self.indices, dtype=np.int64).ravel()
        valid = raw_idx >= 0
        if self.dataset_size is not None:
            valid &= raw_idx < int(self.dataset_size)
        self.indices = raw_idx[valid]

        if self.old_classes is not None:
            old = np.asarray(self.old_classes).ravel()
            if old.size == raw_idx.size:
                old = old[valid]
            self.old_classes = old
        if self.new_classes is not None:
            new = np.asarray(self.new_classes).ravel()
            if new.size == raw_idx.size:
                new = new[valid]
            self.new_classes = new

    @property
    def count(self) -> int:
        return int(self.indices.size)

    @property
    def nbytes(self) -> int:
        """Bytes owned by the sparse transaction payload (not source arrays)."""
        total = int(self.indices.nbytes)
        if self.old_classes is not None:
            total += int(self.old_classes.nbytes)
        if self.new_classes is not None:
            total += int(self.new_classes.nbytes)
        return total

    def as_mask(self, size: Optional[int] = None, *, reason: str = "legacy") -> np.ndarray:
        """Materialize an N-point mask only for a consumer that truly requires it."""
        n = int(size if size is not None else (self.dataset_size or 0))
        if n <= 0:
            raise ValueError("dataset size is required to materialize a classification mask")
        mask = np.zeros(n, dtype=bool)
        idx = self.indices
        if idx.size:
            valid = idx[(idx >= 0) & (idx < n)]
            mask[valid] = True
        print(
            "CLASSIFY_PHASE1_LEGACY_MASK "
            f"reason={reason} points={idx.size} dataset={n} bytes={mask.nbytes}"
        )
        return mask


def _classification_size(app=None, n_points: Optional[int] = None) -> Optional[int]:
    if n_points is not None:
        return int(n_points)
    try:
        data = getattr(app, "data", None)
        cls = data.get("classification") if isinstance(data, dict) else None
        if cls is not None:
            return int(len(cls))
    except Exception:
        pass
    return None


def resolve_changed_indices(
    change=None,
    *,
    changed_indices=None,
    app=None,
    n_points: Optional[int] = None,
) -> np.ndarray:
    """Return canonical point IDs from a sparse delta, indices, or legacy mask."""
    n = _classification_size(app, n_points)

    source = changed_indices if changed_indices is not None else change
    if source is None and app is not None:
        delta = getattr(app, "_last_classification_delta", None)
        if isinstance(delta, SparseClassificationDelta) and delta.indices.size:
            source = delta
        else:
            source = getattr(app, "_last_changed_indices", None)
            if source is None:
                source = getattr(app, "_last_changed_mask", None)

    if isinstance(source, SparseClassificationDelta):
        idx = source.indices
    else:
        arr = np.asarray(source) if source is not None else np.empty(0, dtype=np.int64)
        if arr.dtype == bool:
            # Prefer the already-known sparse cache when this exact legacy mask
            # belongs to the current transaction. This avoids an N-point scan.
            cached = getattr(app, "_last_changed_indices", None) if app is not None else None
            cached_mask = getattr(app, "_last_changed_mask", None) if app is not None else None
            if source is cached_mask and isinstance(cached, np.ndarray):
                idx = np.asarray(cached, dtype=np.int64).ravel()
            else:
                idx = np.flatnonzero(arr).astype(np.int64, copy=False)
        else:
            idx = np.asarray(arr, dtype=np.int64).ravel()

    if idx.size == 0:
        return np.empty(0, dtype=np.int64)
    idx = idx[idx >= 0]
    if n is not None:
        idx = idx[idx < n]
    return idx.astype(np.int64, copy=False)


def make_delta(
    app_or_indices,
    indices=None,
    *,
    old_classes=None,
    new_classes=None,
    dataset_size: Optional[int] = None,
    operation: str = "classification",
    origin_view: str = "unknown",
    tx_id: Optional[int] = None,
    edit_bbox=None,
) -> SparseClassificationDelta:
    """Create a sparse delta using either supported call convention.

    New code may call ``make_delta(indices, dataset_size=...)`` while existing
    Phase-1 integrations may call ``make_delta(app, indices, ...)``.  Supporting
    both keeps the refactor non-breaking while call sites migrate.
    """
    if indices is None:
        app = None
        idx = app_or_indices
    else:
        app = app_or_indices
        idx = indices

    n = int(dataset_size) if dataset_size is not None else _classification_size(app)
    if tx_id is None:
        tx_id = int(getattr(app, "_classification_tx_id", 0) or 0) if app is not None else 0

    return SparseClassificationDelta(
        indices=np.asarray(idx, dtype=np.int64),
        old_classes=old_classes,
        new_classes=new_classes,
        dataset_size=n,
        operation=str(operation or "classification"),
        origin_view=str(origin_view or "unknown"),
        tx_id=int(tx_id or 0),
        edit_bbox=edit_bbox,
    )


def publish_delta(app, delta: SparseClassificationDelta) -> SparseClassificationDelta:
    """Publish sparse transaction state without allocating an N-point mask."""
    app._last_classification_delta = delta
    app._last_changed_indices = delta.indices.copy()
    app._last_changed_mask = None
    return delta


def materialize_changed_mask(
    app,
    change=None,
    *,
    changed_indices=None,
    reason: str = "legacy",
) -> np.ndarray:
    """Compatibility adapter for legacy code paths that still require bool[N]."""
    n = _classification_size(app)
    if n is None:
        return np.empty(0, dtype=bool)
    if isinstance(change, SparseClassificationDelta):
        return change.as_mask(n, reason=reason)
    idx = resolve_changed_indices(
        change, changed_indices=changed_indices, app=app, n_points=n
    )
    delta = SparseClassificationDelta(idx, dataset_size=n)
    return delta.as_mask(n, reason=reason)


def delta_from_payload(app, payload, *, operation: str = "classification", origin_view: str = "unknown"):
    """Normalize Signal(object) payloads while preserving legacy bool masks."""
    if isinstance(payload, SparseClassificationDelta):
        return payload
    idx = resolve_changed_indices(payload, app=app)
    return make_delta(app, idx, operation=operation, origin_view=origin_view)
