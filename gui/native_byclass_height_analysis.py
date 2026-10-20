"""Read-only input bridge for By Class > Height > Analyze Heights.

The VTK dialog analyzes app.data arrays; native Vulkan/LNT main views
intentionally do not populate those arrays.  Acquire an EXACT, fully audited
native snapshot instead, without invoking the class writer.
"""
from __future__ import annotations

import numpy as np


def height_analysis_points(app):
    """Return ``(xyz, classifications, is_native)`` for height analysis.

    Native records are read through the existing exact-record safety checks.
    An incomplete/invalid query raises rather than yielding partial results.
    No classes are written. In a VTK session, original arrays are returned.
    """
    from gui.native_by_class import want_native

    is_native = bool(want_native(app))
    if is_native:
        from gui.native_byclass_parity import read_snapshot
        _view, snapshot = read_snapshot(app)
        xyz = snapshot.get('xyz')
        classes = snapshot.get('classification')
    else:
        data = getattr(app, 'data', None)
        xyz = data.get('xyz') if isinstance(data, dict) else None
        classes = data.get('classification') if isinstance(data, dict) else None

    if xyz is None or classes is None:
        raise ValueError(
            'No point coordinates and classifications are loaded for Height analysis. '
            'Load a LAS/LNT cloud first.'
        )
    xyz = np.asarray(xyz)
    classes = np.asarray(classes)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or classes.ndim != 1:
        raise ValueError('Height analysis requires an N x 3 XYZ array and N class IDs.')
    if len(xyz) != len(classes):
        raise ValueError(
            f'Height analysis received {len(xyz):,} coordinates and '
            f'{len(classes):,} classifications; refusing incomplete data.'
        )
    if not np.isfinite(xyz).all():
        raise ValueError('Height analysis input has nonfinite coordinates.')
    return xyz, classes, is_native
