"""DXF/DWG/SNT/PRJ linework on the native Vulkan main viewport.

The LNT renderer owns the point-cloud surface. Vector attachments use the same
transparent, camera-following Qt overlay as the native drawing tools, avoiding
VTK actors and renderers while retaining layer visibility controls.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gui.native_main_classify import ensure_classifier
from gui.native_section_classify import CanvasOverlay


def is_native_main(app):
    view = getattr(app, '_lnt_main_view', None)
    return view is not None and getattr(view, 'native', None) is not None


class NativeAttachmentHandle:
    """Small VTK-actor compatibility surface used by existing layer widgets."""
    def __init__(self, layer, document, layer_name, bounds=None):
        self._layer = layer
        self._document = document
        self._layer_name = str(layer_name or '0')
        self._visible = True
        self._bounds = bounds

    def SetVisibility(self, visible):
        self._visible = bool(visible)
        self._layer.refresh()

    def GetVisibility(self):
        return int(self._visible)

    def GetBounds(self):
        if self._bounds is None:
            return None
        x0, x1, y0, y1 = self._bounds[:4]
        z0 = self._bounds[4] if len(self._bounds) > 4 else 0.0
        z1 = self._bounds[5] if len(self._bounds) > 5 else z0
        return x0, x1, y0, y1, z0, z1

    def Modified(self):
        self._layer.refresh()


class NativeAttachmentLayer:
    def __init__(self, app, view=None):
        self.app = app
        # An .lnt session supplies its own view; an SNT-only session (§5) passes
        # the promoted-viewport view instead. Either way this raises rather than
        # crashing later with AttributeError on None.
        view = view if view is not None else getattr(app, '_lnt_main_view', None)
        if view is None:
            raise RuntimeError('Native Vulkan main viewport is unavailable')
        self.view = view
        classifier = ensure_classifier(view)
        if classifier is None:
            raise RuntimeError('Native Vulkan main viewport is unavailable')
        self.overlay = CanvasOverlay(classifier.canvas)
        self.documents = {}

    def put(self, kind, name, grouped, bounds=None):
        key = (str(kind).lower(), str(Path(name)))
        handles = {layer: NativeAttachmentHandle(self, key, layer, bounds)
                   for layer in grouped}
        self.documents[key] = {'groups': grouped, 'handles': handles}
        self.refresh()
        return handles

    def remove(self, kind, name):
        self.documents.pop((str(kind).lower(), str(Path(name))), None)
        self.refresh()

    def remove_file(self, kind, name):
        wanted = Path(str(name)).name.lower()
        kind = str(kind).lower()
        for key in list(self.documents):
            if key[0] == kind and Path(key[1]).name.lower() == wanted:
                self.documents.pop(key, None)
        self.refresh()

    def refresh(self):
        shapes = []
        for document in self.documents.values():
            for layer_name, records in document['groups'].items():
                handle = document['handles'].get(layer_name)
                if handle is None or not handle._visible:
                    continue
                for points, data in records:
                    if points:
                        shapes.append(('drawing', points, data))
        self.overlay.set_shapes(shapes)

    def shutdown(self):
        self.documents.clear()
        self.overlay._timer.stop()
        self.overlay.close()


def layer_for(app):
    if not is_native_main(app):
        return None
    layer = getattr(app, '_native_attachment_layer', None)
    view = app._lnt_main_view
    if layer is not None and layer.view is not view:
        saved = [(key, value['groups']) for key, value in layer.documents.items()]
        layer.shutdown()
        layer = app._native_attachment_layer = NativeAttachmentLayer(app)
        for (kind, name), grouped in saved:
            layer.put(kind, name, grouped)
    elif layer is None:
        layer = app._native_attachment_layer = NativeAttachmentLayer(app)
    return layer


def _color(value):
    values = tuple(value or (255, 255, 255))
    if values and max(values) <= 1.0:
        values = tuple(round(float(v) * 255) for v in values)
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(v))) for v in values[:3])


def _xy(points):
    result = []
    for point in points or ():
        try:
            result.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            pass
    return result


def _entity_record(entity):
    kind = str(entity.get('type', '')).lower()
    points = []
    dtype = 'polyline'
    if kind == 'line':
        points = _xy(entity.get('points') or [entity.get('start'), entity.get('end')])
    elif kind in ('polyline', 'lwpolyline', 'polygon'):
        points = _xy(entity.get('points') or entity.get('vertices'))
        if entity.get('closed') and len(points) > 2 and points[0] != points[-1]:
            points.append(points[0])
    elif kind == '3dface':
        points = _xy(entity.get('vertices'))
        if len(points) > 2:
            points.append(points[0])
    elif kind in ('circle', 'arc'):
        centre = entity.get('center', (0, 0, 0))
        radius = float(entity.get('radius', 0.) or 0.)
        if radius > 0:
            start = float(entity.get('start_angle', 0.) or 0.)
            end = float(entity.get('end_angle', 360.) or 360.)
            if end <= start:
                end += 360.
            angles = np.linspace(math.radians(start), math.radians(end), 65)
            points = [(float(centre[0]) + radius * math.cos(a),
                       float(centre[1]) + radius * math.sin(a)) for a in angles]
    elif kind == 'point':
        points = _xy([entity.get('position') or entity.get('point')])
        dtype = 'point'
    elif kind == 'text':
        points = _xy([entity.get('position')])
        dtype = 'text'
    if not points:
        return None
    return points, {'type': dtype, 'source': 'attachment',
                    'text': str(entity.get('text', '')),
                    'rotation': float(entity.get('rotation', 0.) or 0.),
                    'height': float(entity.get('height', 0.) or 0.),
                    'color': _color(entity.get('color')), 'width': 1}


def render_entities(app, kind, attachment, entities=None):
    layer = layer_for(app)
    if layer is None:
        return None
    grouped = {}
    for entity in entities if entities is not None else attachment.get('entities', []):
        record = _entity_record(entity)
        if record is not None:
            grouped.setdefault(str(entity.get('layer', '0')), []).append(record)
    name = attachment.get('full_path') or attachment.get('filename') or kind
    handles = layer.put(kind, name, grouped, attachment.get('fit_bounds') or attachment.get('bounds'))
    return {name: [handle] for name, handle in handles.items()}


def render_dwg_geometry(app, name, layer_geometry):
    layer = layer_for(app)
    if layer is None:
        return None
    grouped = {}
    for layer_name, geometries in layer_geometry.items():
        records = grouped.setdefault(str(layer_name), [])
        for geom in geometries:
            pts = np.asarray(geom.get('pts', []))
            color = _color(geom.get('color'))
            if geom.get('type') == 'lines':
                for a, b in np.asarray(geom.get('segs', []), dtype=np.int64):
                    if 0 <= a < len(pts) and 0 <= b < len(pts):
                        records.append((_xy([pts[a], pts[b]]),
                                        {'type': 'polyline', 'source': 'attachment',
                                         'color': color, 'width': 1}))
            else:
                for point in pts:
                    records.append((_xy([point]), {'type': 'point',
                        'source': 'attachment', 'color': color, 'width': 1}))
    handles = layer.put('dwg', name, grouped)
    return {layer_name: [handle] for layer_name, handle in handles.items()}
