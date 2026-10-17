"""Share native SNT text geometry across repeated, equally styled labels."""
import weakref

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def _group_key(entity):
    """Hashable instancing key without ``repr()`` formatting.

    The previous implementation called ``repr()`` on 11 fields per entity;
    on files with hundreds of thousands of labels that string formatting
    alone dominated the "grouping labels" stage. Tuples of the raw values
    compare identically for grouping purposes. Mutable colour sequences are
    normalised to tuples so the key stays hashable.
    """
    color = entity.get('color')
    if isinstance(color, (list, tuple)):
        try:
            color = tuple(color)
        except Exception:
            color = None
    return (
        entity.get('text'),
        entity.get('layer'),
        color,
        entity.get('height'),
        entity.get('font_size'),
        entity.get('rotation'),
        entity.get('type'),
        entity.get('digitized'),
        entity.get('_lineweight'),
        entity.get('width_factor'),
        entity.get('alignment'),
    )


def text_instance_groups(entities, minimum=16):
    groups = {}
    for entity in entities:
        if entity.get('snt_binary'):
            try:
                key = _group_key(entity)
                hash(key)
            except Exception:
                key = ('individual', id(entity))
        else:
            key = ('individual', id(entity))
        groups.setdefault(key, []).append(entity)
    result = []
    for group in groups.values():
        if len(group) >= minimum and group[0].get('snt_binary'):
            result.append(group)
        else:
            result.extend([entity] for entity in group)
    return result


class SNTInstancedTextActor(vtk.vtkActor):
    """One GPU instanced draw in 2D; retain follower behavior in perspective."""
    def __init__(self, template, entities, renderer):
        super().__init__()
        self._template = template
        self._renderer = renderer
        self._fallback = []
        self._visible = int(template.GetVisibility())
        self._camera_signature = None
        self._observer = None
        self._camera = renderer.GetActiveCamera()
        template.SetPosition(0., 0., 0.)
        self._transform = vtk.vtkTransform()
        self._source = vtk.vtkTransformPolyDataFilter()
        self._source.SetInputData(template.GetMapper().GetInput())
        self._source.SetTransform(self._transform)
        self._instances = vtk.vtkPolyData()
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(self._instances)
        mapper.SetSourceConnection(self._source.GetOutputPort())
        mapper.ScalingOff()
        mapper.OrientOff()
        mapper.ScalarVisibilityOff()
        self.SetMapper(mapper)
        self.GetProperty().DeepCopy(template.GetProperty())
        for key in ('text_content', 'is_grid_label', 'grid_name', '_naksha_base_scale',
                    '_naksha_base_font_size', '_naksha_label_font_size'):
            if hasattr(template, key):
                setattr(self, key, getattr(template, key))
        self._naksha_snt_group_kind = 'text_instances'
        self.update_entities(entities)
        reference = weakref.ref(self)
        def camera_changed(camera, event):
            actor = reference()
            if actor is not None:
                actor._update_camera()
        self._observer = self._camera.AddObserver('ModifiedEvent', camera_changed)
        self._update_camera()

    def update_entities(self, entities):
        self._naksha_snt_entities = tuple(entities)
        values = np.asarray([tuple(e['position'][:3]) if len(e['position']) >= 3
                             else (*e['position'][:2], 0.) for e in entities], dtype=np.float64)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(values, deep=True))
        self._instances.SetPoints(points)
        self._instances.Modified()
        self._camera_signature = None
        self._update_camera()

    def _update_camera(self):
        camera = self._camera
        signature = (bool(camera.GetParallelProjection()), tuple(camera.GetDirectionOfProjection()),
                     tuple(camera.GetViewUp()))
        if signature == self._camera_signature:
            return
        self._camera_signature = signature
        for actor in self._fallback:
            self._renderer.RemoveActor(actor)
        self._fallback = []
        if camera.GetParallelProjection():
            self._transform.SetMatrix(self._template.GetMatrix())
            vtk.vtkActor.SetVisibility(self, self._visible)
        else:
            # Follower orientation in perspective depends on each label's
            # position. Preserve it rather than rotating all instance anchors.
            vtk.vtkActor.SetVisibility(self, 0)
            offset = vtk.vtkActor.GetPosition(self)
            for entity in self._naksha_snt_entities:
                actor = vtk.vtkFollower()
                actor.SetMapper(self._template.GetMapper())
                actor.SetCamera(camera)
                actor.SetProperty(self.GetProperty())
                actor.SetScale(self._template.GetScale())
                actor.SetOrigin(self._template.GetOrigin())
                actor.SetOrientation(self._template.GetOrientation())
                position = entity['position']
                actor.SetPosition(*(float(position[i]) + offset[i] if i < len(position)
                                    else offset[i] for i in range(3)))
                actor.SetVisibility(self._visible)
                self._renderer.AddActor(actor)
                self._fallback.append(actor)

    def SetScale(self, *value):
        self._template.SetScale(*value)
        self._camera_signature = None
        self._update_camera()

    def GetScale(self):
        return self._template.GetScale()

    def SetOrientation(self, *value):
        self._template.SetOrientation(*value)
        self._camera_signature = None
        self._update_camera()

    def GetOrientation(self):
        return self._template.GetOrientation()

    def SetOrigin(self, *value):
        self._template.SetOrigin(*value)
        self._camera_signature = None
        self._update_camera()

    def GetOrigin(self):
        return self._template.GetOrigin()

    def SetPosition(self, *value):
        vtk.vtkActor.SetPosition(self, *value)
        self._camera_signature = None
        self._update_camera()

    def SetVisibility(self, value):
        self._visible = int(value)
        vtk.vtkActor.SetVisibility(self, self._visible if not self._fallback else 0)
        for actor in self._fallback:
            actor.SetVisibility(self._visible)

    def GetVisibility(self):
        return self._visible

    def dispose(self):
        if self._observer is not None:
            self._camera.RemoveObserver(self._observer)
            self._observer = None
        for actor in self._fallback:
            self._renderer.RemoveActor(actor)
        self._fallback = []
