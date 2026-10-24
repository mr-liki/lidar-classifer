"""Compare real VTK and Vulkan pixels for both TIN windings, including steep faces.

Run with venv/Scripts/python.exe audit/lnt_steep_shading_parity.py.
The native LNT triangulator is clockwise; the VTK Triangle path is CCW.
"""
import ctypes as C
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PySide6.QtWidgets import QApplication, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui import shading_display as shading
from gui.render_backend import VulkanRenderBackend


class Tile(C.Structure):
    _fields_ = [('key', C.c_uint64), ('positions', C.c_void_p), ('cls', C.c_void_p),
                ('faces', C.c_void_p), ('coverage', C.c_void_p), ('coverage_cls', C.c_void_p),
                ('vertices', C.c_uint32), ('triangles', C.c_uint32), ('points', C.c_uint32),
                ('result', C.c_int32)]


class Draw(C.Structure):
    _fields_ = [('key', C.c_uint64), ('first', C.c_uint32), ('count', C.c_uint32),
                ('coverage', C.c_uint32), ('flags', C.c_uint32)]


def upload_tile(backend, xyz, faces, classes, palette):
    dll, handle = backend._dll, C.c_uint64(backend._handle)
    dll.nkv_surface_tiles_configure.argtypes = [C.c_uint64] * 4
    dll.nkv_surface_tiles_upload.argtypes = [C.c_uint64, C.POINTER(Tile), C.c_uint32]
    dll.nkv_surface_tiles_set_draw_list.argtypes = [C.c_uint64, C.POINTER(Draw), C.c_uint32]
    dll.nkv_surface_tiles_set_palette.argtypes = [C.c_uint64, C.c_void_p, C.c_float]
    ptr = lambda a: C.c_void_p(a.ctypes.data)
    assert dll.nkv_surface_tiles_configure(handle, 1024, 1024, 64)
    assert dll.nkv_surface_tiles_set_palette(handle, ptr(palette), C.c_float(1.))
    tile = Tile(1, ptr(xyz), ptr(classes), ptr(faces), None, None, 3, 1, 0, 0)
    assert dll.nkv_surface_tiles_upload(handle, C.byref(tile), 1) == 1
    assert tile.result == 1
    draw = Draw(1, 0, 1, 0, 0)
    assert dll.nkv_surface_tiles_set_draw_list(handle, C.byref(draw), 1)


def vtk_frame(xyz, colors, mixed):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(3)
    for i in range(3):
        cells.InsertCellPoint(i)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    normal = shading._compute_face_normals(xyz, np.array([[0, 1, 2]]))
    shading._attach_explicit_cell_normals(poly, normal)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToDirectScalars()
    if mixed:
        poly.GetPointData().SetScalars(numpy_to_vtk(colors, deep=True))
        mapper.SetScalarModeToUsePointData()
    else:
        light = np.array([.5, .5, np.sqrt(.5)])
        shade = max(float(normal[0] @ light), .25)
        rgb = np.floor(colors[:1].astype(float) * shade).astype(np.uint8)
        poly.GetCellData().SetScalars(numpy_to_vtk(rgb, deep=True))
        mapper.SetScalarModeToUseCellData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    app = SimpleNamespace(vtk_widget=SimpleNamespace(renderer=renderer),
                          last_shade_azimuth=45., shading_sharpness_angle=45.,
                          shade_ambient=.25)
    if mixed:
        shading._configure_microstation_color_blend_lighting(app, actor)
    else:
        actor.GetProperty().LightingOff()
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 500)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(1.25)
    camera.SetClippingRange(.001, 1000)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(400, 400)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.ReadFrontBufferOff()
    capture.Update()
    rgb = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
    result = np.flipud(rgb.reshape(400, 400, -1)[:, :, :3]).copy()
    window.Finalize()
    return result


def main():
    qt = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.resize(400, 400)
    widget.show()
    qt.processEvents()
    backend = VulkanRenderBackend()
    assert backend.initialize(int(widget.winId()), 400, 400)
    palette = np.zeros((256, 3), np.uint8)
    palette[2], palette[5], palette[6] = (255, 0, 0), (0, 255, 0), (128, 128, 128)
    app = SimpleNamespace()
    key, fill = shading._shading_key_fill_intensities(app, .5, 0.)
    # Interior pixels only: avoid raster edge coverage and quantization ties.
    yy, xx = np.indices((400, 400))
    bx, by = ((xx + .5) / 400 * 2.5 - .25) / 2, ((399 - yy + .5) / 400 * 2.5 - .25) / 2
    mask = (bx > .08) & (by > .08) & (bx + by < .92)
    failures = []
    try:
        for height in (.4, 20.):
            xyz = np.array([[-1., -1., 0.], [1., -1., height], [-1., 1., .1]])
            for mixed in (False, True):
                classes = np.array([2, 5, 6] if mixed else [2, 2, 2], np.uint8)
                reference = vtk_frame(xyz, palette[classes], mixed)
                for winding in ('ccw', 'cw'):
                    faces = np.array([[0, 1, 2] if winding == 'ccw' else [0, 2, 1]], np.int32)
                    assert backend.set_shaded_class_surface_indexed(xyz, faces, classes, palette)
                    backend.set_color_parity_params(1, 0, .08, 45.)
                    backend.set_crisp_shading_parameters(45., 45., .25, key, fill)
                    backend.set_camera_view_up(0., 1., 0.)
                    backend.set_camera_ortho((0., 0., 0.), (0., 0., -1.), 1.25, .001, 1000.)
                    backend.set_point_cloud_visible(False)
                    for path in ('compact', 'tile'):
                        if path == 'tile':
                            upload_tile(backend, xyz, faces, classes, palette)
                        for _ in range(3):
                            backend.request_render()
                            qt.processEvents()
                        candidate = backend.capture_frame()[:, :, :3]
                        error = np.abs(candidate[mask].astype(int) - reference[mask].astype(int))
                        maximum, mean = int(error.max()), float(error.mean())
                        label = f'height={height} mixed={mixed} winding={winding} path={path}'
                        print(f'{label}: mean={mean:.3f} max={maximum}', flush=True)
                        if maximum > 2 or mean > 1:
                            failures.append(label)
    finally:
        backend.shutdown()
        widget.close()
    assert not failures, 'VTK pixel mismatch: ' + ', '.join(failures)
    print('PASS: real VTK/Vulkan pixels match for gentle and steep faces in both windings')


if __name__ == '__main__':
    main()
