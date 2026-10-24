"""Run through the packaged executable's --run-python-script command."""
import ctypes
import faulthandler
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    faulthandler.dump_traceback_later(45, repeat=True)
    assert getattr(sys, 'frozen', False), 'Run this with the packaged executable'
    root = Path(sys._MEIPASS)
    assert os.environ.get('NAKSHA_RENDER_BACKEND') == 'vulkan'
    assert os.environ.get('NAKSHA_VULKAN_MAIN_VIEWPORT') == '1'
    modules = ['numpy', 'scipy', 'PySide6', 'vtk', 'laspy', 'lazrs', 'numba',
               'triangle', 'snt_core', 'CSF', 'torch', 'open3d', 'pyproj',
               'rasterio', 'jakteristics', 'zstandard']
    imported = {}
    for name in modules:
        module = importlib.import_module(name)
        location = str(getattr(module, '__file__', ''))
        assert not location or str(root).lower() in location.lower(), (name, location)
        imported[name] = location
        print('BUNDLED', name, location, flush=True)
    from gui import lnt_native
    assert lnt_native.lib() is not None
    ctypes.CDLL(str(root / 'native/libzstd.dll'))
    ctypes.CDLL(str(root / 'native/naksha_vulkan/build_msvc/vulkan-1.dll'))
    from PySide6.QtWidgets import QApplication, QWidget
    from gui.render_backend import VulkanRenderBackend
    import numpy as np
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.resize(320, 240)
    window.show()
    app.processEvents()
    backend = VulkanRenderBackend()
    assert backend.initialize(int(window.winId()), 320, 240)
    try:
        # Clockwise steep triangle: exercise the LNT shading correction too.
        xyz = np.array([[-1., -1., 0.], [1., -1., 20.], [-1., 1., .1]])
        faces = np.array([[0, 2, 1]], np.int32)
        classes = np.array([2, 5, 6], np.uint8)
        palette = np.zeros((256, 3), np.uint8)
        palette[2], palette[5], palette[6] = (255, 0, 0), (0, 255, 0), (128, 128, 128)
        assert backend.set_shaded_class_surface_indexed(xyz, faces, classes, palette)
        assert backend.set_crisp_shading_parameters(45., 45., .25, .80325, .24675)
        backend.set_camera_view_up(0., 1., 0.)
        backend.set_camera_ortho((0., 0., 0.), (0., 0., -1.), 1.25, .001, 1000.)
        backend.set_point_cloud_visible(False)
        for _ in range(3):
            backend.request_render()
            app.processEvents()
        image = backend.capture_frame()
        assert image is not None and np.count_nonzero(image[:, :, :3]) > 1000
        result = dict(frozen=True, bundle=str(root), imports=imported,
                      vulkan_render=True, frame_shape=list(image.shape),
                      lnt_dll=lnt_native.LNT_DLL)
    finally:
        backend.shutdown()
        window.close()
    from gui.app_window import NakshaApp
    from PySide6.QtWidgets import QMessageBox
    def startup_error(*args, **kwargs):
        raise RuntimeError('Packaged startup dialog: ' + str(args[2] if len(args) > 2 else args))
    QMessageBox.warning = startup_error
    QMessageBox.critical = startup_error
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.Yes
    gui = NakshaApp()
    gui.show()
    app.processEvents()
    assert gui.render_backend.active and gui.render_backend.vulkan_backend is not None
    result['gui_startup'] = True
    result['gpu_name'] = gui.render_backend.gpu_name
    gui.close()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS OFFLINE PACKAGE: bundled dependencies, native LNT and Vulkan rendering', flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)
