"""Exercise the native .lnt viewer's Shaded -> mesh -> points path."""
import ctypes
import os
import sys
import time

from gui import lnt_native as native


def main(path):
    dll = native.lib()
    dll.lnt_view_surface_triangles.argtypes = [ctypes.c_void_p]
    dll.lnt_view_surface_triangles.restype = ctypes.c_uint64
    error = ctypes.create_string_buffer(512)
    handle = dll.lnt_view_open(None, 640, 480, os.fsencode(path),
                               os.fsencode(native.VULKAN_DLL), 512, 8.0, error, 512)
    if not handle:
        raise RuntimeError(error.value.decode(errors="replace"))
    try:
        stats = native.LntViewStats()
        assert dll.lnt_view_frame(handle, 0.016, ctypes.byref(stats))
        shade = native.ShadingUniforms(
            native.SHADE_NORMAL, native.RAMP_GRAYSCALE, 0.0, 1.0, 0.0, 1.0,
            (ctypes.c_float * 3)(0.5, -0.5, -0.7071), 0.0)
        assert dll.lnt_view_set_shading(handle, ctypes.byref(shade))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert dll.lnt_view_frame(handle, 0.016, ctypes.byref(stats))
            if dll.lnt_view_surface_triangles(handle):
                break
            time.sleep(0.03)
        triangles = dll.lnt_view_surface_triangles(handle)
        assert triangles > 0, "Shaded mode did not upload a surface"
        shade.display_mode = native.SHADE_CLASSIFICATION
        assert dll.lnt_view_set_shading(handle, ctypes.byref(shade))
        assert dll.lnt_view_frame(handle, 0.016, ctypes.byref(stats))
        print(f"native Shaded surface: {triangles:,} triangles; return to points: OK")
    finally:
        dll.lnt_view_close(handle)


if __name__ == "__main__":
    main(sys.argv[1])
