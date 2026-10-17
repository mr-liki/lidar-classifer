"""Settings updates for existing native viewports; never constructs VTK state."""


def native_viewport_active(app):
    owner = getattr(app, 'render_backend', None)
    return bool(getattr(owner, '_viewport_installed', False) or
                getattr(getattr(app, '_lnt_main_view', None), 'native', None) is not None)


def apply_native_background(app, rgb):
    owner = getattr(app, 'render_backend', None)
    backend = getattr(owner, 'vulkan_backend', None)
    if backend is not None:
        backend.set_clear_color(*rgb)
        backend.request_render()
    sections = getattr(app, '_native_sections', None)
    for canvas in getattr(sections, 'views', {}).values():
        canvas.backend.set_clear_color(*rgb)
        canvas._dirty = True


def refresh_native_overlays(app):
    view = getattr(app, '_lnt_main_view', None) or getattr(app, '_snt_viewport_view', None)
    drawing = getattr(view, 'drawing', None)
    if drawing is not None:
        drawing.reload_styles()
    overlay = getattr(view, 'section_overlay', None)
    if overlay is not None:
        overlay.update()
