"""Resolve shading context clicks against the currently loaded file identities."""
import ntpath
import struct


def show_shading_at_world(app, world_xy):
    """Fallback for standalone clouds without SNT/PRJ block polygons."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None or view.shading_name not in ("shaded", "shaded_class", "surface"):
        return False
    path = getattr(view, "lnt_path", None)
    if not path:
        return False
    cached = getattr(view, "_shading_context_bounds", None)
    if cached is None or cached[0] != path:
        try:
            with open(path, "rb") as stream:
                header = stream.read(64)
            if header[:4] != b"LNT\0":
                return False
            low = struct.unpack_from("<3d", header, 16)
            high = struct.unpack_from("<3d", header, 40)
            cached = (path, low, high)
            view._shading_context_bounds = cached
        except (OSError, struct.error):
            return False
    _, low, high = cached
    x, y = world_xy
    return bool(low[0] <= x <= high[0] and low[1] <= y <= high[1]
                and app.show_shading_controls())


def show_loaded_block_shading(app, block):
    view = getattr(app, "_lnt_main_view", None)
    if view is None or view.shading_name not in ("shaded", "shaded_class", "surface"):
        return False
    sources = list(getattr(app, "_lnt_source_paths", ()) or ())
    source = getattr(app, "_lnt_source_path", None)
    if source:
        sources.append(source)

    def identity(value):
        return ntpath.splitext(ntpath.basename(str(value or "")))[0].casefold()

    loaded = {identity(path) for path in sources if path}
    # Use only this winning polygon's names, never aliases collected from
    # overlapping neighbouring polygons by the block-loading menu.
    names = [block.get("file_path"), block.get("grid_name"), block.get("block_file")]
    names.extend(block.get("alt_names") or ())
    if not any(identity(name) in loaded for name in names if name):
        return False
    return bool(app.show_shading_controls())
