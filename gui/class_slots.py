"""User class numbers <-> stored one-byte class slots (native .lnt path).

Native point records keep one byte per class (LAS / LNT: 0..255), so at most
256 different classes can be stored in one file. Users may still number their
own classes anything (e.g. 999 in a PTC):

* a number 0..255 is stored as itself (LAS files open unchanged);
* a larger number is given a free slot - one that no class defined in the
  palette and no stored point uses - and the pairing is saved beside the file
  in ``<file>.lnt.classmap.json``, so reopening the file shows 999 again.

Everything that crosses into native data (classify, From/visibility filters,
palette/LUT uploads) goes through ``slot_for`` / ``stored_palette``; everything
that shows a stored class to the user goes through ``code_for``.
"""
import json
import os

SUFFIX = '.classmap.json'
SLOTS = 256


class NoFreeSlot(RuntimeError):
    """All 256 stored class slots are in use."""


def _lnt_path(app):
    view = getattr(app, '_lnt_main_view', None)
    path = getattr(view, 'lnt_path', None)
    return str(path) if path else None


def _state(app):
    """{'path', 'code_to_slot', 'slot_to_code'} for the open .lnt (loaded lazily)."""
    path = _lnt_path(app)
    state = getattr(app, '_class_slot_state', None)
    if state is not None and state['path'] == path:
        return state
    code_to_slot = {}
    if path:
        try:
            with open(path + SUFFIX, encoding='utf-8') as handle:
                raw = json.load(handle).get('code_to_slot', {})
            for code, slot in raw.items():
                code, slot = int(code), int(slot)
                if code > 255 and 0 <= slot < SLOTS:
                    code_to_slot[code] = slot
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    state = {'path': path, 'code_to_slot': code_to_slot,
             'slot_to_code': {slot: code for code, slot in code_to_slot.items()}}
    app._class_slot_state = state
    return state


def _save(state):
    if not state['path']:
        return
    data = {'version': 1,
            'note': 'User class numbers above 255 and the one-byte slot each is stored as.',
            'code_to_slot': {str(c): s for c, s in sorted(state['code_to_slot'].items())}}
    tmp = state['path'] + SUFFIX + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=1)
    os.replace(tmp, state['path'] + SUFFIX)


def _stored_classes(app):
    """Class slots that hold points in the open file (best effort)."""
    view = getattr(app, '_lnt_main_view', None)
    counter = getattr(view, 'class_counts', None)
    used = set()
    counts = None
    try:
        counts = counter.counts(getattr(app, '_lnt_section_store', None)) if counter else None
        used.update(int(c) for c, n in (counts or {}).items() if n)
    except Exception:   # noqa: BLE001 - counts are a hint only
        pass
    # A section is only a subset: never allocate against an incomplete histogram.
    path = _lnt_path(app)
    if counts is None and path and os.path.isfile(path):
        from gui.lnt_class_counts import scan_class_histogram
        hist = scan_class_histogram(path)
        used.update(int(c) for c, n in enumerate(hist) if n)
    store = getattr(app, '_lnt_section_store', None)
    try:
        import numpy as np
        cls = store.data['classification']
        if len(cls):
            used.update(int(c) for c in np.flatnonzero(np.bincount(cls, minlength=SLOTS)))
    except Exception:   # noqa: BLE001
        pass
    return used


def _defined_codes(app):
    codes = set()
    palettes = [getattr(app, 'class_palette', None) or {}]
    palettes += list((getattr(app, 'view_palettes', None) or {}).values())
    for palette in palettes:
        for code in (palette or {}):
            try:
                codes.add(int(code))
            except (TypeError, ValueError):
                pass
    return codes


def slot_for(app, code, allocate=False):
    """Stored slot for a user class number, or None (unknown / unmapped)."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    if 0 <= code <= 255:
        return code
    if code < 0:
        return None
    state = _state(app)
    slot = state['code_to_slot'].get(code)
    if slot is not None or not allocate:
        return slot
    taken = {c for c in _defined_codes(app) if 0 <= c <= 255}
    taken |= _stored_classes(app) | set(state['slot_to_code'])
    free = [s for s in range(SLOTS - 1, -1, -1) if s not in taken]
    if not free:
        raise NoFreeSlot(f'Class {code} cannot be stored: all 256 class slots are in use.')
    slot = free[0]
    state['code_to_slot'][code] = slot
    state['slot_to_code'][slot] = code
    try:
        _save(state)
    except OSError:
        del state['code_to_slot'][code]
        del state['slot_to_code'][slot]
        raise
    print(f'[class-slots] class {code} stored as slot {slot} ({state["path"]}{SUFFIX})')
    _refresh_palettes(app)
    return slot


def _refresh_palettes(app):
    """Re-upload the 256-entry class tables so a newly mapped slot shows its colour."""
    view = getattr(app, '_lnt_main_view', None)
    push = getattr(view, '_push_applied_palette', None)
    if callable(push) and getattr(app, 'class_palette', None):
        try:
            push()
        except Exception as exc:   # noqa: BLE001
            print(f'[class-slots] main palette refresh failed: {exc}')
    sections = getattr(app, '_native_sections', None)
    for canvas in getattr(sections, 'views', {}).values():
        try:
            canvas.apply_palette(canvas.palette, canvas.mode)
        except Exception as exc:   # noqa: BLE001
            print(f'[class-slots] section palette refresh failed: {exc}')


def slots_for(app, codes):
    """Slots of the given user numbers; numbers with no slot match no point."""
    out = []
    for code in codes or ():
        slot = slot_for(app, code)
        if slot is not None:
            out.append(slot)
    return out


def code_for(app, slot):
    """User class number shown for a stored slot."""
    slot = int(slot)
    return _state(app)['slot_to_code'].get(slot, slot)


def stored_palette(app, palette):
    """``palette`` (user numbers) keyed by stored slot, for the 256-entry LUTs.
    A mapped number takes its slot; numbers without a slot are not drawable yet."""
    out = {}
    mapped = []
    for code, rule in (palette or {}).items():
        try:
            code = int(code)
        except (TypeError, ValueError):
            continue
        if 0 <= code <= 255:
            out.setdefault(code, rule)
        else:
            slot = slot_for(app, code)
            if slot is not None:
                mapped.append((slot, rule))
    for slot, rule in mapped:      # a mapped class owns its slot
        out[slot] = rule
    return out


def user_counts(app, counts):
    """{slot: n} -> {user number: n}."""
    return {code_for(app, slot): n for slot, n in (counts or {}).items()}
