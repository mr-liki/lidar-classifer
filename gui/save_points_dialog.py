"""Pre-save options popup for Save As (TerraScan-style "Save points")."""

import numpy as np

from PySide6.QtCore import QEvent, QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from gui.dialogs.view_fields_dialog import (
    FIELD_SPECS,
    IMAGE_FIELD_LABEL,
    _read_dimension_names,
    find_image_dimension_name,
)
from gui.dialogs.view_fields_dialog import _SETTINGS_KEY as VIEW_FIELDS_SETTINGS_KEY

from .theme_manager import ThemeColors, get_dialog_stylesheet

# Every LAS/LAZ version the writer can produce, listed the way TerraScan's
# "Save points" window does: grouped by container type, oldest version first.
LAS_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4")
FORMAT_CHOICES = tuple(
    f"{container} {version}"
    for container in ("LAS", "LAZ")
    for version in LAS_VERSIONS
)
DEFAULT_FORMAT = "LAZ 1.4"

ATTRIBUTES_SETTINGS_KEY = "save_points/attributes_checked"


def _split_labels(raw):
    """Parse the attributes key: comma-joined labels ('' means none chosen).

    Stored as one string because QSettings collapses a single-element list
    back to a plain string, which would make a one-attribute choice
    indistinguishable from no choice at all. Attribute labels never contain
    commas.
    """
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def _saved_labels(value):
    """Normalise a saved View Fields list, which may arrive as one string."""
    if value is None:
        return None
    if isinstance(value, str):
        return {value} if value else None
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return None


def available_attribute_rows(filename=None, available_dimensions=None):
    """View Fields rows as (label, available) pairs for the loaded dataset.

    Mirrors ``ViewFieldsDialog._populate_fields``: same labels, same order and
    the same availability rule, so the two windows always show the same
    options. ``available_dimensions`` carries keys the app already holds (for
    example dimensions loaded from a PTC) that the LAS header alone cannot
    reveal.
    """
    dims = _read_dimension_names(filename)
    dims.update(str(name).casefold() for name in (available_dimensions or ()))
    image_dimension = find_image_dimension_name(dims)

    rows = []
    for label, required_dims, always_available in FIELD_SPECS:
        if label == IMAGE_FIELD_LABEL:
            available = image_dimension is not None
        else:
            available = always_available or any(
                d in dims for d in required_dims
            )
        rows.append((label, available))
    return rows


# Plural wording for the multi-selection summary: "2 lines: 0, 2".
_ITEM_PLURALS = {
    "class": "classes",
    "classes": "classes",
    "line": "lines",
    "lines": "lines",
}


def _swatch_icon(color, size=12):
    """Small filled rectangle drawn before a class name in the lists.

    Colors arrive as an (r, g, b) tuple from the Display Mode table; anything
    unusable simply yields no icon so a row never loses its text.
    """
    try:
        qcolor = QColor(*color) if isinstance(color, (tuple, list)) else QColor(color)
    except Exception:
        return None
    if not qcolor.isValid():
        return None

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(qcolor.darker(160), 1))
    painter.setBrush(qcolor)
    painter.drawRoundedRect(QRectF(0.5, 0.5, size - 1.5, size - 1.5), 2.5, 2.5)
    painter.end()
    return QIcon(pixmap)


def _plural_item_word(any_label):
    words = str(any_label or "").split()
    word = words[-1].lower() if words else "item"
    return _ITEM_PLURALS.get(word, word + "s")


class MultiClassCombo(QComboBox):
    """Real combo box (so it styles like the others) whose popup list allows Ctrl/Shift multi-selection.

    The ``any_label`` entry ("Any class", "All Lines", ...) is exclusive:
    picking it clears every real item, and picking any real item clears it.
    """

    def __init__(self, class_items=(), parent=None, any_label="Any class",
                 items_provider=None):
        super().__init__(parent)
        self._any_label = any_label
        self._items_word = _plural_item_word(any_label)
        # Deferred entries: called once, the first time the list is opened, so
        # building the dialog never walks the point data.
        self._items_provider = items_provider
        self.addItem(any_label)  # single display row; text set in _refresh_text
        self._popup = QFrame(self, Qt.Popup)
        self._popup.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self._popup)
        lay.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self._list)
        self._list.addItem(any_label)
        self._list.item(0).setData(Qt.UserRole, None)
        for entry in class_items or ():
            self._add_entry(entry)
        self._list.item(0).setSelected(True)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._refresh_text()
        if not class_items and items_provider is None:
            # No PTC / palette loaded: blank, inert box; everything is saved.
            self.setText("")
            self.setEnabled(False)

    def _add_entry(self, entry):
        """Append one (code, label[, color]) row."""
        code, label = int(entry[0]), str(entry[1])
        # Third element (optional) is the Display Mode color for this row.
        item = QListWidgetItem(label)
        icon = _swatch_icon(entry[2] if len(entry) > 2 else None)
        if icon is not None:
            item.setIcon(icon)
        item.setData(Qt.UserRole, code)
        self._list.addItem(item)

    def has_deferred_items(self):
        return self._items_provider is not None

    def _materialize_items(self):
        """Run the deferred provider once, when the list first opens."""
        provider, self._items_provider = self._items_provider, None
        if provider is None:
            return
        try:
            entries = list(provider() or [])
        except Exception as exc:
            print(f"Save points: could not list items: {exc}")
            entries = []
        for entry in entries:
            self._add_entry(entry)

    def setText(self, text):
        self.setItemText(0, text)

    def hidePopup(self):
        pass

    def showPopup(self):
        # Deferred entries arrive here, never while the dialog is being built.
        self._materialize_items()
        rows = min(self._list.count(), 10)
        self._popup.setFixedWidth(max(self.width(), 200))
        self._list.setFixedHeight(rows * self._list.sizeHintForRow(0) + 6)
        self._popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        self._popup.show()

    def _on_selection_changed(self):
        any_item = self._list.item(0)
        selected = self._list.selectedItems()
        others = [i for i in selected if i is not any_item]
        self._list.blockSignals(True)
        if others and any_item.isSelected():
            any_item.setSelected(False)  # picking a class drops "Any class"
        elif not others:
            any_item.setSelected(True)   # never leave nothing selected
        self._list.blockSignals(False)
        self._refresh_text()

    def selected_classes(self):
        """Sorted codes, or None when the exclusive "Any ..." entry is selected."""
        codes = [
            i.data(Qt.UserRole) for i in self._list.selectedItems()
            if i.data(Qt.UserRole) is not None
        ]
        return sorted(codes) or None

    def _refresh_text(self):
        if not self.isEnabled():
            return
        codes = self.selected_classes()
        if codes is None:
            self.setText(self._any_label)
            self.setItemIcon(0, QIcon())
        elif len(codes) == 1:
            item = self._list.selectedItems()[0]
            self.setText(item.text())
            # Show the chosen class's color chip in the closed field too.
            self.setItemIcon(0, item.icon())
        else:
            self.setText(
                f"{len(codes)} {self._items_word}: " + ", ".join(map(str, codes))
            )


class AttributesToSaveDialog(QDialog):
    """TerraScan-style "Attributes to save" checklist opened from Save points.

    The rows are exactly the View Fields rows - same labels, same order, same
    availability rule. Checked state follows the View Fields selection once one
    has been applied there; with no such choice yet every attribute the loaded
    file provides starts checked (TerraScan's "all on" default). Fields the
    loaded file cannot provide stay visible but disabled, exactly as in View
    Fields. On save, unchecked rows are cleared in the written file: the
    attributes you did not choose carry no values after the reload.
    """

    def __init__(self, parent=None, filename=None, available_dimensions=None,
                 settings=None):
        super().__init__(parent)
        self.setWindowTitle("Attributes to save")
        self.setModal(True)
        self.setMinimumSize(340, 320)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(get_dialog_stylesheet())

        # Passed in by tests so they never touch the user's real settings.
        self._settings = settings if settings is not None else QSettings(
            "NakshaAI", "LidarApp"
        )
        self._rows = available_attribute_rows(filename, available_dimensions)

        self.attributes_list = QListWidget()
        self.attributes_list.setObjectName("attributesList")
        self.attributes_list.setMinimumWidth(200)
        self.attributes_list.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        all_on_btn = QPushButton("All on")
        all_on_btn.setToolTip("Check every attribute the loaded file provides")
        all_off_btn = QPushButton("All off")
        all_off_btn.setToolTip("Uncheck every attribute")
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(all_on_btn)
        side.addWidget(all_off_btn)
        side.addStretch(1)

        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        cancel_btn = QPushButton("Cancel")
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.attributes_list, 1)
        body.addLayout(side)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(14)
        root.addLayout(body, 1)
        root.addLayout(buttons)

        all_on_btn.clicked.connect(self._select_all)
        all_off_btn.clicked.connect(self._clear_all)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)

        self._populate_attributes()

    def _initial_checked(self):
        """Labels checked in the View Fields window, or None when unknown."""
        if self._settings.contains(ATTRIBUTES_SETTINGS_KEY):
            return _split_labels(
                self._settings.value(ATTRIBUTES_SETTINGS_KEY, "")
            )
        return _saved_labels(self._settings.value(VIEW_FIELDS_SETTINGS_KEY))

    def _populate_attributes(self):
        checked = self._initial_checked()
        for label, available in self._rows:
            item = QListWidgetItem(label)
            item.setFlags(
                Qt.ItemIsUserCheckable
                | (Qt.ItemIsEnabled | Qt.ItemIsSelectable
                   if available else Qt.NoItemFlags)
            )
            # No View Fields choice yet: everything the file provides is on.
            if checked is None:
                is_checked = available
            else:
                is_checked = available and label in checked
            item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
            if not available:
                item.setToolTip("Not present in the loaded file")
            self.attributes_list.addItem(item)

    def _iter_available_items(self):
        """Skip disabled rows (attributes the loaded file does not provide)."""
        for i in range(self.attributes_list.count()):
            item = self.attributes_list.item(i)
            if item.flags() & Qt.ItemIsEnabled:
                yield item

    def _select_all(self):
        for item in self._iter_available_items():
            item.setCheckState(Qt.Checked)

    def _clear_all(self):
        for item in self._iter_available_items():
            item.setCheckState(Qt.Unchecked)

    def selected_attributes(self):
        """Checked attribute labels, in View Fields order."""
        return [
            item.text()
            for item in self._iter_available_items()
            if item.checkState() == Qt.Checked
        ]

    def _on_accept(self):
        self._settings.setValue(
            ATTRIBUTES_SETTINGS_KEY, ",".join(self.selected_attributes())
        )
        self.accept()


# Digitizer shape kinds the By Class fence picker offers (see
# menu_sidebar_system._show_fence_selection_dialog); the curve tool's finished
# curves are appended after them.
FENCE_SHAPE_TYPES = (
    "rectangle",
    "circle",
    "polygon",
    "freehand",
    "line",
    "smart_line",
    "polyline",
    "smartline",
)


def _fence_coords(fence, digitizer=None):
    """Fence vertices as an Nx2 float array, or None when unusable.

    Mirrors the fence picker's lookup: the digitizer's own getter first, then
    the plain coordinate keys. Fewer than three vertices cannot enclose points.
    """
    if not isinstance(fence, dict):
        return None
    raw = None
    getter = getattr(digitizer, "_get_drawing_coords", None)
    if callable(getter):
        try:
            raw = getter(fence)
        except Exception:
            raw = None
    if raw is None:
        for key in ("coords", "coordinates", "points", "interpolated"):
            if fence.get(key) is not None:
                raw = fence[key]
                break
    try:
        coords = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
        return None
    coords = coords[:, :2]
    if not np.all(np.isfinite(coords)):
        return None
    return coords


def _fence_ring(fence, digitizer=None):
    """Closed ring for a fence; open line-like fences get their ends joined."""
    coords = _fence_coords(fence, digitizer)
    if coords is None:
        return None
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    return coords


def _fence_size_suffix(fence, digitizer=None):
    """Extent shown after a fence name, the way the picker shows it."""
    coords = _fence_coords(fence, digitizer)
    if coords is None:
        return ""
    width = float(coords[:, 0].max() - coords[:, 0].min())
    height = float(coords[:, 1].max() - coords[:, 1].min())
    return f"  ({width:.1f} x {height:.1f} m)"


def collect_save_fences(app):
    """(label, fence) rows for the fence tool's By Class fence list.

    Same source list the picker shows - every digitizer drawing of a
    fence-capable type, then the curve tool's finished curves - and the same
    numbering, so both windows read alike. Fences without usable vertices are
    left out because no point could be tested against them.
    """
    digitizer = getattr(app, "digitizer", None)
    shapes = [
        drawing
        for drawing in (getattr(digitizer, "drawings", None) or [])
        if isinstance(drawing, dict)
        and drawing.get("type") in FENCE_SHAPE_TYPES
        # Cell circles belong to Create > Cell — never fences.
        and not digitizer.is_cell_creation_drawing(drawing)
    ]

    curves = []
    curve_tool = getattr(app, "curve_tool", None)
    if hasattr(curve_tool, "get_curves_as_fences"):
        try:
            curves = list(curve_tool.get_curves_as_fences() or [])
        except Exception as exc:
            print(f"Save points: could not list curve fences: {exc}")
            curves = []

    rows = []
    for index, fence in enumerate(shapes):
        if _fence_ring(fence, digitizer) is None:
            continue
        kind = str(fence.get("type", "fence")).replace("_", " ").title()
        rows.append(
            (f"#{index + 1}: {kind}{_fence_size_suffix(fence, digitizer)}", fence)
        )
    for offset, fence in enumerate(curves):
        if _fence_ring(fence, digitizer) is None:
            continue
        number = int(fence.get("curve_index", offset) or 0) + 1
        rows.append((f"Curve #{number}{_fence_size_suffix(fence, digitizer)}", fence))
    return rows


def _polygon_inside_mask(points_xy, ring):
    """Vectorised ray casting: which points fall inside the ring."""
    inside = np.zeros(len(points_xy), dtype=bool)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        inside ^= ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
        )
        x1, y1 = x2, y2
    return inside


def fence_inside_mask(xyz, fences, digitizer=None):
    """Union of point-in-fence masks for the chosen fences."""
    points = np.asarray(xyz)
    combined = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return combined
    points_xy = np.asarray(points[:, :2], dtype=float)
    for fence in fences or ():
        ring = _fence_ring(fence, digitizer)
        if ring is None:
            continue
        low = ring.min(axis=0)
        high = ring.max(axis=0)
        # Bounding-box pre-filter: exact test only on the candidates.
        box = (
            (points_xy[:, 0] >= low[0])
            & (points_xy[:, 0] <= high[0])
            & (points_xy[:, 1] >= low[1])
            & (points_xy[:, 1] <= high[1])
        )
        if not box.any():
            continue
        candidates = points_xy[box]
        try:
            from matplotlib.path import Path

            inside = Path(ring).contains_points(candidates)
        except Exception:
            inside = _polygon_inside_mask(candidates, ring)
        combined[box] |= np.asarray(inside, dtype=bool)
    return combined


class FenceToSaveDialog(QDialog):
    """TerraScan-style fence chooser behind "Inside fence only".

    Lists the fences the By Class fence tool's picker lists - same rows, same
    order, same numbering. Tick one or more and press OK: Save points then
    keeps only the points inside them, together with the Save class, Line,
    Attributes and Format choices from the main window.
    """

    def __init__(self, parent=None, fence_items=()):
        super().__init__(parent)
        self.setWindowTitle("Fence to save")
        self.setModal(True)
        self.setMinimumSize(380, 320)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(get_dialog_stylesheet())

        self._fences = [fence for _label, fence in fence_items]
        self._hover_actor = None

        self.fence_list = QListWidget()
        # Hovering a row outlines that fence on the canvas, like the By Class
        # fence picker does.
        self.fence_list.setMouseTracking(True)
        self.fence_list.viewport().setMouseTracking(True)
        self.fence_list.viewport().installEventFilter(self)
        self.fence_list.itemEntered.connect(self._on_item_hovered)
        self.finished.connect(lambda _result: self._clear_hover_highlight())
        self.fence_list.setObjectName("fenceList")
        self.fence_list.setMinimumWidth(220)
        self.fence_list.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        all_btn = QPushButton("All on")
        all_btn.setToolTip("Tick every fence drawn in this dataset")
        none_btn = QPushButton("All off")
        none_btn.setToolTip("Untick every fence")
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(all_btn)
        side.addWidget(none_btn)
        side.addStretch(1)

        self.status_label = QLabel("No fences selected")
        self.status_label.setObjectName("dialogCaption")

        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        cancel_btn = QPushButton("Cancel")
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.fence_list, 1)
        body.addLayout(side)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(14)
        root.addLayout(body, 1)
        root.addWidget(self.status_label)
        root.addLayout(buttons)

        all_btn.clicked.connect(lambda: self._set_all(Qt.Checked))
        none_btn.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        self.fence_list.itemChanged.connect(self._refresh_status)

        for label, _fence in fence_items:
            item = QListWidgetItem(str(label))
            item.setFlags(
                Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            )
            item.setCheckState(Qt.Unchecked)
            self.fence_list.addItem(item)
        self._refresh_status()

    # -- canvas hover highlight ------------------------------------------
    def _main_view(self):
        """The main window's VTK widget, found through the parent chain."""
        widget = self.parent()
        while widget is not None:
            vtk_widget = getattr(widget, "vtk_widget", None)
            if vtk_widget is not None and getattr(vtk_widget, "renderer", None) is not None:
                return widget, vtk_widget
            widget = widget.parent() if hasattr(widget, "parent") else None
        return None, None

    def _clear_hover_highlight(self):
        actor, self._hover_actor = self._hover_actor, None
        if actor is None:
            return
        _app, vtk_widget = self._main_view()
        if vtk_widget is None:
            return
        try:
            vtk_widget.renderer.RemoveViewProp(actor)
            vtk_widget.render()
        except Exception:
            pass

    def _on_item_hovered(self, item):
        self._clear_hover_highlight()
        row = self.fence_list.row(item)
        if not 0 <= row < len(self._fences):
            return
        app, vtk_widget = self._main_view()
        if vtk_widget is None:
            return
        ring = _fence_ring(self._fences[row], getattr(app, "digitizer", None))
        if ring is None:
            return
        try:
            import vtk

            points = vtk.vtkPoints()
            points.SetDataTypeToDouble()
            for x, y in ring:
                points.InsertNextPoint(float(x), float(y), 0.0)
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ring))
            for i in range(len(ring)):
                line.GetPointIds().SetId(i, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(line)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(cells)

            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            mapper.SetTransformCoordinate(coord)

            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            actor.GetProperty().SetLineWidth(6)
            actor.GetProperty().SetDisplayLocationToForeground()
            vtk_widget.renderer.AddViewProp(actor)
            self._hover_actor = actor
            vtk_widget.render()
        except Exception as exc:
            print(f"Fence to save: hover highlight failed: {exc}")

    def eventFilter(self, obj, event):
        if obj is self.fence_list.viewport():
            if event.type() == QEvent.Leave:
                self._clear_hover_highlight()
            elif (
                event.type() == QEvent.MouseMove
                and self.fence_list.itemAt(event.position().toPoint()) is None
            ):
                # Pointer moved onto the empty part of the list.
                self._clear_hover_highlight()
        return super().eventFilter(obj, event)

    def set_selected(self, fences):
        """Tick the fences chosen in an earlier visit to this dialog."""
        wanted = {id(fence) for fence in (fences or [])}
        self.fence_list.blockSignals(True)
        for row in range(self.fence_list.count()):
            state = Qt.Checked if id(self._fences[row]) in wanted else Qt.Unchecked
            self.fence_list.item(row).setCheckState(state)
        self.fence_list.blockSignals(False)
        self._refresh_status()

    def _set_all(self, state):
        self.fence_list.blockSignals(True)
        for row in range(self.fence_list.count()):
            self.fence_list.item(row).setCheckState(state)
        self.fence_list.blockSignals(False)
        self._refresh_status()

    def _checked_rows(self):
        return [
            row
            for row in range(self.fence_list.count())
            if self.fence_list.item(row).checkState() == Qt.Checked
        ]

    def selected_fences(self):
        """Fences ticked here, in the order the fence picker lists them."""
        return [self._fences[row] for row in self._checked_rows()]

    def selected_labels(self):
        return [self.fence_list.item(row).text() for row in self._checked_rows()]

    def _refresh_status(self):
        count = len(self._checked_rows())
        if count == 0:
            text = "No fences selected"
        elif count == 1:
            text = "1 fence selected"
        else:
            text = f"{count} fences selected"
        self.status_label.setText(text)


def _format_selection_qss():
    """Blue, low-opacity highlight for the Format dropdown's selected row.

    ``dialog_selection`` is a neutral grey that reads as hover rather than
    chosen; the Format row is the one choice worth seeing at a glance, so it
    uses the app accent at low opacity instead.
    """
    accent = QColor(ThemeColors.get("accent") or "#007acc")
    if not accent.isValid():
        accent = QColor("#007acc")
    highlight = f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, 0.45)"
    text = ThemeColors.get("text_on_active") or "#ffffff"
    return f"""
    QAbstractItemView#saveFormatView {{
        selection-background-color: {highlight};
        selection-color: {text};
    }}
    QAbstractItemView#saveFormatView::item:selected {{
        background-color: {highlight};
        color: {text};
    }}
    """


class SavePointsDialog(QDialog):
    """Options shown before the Save As file browser.

    Working options: Save class (any, or one or more classes), Line (the
    loaded data's flight lines), Format ("LAS 1.4", "LAZ 1.3", ...), Attributes
    and Inside fence only (the fence tool's By Class fence list). Points and
    Transform are shown for parity with the reference layout but stay disabled.
    """

    def __init__(self, parent, class_items, default_format="LAZ 1.4",
                 filename=None, available_dimensions=None, line_items=None,
                 line_provider=None, fence_items=None):
        super().__init__(parent)
        self._filename = filename
        self._available_dimensions = available_dimensions
        self._selected_attributes = None
        self._fence_items = list(fence_items or ())
        self._selected_fences = []
        self.setWindowTitle("Save points")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(get_dialog_stylesheet() + _format_selection_qss())

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.class_combo = MultiClassCombo(class_items)
        self.class_combo.setToolTip(
            "Ctrl+click or Shift+click to select several classes"
        )
        self.points_combo = QComboBox()
        self.points_combo.addItem("All points")
        # Line choices come straight from the loaded dataset's Point Source
        # IDs (recovered from the LAS/LAZ file when a legacy load path left
        # them out) - never Display Mode's computed list.
        self.line_combo = MultiClassCombo(
            line_items or [], any_label="All Lines", items_provider=line_provider
        )
        if line_items or line_provider is not None:
            self.line_combo.setToolTip(
                "Flight lines to save; Ctrl+click or Shift+click to select "
                "several. Read from the loaded LAS/LAZ data"
            )
        else:
            self.line_combo.setToolTip("No flight-line data in this dataset")
        self.format_combo = QComboBox()
        # The popup's selected row carries the app accent at low opacity
        # (see _format_selection_qss) so the current format reads at a
        # glance instead of looking merely hovered.
        self.format_combo.setObjectName("saveFormatCombo")
        self.format_combo.view().setObjectName("saveFormatView")
        for fmt in FORMAT_CHOICES:
            self.format_combo.addItem(fmt)
        idx = self.format_combo.findText(default_format)
        if idx < 0:
            idx = self.format_combo.findText(DEFAULT_FORMAT)
        self.format_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.transform_combo = QComboBox()
        self.transform_combo.addItem("None")
        self.fence_check = QCheckBox("Inside fence only")

        self.attributes_btn = QPushButton("Attributes...")
        self.attributes_btn.setToolTip(
            "Choose the point attributes to save (same list as View Fields). "
            "Unchecked attributes are cleared in the saved file"
        )
        self.attributes_btn.clicked.connect(self._open_attributes_dialog)
        for w in (self.points_combo, self.transform_combo):
            w.setEnabled(False)
            w.setToolTip("Not available for this dataset")

        # "Inside fence only" opens the fence tool's By Class fence list
        # in its own dialog; the choice is applied by the save filter.
        if self._fence_items:
            self.fence_check.setToolTip(
                "Keep only the points inside a fence drawn with the "
                "fence tool"
            )
            self.fence_check.clicked.connect(self._choose_fences)
        else:
            self.fence_check.setEnabled(False)
            self.fence_check.setToolTip("No fences drawn in this dataset")

        rows = (
            ("Save class:", self.class_combo),
            ("Points:", self.points_combo),
            ("Line:", self.line_combo),
            ("Format:", self.format_combo),
        )
        for r, (text, widget) in enumerate(rows):
            grid.addWidget(_label(text), r, 0, Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(widget, r, 1)
        grid.addWidget(self.attributes_btn, 3, 2)
        grid.addWidget(_label("Transform:"), 4, 0, Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self.transform_combo, 4, 1)
        grid.addWidget(self.fence_check, 5, 1)
        grid.setColumnStretch(1, 1)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(14)
        root.addLayout(grid)
        root.addLayout(buttons)

    def selected_classes(self):
        """List of class codes to save, or None for any class."""
        return self.class_combo.selected_classes()

    def selected_lines(self):
        """Flight-line ids to save, or None for any line."""
        return self.line_combo.selected_classes()

    def selected_format(self):
        """Format text such as 'LAZ 1.4'."""
        return self.format_combo.currentText()

    def selected_attributes(self):
        """Attribute labels chosen in the popup, or None if it was never used."""
        return self._selected_attributes

    def _open_attributes_dialog(self):
        dlg = AttributesToSaveDialog(
            self,
            filename=self._filename,
            available_dimensions=self._available_dimensions,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._selected_attributes = dlg.selected_attributes()
        self.attributes_btn.setToolTip(
            "Saving %d attribute(s) - same list as View Fields"
            % len(self._selected_attributes)
        )

    def selected_fences(self):
        """Fences chosen for "Inside fence only", or an empty list."""
        return list(self._selected_fences)

    def _choose_fences(self, checked=False):
        """Open the fence chooser; keep the tick only when one is picked.

        Clicking an already ticked box re-opens the chooser, so the
        selection can be changed without unticking it first.
        """
        dlg = FenceToSaveDialog(self, self._fence_items)
        dlg.set_selected(self._selected_fences)
        if dlg.exec() != QDialog.Accepted:
            self.fence_check.setChecked(bool(self._selected_fences))
            return
        self._selected_fences = dlg.selected_fences()
        self.fence_check.setChecked(bool(self._selected_fences))
        self._refresh_fence_text()

    def _refresh_fence_text(self):
        count = len(self._selected_fences)
        if count == 0:
            self.fence_check.setText("Inside fence only")
            self.fence_check.setToolTip(
                "Keep only the points inside a fence drawn with the "
                "fence tool"
            )
            return
        word = "fence" if count == 1 else "fences"
        self.fence_check.setText(f"Inside fence only ({count} {word})")
        self.fence_check.setToolTip(
            "Saving only the points inside: "
            + ", ".join(self._fence_labels_for(self._selected_fences))
        )

    def _fence_labels_for(self, fences):
        """Labels the fence dialog showed for these fences."""
        wanted = {id(fence) for fence in fences}
        return [
            label for label, fence in self._fence_items if id(fence) in wanted
        ] or ["the selected fence"]


def _label(text):
    from PySide6.QtWidgets import QLabel
    return QLabel(text)
