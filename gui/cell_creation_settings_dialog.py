"""Cell Creation Settings window.

The dialog reuses the Block Creation Settings window's shell: same CL
group, Clear / Create actions and styling. Its Cell Settings group is
cell-specific — Total Diameter (in mm), Color and Thickness only, without the
block dialog's Mode / Tolerance above / Tolerance below / Distance fields.

The **Select CL** workflow is a verbatim copy of the block dialog's
implementation — hover highlighting, click-to-select, Shift+click to add,
and the dotted rubber-band drag box that selects every CL it crosses
(including open lines attached from SNT files) — so picking centre lines
behaves identically in both windows.

"Create" draws one circle (Total Diameter, Color, Thickness) at every vertex of
the selected CL lines, start and end included; coincident vertices share a
single circle. Circles are ordinary digitizer drawings - nothing here writes
point-cloud data.
"""

import numpy as np
import vtk
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.block_creation_settings_dialog import (
    _CL_DRAG_THRESHOLD_PX,
    _get_block_dialog_style,
    _is_snt_selectable_line_entity,
)
from gui.icon_provider import _resolve_ribbon_icon_color

# Fallback swatch colour, matching the built-in "icon_create_cell" theme entry.
_DEFAULT_CELL_COLOR = "#2979ff"

# Default Total Diameter of the generated cell, in millimetres.
_DEFAULT_CELL_DIAMETER_MM = 3000.0

# Vertices closer than this (metres) are the same location: one circle only.
_VERTEX_MERGE_TOLERANCE_M = 0.01

# Segments used to draw each cell circle.
_CELL_CIRCLE_SEGMENTS = 72

# Default outline thickness of the generated cell, in pixels.
_DEFAULT_CELL_THICKNESS_PX = 2


def _remove_custom_color_section(dialog):
    """Hide the "Custom colors" block of a non-native colour dialog.

    Qt offers no option to drop that section, so it is hidden widget by widget:
    the swatch grid holding the custom colours, the "Custom colors" label and
    the "Add to Custom Colors" button. Everything is looked up by text and by
    Qt's own creation order, and any failure is tolerated — the picker still
    works, it just keeps the stock section.
    """
    label = next(
        (lab for lab in dialog.findChildren(QLabel)
         if lab.text().replace("&", "").strip().lower() == "custom colors"),
        None,
    )
    if label is None:
        return False

    # The grids are private Qt widgets (PySide wraps them as plain QWidget), so
    # they are identified through the meta-object name, and by creation order:
    # Qt builds the "Basic colors" grid first and the custom one second.
    children = [child for child in dialog.children() if isinstance(child, QWidget)]
    grids = [child for child in children
             if child.metaObject().className().endswith("QWellArray")]

    doomed = [label]
    if len(grids) > 1:
        grid = grids[-1]
        doomed.append(grid)
        # Blank spacer label that separates the two colour sections.
        preceding = children[:children.index(grid)]
        if preceding and isinstance(preceding[-1], QLabel) and not preceding[-1].text().strip():
            doomed.append(preceding[-1])

    doomed += [button for button in dialog.findChildren(QPushButton)
               if button.text().replace("&", "").strip().lower() == "add to custom colors"]

    for widget in doomed:
        widget.setVisible(False)
    return True


def _default_cell_color():
    """The Create Cell icon's own colour for the active theme.

    Read through the same resolver the ribbon icon uses, so the swatch default
    always matches what the button icon renders, and follow the theme when it
    changes (dark #2979ff / light #1565c0).
    """
    try:
        resolved = _resolve_ribbon_icon_color(
            button_text="Create Cell",
            section_title="Create",
            ribbon_scope="BlockRibbon",
        )
    except Exception:
        resolved = None
    return QColor(resolved) if resolved else QColor(_DEFAULT_CELL_COLOR)


class _CellColorButton(QPushButton):
    """Colour swatch: shows the current colour and opens a colour picker."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(24)
        self._color = QColor(color)
        self.clicked.connect(self._pick_color)
        self._refresh()

    def color(self):
        return QColor(self._color)

    def set_color(self, color):
        if color is not None and QColor(color).isValid():
            self._color = QColor(color)
            self._refresh()

    def _refresh(self):
        # Keep the hex label legible on top of the swatch itself.
        text_color = "#000000" if self._color.lightnessF() > 0.6 else "#ffffff"
        self.setText(self._color.name().upper())
        self.setToolTip(f"Cell color {self._color.name().upper()} — click to change")
        self.setStyleSheet(
            "QPushButton {"
            f" background-color: {self._color.name()}; color: {text_color};"
            " border: 1px solid rgba(255, 255, 255, 0.35); border-radius: 3px;"
            " padding: 2px 8px; text-align: left; }"
            "QPushButton:hover { border-color: rgba(255, 255, 255, 0.7); }"
        )

    def _pick_color(self):
        # Parented to the settings window, never to the swatch itself: a
        # stylesheet set on a widget cascades into its children, so a picker
        # parented to the swatch inherited the swatch's QPushButton rule and
        # repainted its own OK / Cancel buttons in the colour being picked.
        dialog = QColorDialog(self._color, self.window())
        dialog.setWindowTitle("Select Cell Color")
        # The custom-colors section only exists in the widget-based dialog.
        dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        _remove_custom_color_section(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_color(dialog.currentColor())

# Screen-space attribute the rubber band actor is stored under on the
# digitizer (see DigitizeManager._update_preview_actor_screen). Deliberately
# distinct from the block dialog's attribute so the two windows can never
# clear or redraw each other's selection rectangle.
_CL_DRAG_ACTOR_ATTR = "_cell_cl_rubber_band_actor"


class CellCreationSettingsDialog(QDialog):
    """Cell Creation Settings — block-dialog UI with a working Select CL."""

    def __init__(self, app=None, parent=None):
        _parent = parent if isinstance(parent, QWidget) else (app if isinstance(app, QWidget) else None)
        super().__init__(_parent)
        self.app = app
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.selected_cl_lines = []
        self._hovered_cl_line = None
        self._cl_select_active = False
        # Rubber-band drag state (display coords, bottom-left origin).
        self._cl_drag_start = None      # (x, y) where the left button went down
        self._cl_drag_active = False    # True once past the drag threshold
        self._snt_cl_pseudo_drawings = []
        self._snt_hover_actor = None       # transient VTK actor for hovered SNT CL line
        self._snt_selected_actors = {}     # id(drawing) -> VTK actor for selected SNT CL lines
        self._value_editors = []
        self.setWindowTitle("Cell Creation Settings")
        self.setModal(False)
        self.setWindowFlags(
            Qt.Dialog
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumWidth(400)

        self.setStyleSheet(_get_block_dialog_style())
        self._build_ui()

        # Lock the window height to the content, matching the block dialog's
        # fixed-height behaviour: with only Mode + Total Diameter the content is
        # shorter than the block dialog, and the window hugs it instead of
        # leaving dead space under the buttons.
        self._base_height = max(
            self.layout().sizeHint().height(),
            self.sizeHint().height(),
        )
        self.setFixedHeight(self._base_height)
        self.resize(400, self._base_height)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        content = QWidget()
        content.setObjectName("dialogContent")
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(2, 2, 2, 2)
        content_lay.setSpacing(8)
        root.addWidget(content)

        # All widgets below go into content_lay; alias as root.
        root = content_lay

        cl_group = QGroupBox("CL")
        cl_layout = QHBoxLayout(cl_group)
        cl_layout.setContentsMargins(10, 6, 10, 6)
        cl_layout.setSpacing(8)

        self.select_cl_btn = QPushButton("Select CL")
        self.select_cl_btn.setCheckable(True)
        self.select_cl_btn.setAutoDefault(False)
        self.select_cl_btn.setDefault(False)
        self.select_cl_btn.setCursor(Qt.PointingHandCursor)
        self.select_cl_btn.clicked.connect(self._toggle_cl_selection)
        cl_layout.addWidget(self.select_cl_btn, 0, Qt.AlignLeft)

        self.cl_selection_label = QLabel("No CL selected")
        self.cl_selection_label.setVisible(False)
        cl_layout.addWidget(self.cl_selection_label)
        cl_layout.addStretch()
        root.addWidget(cl_group)

        settings_group = QGroupBox("Cell Settings")
        form = QFormLayout(settings_group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.setContentsMargins(10, 6, 10, 6)

        # Cell Settings carries the diameter and its own colour / outline
        # thickness: the block-only fields (Mode, Tolerance above / below and
        # Distance) have no meaning for cells.
        self.total_diameter_spin = self._make_diameter_spin(_DEFAULT_CELL_DIAMETER_MM)
        form.addRow("Total Diameter:", self.total_diameter_spin)

        self.cell_color_btn = _CellColorButton(_default_cell_color())
        form.addRow("Color:", self.cell_color_btn)

        self.thickness_spin = self._make_pixel_spin(_DEFAULT_CELL_THICKNESS_PX)
        form.addRow("Thickness:", self.thickness_spin)

        root.addWidget(settings_group)

        btn_row = QHBoxLayout()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setDefault(False)
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_selection_and_reset)
        btn_row.addWidget(self.clear_btn, 0, Qt.AlignLeft)

        btn_row.addStretch()

        self.create_btn = QPushButton("Create")
        self.create_btn.setObjectName("okButton")
        self.create_btn.setAutoDefault(False)
        self.create_btn.setDefault(False)
        self.create_btn.setCursor(Qt.PointingHandCursor)
        self.create_btn.clicked.connect(self._on_create_clicked)

        # No Cancel button: like the block dialog, the window closes through
        # the title-bar X and "Clear" already resets the settings.
        btn_row.addWidget(self.create_btn)
        root.addLayout(btn_row)

    def _make_diameter_spin(self, value):
        """Total Diameter box, in millimetres."""
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0, 999999.0)
        spin.setSingleStep(0.05)
        spin.setSuffix(" mm")
        spin.setValue(value)
        self._register_value_editor(spin)
        return spin

    def _make_pixel_spin(self, value):
        spin = QSpinBox()
        spin.setRange(1, 100)
        spin.setSingleStep(1)
        spin.setSuffix(" px")
        spin.setValue(value)
        self._register_value_editor(spin)
        return spin

    def _register_value_editor(self, spin_box):
        spin_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._value_editors.append(spin_box)
        spin_box.installEventFilter(self)
        line_edit = spin_box.lineEdit()
        if line_edit is not None:
            self._value_editors.append(line_edit)
            line_edit.installEventFilter(self)

    def _reset_settings_fields(self):
        self.total_diameter_spin.setValue(_DEFAULT_CELL_DIAMETER_MM)
        self.cell_color_btn.set_color(_default_cell_color())
        self.thickness_spin.setValue(_DEFAULT_CELL_THICKNESS_PX)

    def _clear_selection_and_reset(self):
        """Clear's behaviour: drop any picked CL, then restore the defaults."""
        self._toggle_cl_selection(False)
        self._clear_selected_cl_lines()
        self._reset_settings_fields()
        self._render_view()

    def _unique_vertices(self):
        """Every vertex of the selected CL lines, one entry per location.

        Start and end vertices are included. Vertices that coincide - lines
        meeting at a shared node, or the same point listed twice - collapse to
        a single entry so only one circle is made there.
        """
        tolerance = _VERTEX_MERGE_TOLERANCE_M
        seen = {}
        vertices = []
        for drawing in self.selected_cl_lines:
            for vertex in drawing.get("coords") or []:
                x, y = float(vertex[0]), float(vertex[1])
                z = float(vertex[2]) if len(vertex) > 2 else 0.0
                # Grid-bucket lookup: a vertex is a duplicate when any point
                # within one tolerance of it (3x3 neighbouring cells) is kept.
                cell = (round(x / tolerance), round(y / tolerance))
                duplicate = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for ox, oy in seen.get((cell[0] + dx, cell[1] + dy), ()):
                            if (x - ox) ** 2 + (y - oy) ** 2 <= tolerance ** 2:
                                duplicate = True
                                break
                        if duplicate:
                            break
                    if duplicate:
                        break
                if duplicate:
                    continue
                seen.setdefault(cell, []).append((x, y))
                vertices.append((x, y, z))
        return vertices

    def _on_create_clicked(self):
        """Draw one circle at every vertex of the selected CL lines."""
        digitizer = getattr(self.app, "digitizer", None)
        if digitizer is None:
            print("Cell Creation: digitizer not available")
            return
        if not self.selected_cl_lines:
            self._show_status("Select at least one CL first", 3000)
            return

        vertices = self._unique_vertices()
        if not vertices:
            self._show_status("The selected CL has no vertices", 3000)
            return

        # Total Diameter is entered in millimetres; the scene is in metres.
        radius = float(self.total_diameter_spin.value()) / 2000.0
        if radius <= 0.0:
            self._show_status("Total Diameter must be greater than 0", 3000)
            return
        qcolor = self.cell_color_btn.color()
        color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
        width = int(self.thickness_spin.value())

        # One undo step covers the whole batch of circles.
        save_state = getattr(digitizer, "_save_state", None)
        if callable(save_state):
            save_state()

        thetas = np.linspace(0.0, 2.0 * np.pi, _CELL_CIRCLE_SEGMENTS, endpoint=False)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        created = 0
        for cx, cy, cz in vertices:
            coords = [
                (cx + radius * c, cy + radius * s, cz) for c, s in zip(cos_t, sin_t)
            ]
            coords.append(coords[0])
            actor = digitizer._make_polyline_actor(
                coords, color=color, width=width, line_style="solid"
            )
            digitizer._add_actor_to_overlay(actor)
            drawing = {
                "type": "circle",
                "coords": coords,
                "actor": actor,
                "bounds": actor.GetBounds(),
                "center": (cx, cy, cz),
                "radius": radius,
                "original_color": color,
                "original_width": width,
                "original_style": "solid",
                "source": "cell_creation",
            }
            tag = getattr(digitizer, "_tag_sbm", None)
            if callable(tag):
                tag(drawing)
            digitizer.drawings.append(drawing)
            emit = getattr(digitizer, "_emit_drawing_finalized", None)
            if callable(emit):
                emit(drawing)
            created += 1

        self._remember_source_cl_lines()
        print(
            f"Cell Creation: {created} circle(s), diameter "
            f"{self.total_diameter_spin.value():g} mm, from "
            f"{len(self.selected_cl_lines)} CL line(s)"
        )
        self._show_status(f"Created {created} cell circle(s)", 4000)
        self._render_view()

    def _remember_source_cl_lines(self):
        """Keep the SNT CL lines the cells were built on, for Shapefile export.

        SNT centre lines live in the attached SNT file, not in the digitizer's
        drawing list, so the drawings export would otherwise write the cells
        without the lines they belong to. Lines drawn with the digitizer are
        already exported as drawings and are not repeated here.
        """
        stored = getattr(self.app, "cell_source_cl_lines", None)
        if stored is None:
            stored = []
            setattr(self.app, "cell_source_cl_lines", stored)
        known = {
            (rec.get("snt_filename"), rec.get("snt_entity_id"),
             tuple(map(tuple, rec.get("coords") or ())))
            for rec in stored
        }
        for drawing in self.selected_cl_lines:
            if drawing.get("source") != "snt_cl":
                continue
            coords = [tuple(float(c) for c in v[:3]) for v in drawing.get("coords") or []]
            if len(coords) < 2:
                continue
            key = (drawing.get("snt_filename"), drawing.get("snt_entity_id"), tuple(coords))
            if key in known:
                continue
            known.add(key)
            stored.append({
                "coords": coords,
                "snt_filename": drawing.get("snt_filename"),
                "snt_entity_id": drawing.get("snt_entity_id"),
                "layer": drawing.get("layer", ""),
            })

    def _show_status(self, message, timeout_ms=3000):
        if hasattr(self.app, "statusBar"):
            try:
                self.app.statusBar().showMessage(message, timeout_ms)
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════════════════════
    #  Select CL — copied from Block Creation Settings (same behaviour)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_snt_cl_pseudo_drawings(self):
        """Return drawing-like dicts for selectable open lines in loaded SNTs.

        These are transient; they are rebuilt each time Select CL is armed and discarded
        when it is disarmed, so they hold no VTK resources and cannot leak. Only genuine
        open line geometry is admitted; grid/bbox/trajectory (including TRJ) support
        data is excluded.
        """
        result = []
        for attachment in getattr(self.app, "snt_attachments", []) or []:
            parsed = attachment.get("parsed") or {}
            entities = parsed.get("entities") or []
            fname = str(attachment.get("filename") or "SNT")
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                if not _is_snt_selectable_line_entity(ent):
                    continue
                vertices = ent.get("vertices") or ent.get("coords") or ent.get("points") or []
                coords = [
                    (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0)
                    for v in vertices
                ]
                result.append({
                    "type": "smartline",
                    "source": "snt_cl",
                    "layer": str(ent.get("layer", "")).strip(),
                    "snt_filename": fname,
                    "snt_entity_id": ent.get("_snt_entity_id"),
                    "snt_entity_type": str(ent.get("type", "")).strip().upper(),
                    "snt_open_line": True,
                    "coords": coords,
                })
        return result

    # ── SNT CL highlight helpers ──────────────────────────────────────────────

    def _make_snt_cl_actor(self, drawing, color, width):
        """Build a transient VTK overlay actor for an SNT CL pseudo-drawing."""
        digitizer = getattr(self.app, "digitizer", None)
        if digitizer is None:
            return None
        coords = drawing.get("coords", [])
        if len(coords) < 2:
            return None
        try:
            poly = digitizer._build_styled_polydata_world(coords, line_style="solid")
            origin = getattr(poly, "_world_origin", None)
            if origin is None:
                origin = np.zeros(3, dtype=np.float64)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
            prop.SetLineWidth(float(width))
            prop.SetOpacity(1.0)
            actor.PickableOff()
            actor.SetPosition(float(origin[0]), float(origin[1]), float(origin[2]))
            digitizer._add_actor_to_overlay(actor)
            return actor
        except Exception:
            return None

    def _remove_snt_cl_actor(self, actor):
        """Remove a transient SNT CL overlay actor from the renderer."""
        if actor is None:
            return
        digitizer = getattr(self.app, "digitizer", None)
        if digitizer is None:
            return
        try:
            digitizer._remove_actor_from_overlay(actor)
        except Exception:
            pass

    def _highlight_snt_cl_line(self, drawing):
        """Show a cyan highlight actor over a hovered SNT CL pseudo-drawing."""
        self._snt_hover_actor = self._make_snt_cl_actor(drawing, (0.0, 1.0, 1.0), 5)
        if self._snt_hover_actor is not None:
            self._render_view()

    def _unhighlight_snt_cl_line(self):
        """Remove the hover highlight actor for SNT CL lines."""
        if self._snt_hover_actor is not None:
            self._remove_snt_cl_actor(self._snt_hover_actor)
            self._snt_hover_actor = None
            self._render_view()

    def _select_snt_cl_line(self, drawing):
        """Add a persistent cyan selection actor for a selected SNT CL pseudo-drawing."""
        key = id(drawing)
        if key in self._snt_selected_actors:
            return
        actor = self._make_snt_cl_actor(drawing, (0.0, 1.0, 1.0), 5)
        if actor is not None:
            self._snt_selected_actors[key] = actor

    def _deselect_snt_cl_line(self, drawing):
        """Remove the selection actor for a SNT CL pseudo-drawing."""
        key = id(drawing)
        actor = self._snt_selected_actors.pop(key, None)
        self._remove_snt_cl_actor(actor)

    def _clear_all_snt_cl_actors(self):
        """Remove every transient SNT CL actor (hover + all selected)."""
        self._unhighlight_snt_cl_line()
        for actor in list(self._snt_selected_actors.values()):
            self._remove_snt_cl_actor(actor)
        self._snt_selected_actors.clear()

    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_cl_selection(self, checked):
        self._cl_select_active = bool(checked)
        self.select_cl_btn.blockSignals(True)
        self.select_cl_btn.setChecked(self._cl_select_active)
        self.select_cl_btn.blockSignals(False)
        self.cl_selection_label.setVisible(bool(self.selected_cl_lines))

        vtk_widget = getattr(self.app, "vtk_widget", None)
        if vtk_widget is not None:
            if self._cl_select_active:
                self._snt_cl_pseudo_drawings = self._build_snt_cl_pseudo_drawings()
                vtk_widget.installEventFilter(self)
                vtk_widget.setCursor(Qt.PointingHandCursor)
                if hasattr(self.app, "statusBar"):
                    n_snt = len(self._snt_cl_pseudo_drawings)
                    snt_hint = f" or an SNT line ({n_snt} available)" if n_snt else ""
                    self.app.statusBar().showMessage(
                        f"Select CL active: click a smart line{snt_hint}, "
                        "or drag a dotted box to select several "
                        "(Shift+click/drag to add more)",
                        0,
                    )
            else:
                self._cancel_cl_drag()
                self._snt_cl_pseudo_drawings = []
                vtk_widget.removeEventFilter(self)
                vtk_widget.setCursor(Qt.ArrowCursor)
                self._clear_hover_cl_line()
                if hasattr(self.app, "statusBar"):
                    self.app.statusBar().showMessage("Select CL inactive", 2000)
                self._refresh_cl_selection_label()

    def eventFilter(self, obj, event):
        if any(editor is obj for editor in self._value_editors) and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                spin_box = obj if isinstance(obj, QAbstractSpinBox) else obj.parent()
                if isinstance(spin_box, QAbstractSpinBox):
                    spin_box.interpretText()
                    spin_box.selectAll()
                    return True

        if not self._cl_select_active or obj is not getattr(self.app, "vtk_widget", None):
            return super().eventFilter(obj, event)

        event_type = event.type()
        if event_type == QEvent.MouseMove:
            if self._cl_drag_start is not None:
                # While the rubber band is live, swallow the move so the VTK
                # interactor cannot start its own gesture underneath the box.
                self._update_cl_drag(event)
                return True
            self._update_hover_cl_line(event)
            return False

        if event_type == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            # Selection is committed on release: a press/release that never
            # moves past the threshold is treated as the plain click-select,
            # anything further becomes a rubber-band rectangle selection.
            self._begin_cl_drag(event)
            return True

        if event_type == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._cl_drag_start is not None:
                self._finish_cl_drag(event)
                return True
            return False

        if event_type == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._cancel_cl_drag()
            self._toggle_cl_selection(False)
            return True

        return False

    def _event_display_pos(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x = int(pos.x())
        y = int(pos.y())
        vtk_widget = getattr(self.app, "vtk_widget", None)
        height = vtk_widget.height() if vtk_widget is not None else 0
        return x, max(0, height - y)

    # ── Rubber-band (drag rectangle) selection ──────────────────────────────

    def _begin_cl_drag(self, event):
        """Remember where the left button went down; no selection yet."""
        self._cl_drag_start = self._event_display_pos(event)
        self._cl_drag_active = False

    def _update_cl_drag(self, event):
        """Extend the live rubber band once the pointer moves far enough."""
        if self._cl_drag_start is None:
            return
        pos = self._event_display_pos(event)
        if not self._cl_drag_active:
            sx, sy = self._cl_drag_start
            if (
                abs(pos[0] - sx) < _CL_DRAG_THRESHOLD_PX
                and abs(pos[1] - sy) < _CL_DRAG_THRESHOLD_PX
            ):
                return
            self._cl_drag_active = True
            # Drop the hover highlight so it does not sit under the box.
            self._clear_hover_cl_line()
        self._draw_cl_drag_rect(self._cl_drag_start, pos)

    def _finish_cl_drag(self, event):
        """Release: plain click selects one line, a drag selects a rectangle."""
        start = self._cl_drag_start
        dragged = self._cl_drag_active
        self._cl_drag_start = None
        self._cl_drag_active = False

        if not dragged:
            # Unchanged click-to-select behaviour (hover/Shift+click).
            self._select_hovered_cl_line(event)
            return

        end = self._event_display_pos(event)
        self._clear_cl_drag_rect()
        drawings = self._drawings_in_display_rect(start, end)
        if not drawings:
            self._render_view()
            return
        self._select_cl_drawings(drawings)

    def _cancel_cl_drag(self):
        """Abandon an in-flight drag and remove its rectangle."""
        had_drag = self._cl_drag_start is not None or self._cl_drag_active
        self._cl_drag_start = None
        self._cl_drag_active = False
        if had_drag:
            self._clear_cl_drag_rect()

    def _draw_cl_drag_rect(self, start, end):
        """Show the dotted rubber-band rectangle in screen space."""
        digitizer = getattr(self.app, "digitizer", None)
        update = getattr(digitizer, "_update_preview_actor_screen", None)
        if update is None:
            return
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        try:
            update(
                _CL_DRAG_ACTOR_ATTR,
                points,
                color=(0.0, 1.0, 1.0),
                width=1,
                line_style="dotted",
            )
        except Exception as exc:
            print(f"Failed to draw CL selection rectangle: {exc}")
            return
        self._render_view()

    def _clear_cl_drag_rect(self):
        """Remove the dotted rubber-band rectangle from the scene."""
        digitizer = getattr(self.app, "digitizer", None)
        remover = getattr(digitizer, "_remove_preview_actor_2d", None)
        if remover is not None:
            try:
                remover(_CL_DRAG_ACTOR_ATTR)
            except Exception:
                pass
        self._render_view()

    @staticmethod
    def _segment_intersects_rect(p1, p2, rect):
        """Liang-Barsky test: does a 2D segment touch an axis-aligned box?"""
        xmin, xmax, ymin, ymax = rect
        x1 = float(p1[0])
        y1 = float(p1[1])
        dx = float(p2[0]) - x1
        dy = float(p2[1]) - y1
        t0 = 0.0
        t1 = 1.0
        for p, q in (
            (-dx, x1 - xmin),
            (dx, xmax - x1),
            (-dy, y1 - ymin),
            (dy, ymax - y1),
        ):
            if abs(p) < 1e-12:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
        return t0 <= t1

    def _drawings_in_display_rect(self, start, end):
        """Every selectable CL/smart line crossing the dragged screen box."""
        digitizer = getattr(self.app, "digitizer", None)
        if digitizer is None:
            return []
        renderer = getattr(digitizer, "renderer", None)
        if renderer is None:
            return []

        xmin, xmax = sorted((float(start[0]), float(end[0])))
        ymin, ymax = sorted((float(start[1]), float(end[1])))
        rect = (xmin, xmax, ymin, ymax)

        candidates = list(getattr(digitizer, "drawings", []))
        candidates += list(self._snt_cl_pseudo_drawings or [])

        hits = []
        for drawing in candidates:
            if not self._is_selectable_cl_line(drawing):
                continue

            coords = drawing.get("coords", [])
            if len(coords) < 2:
                continue

            for idx in range(len(coords) - 1):
                p1 = coords[idx]
                p2 = coords[idx + 1]

                renderer.SetWorldPoint(p1[0], p1[1], p1[2], 1.0)
                renderer.WorldToDisplay()
                s1 = renderer.GetDisplayPoint()

                renderer.SetWorldPoint(p2[0], p2[1], p2[2], 1.0)
                renderer.WorldToDisplay()
                s2 = renderer.GetDisplayPoint()

                if self._segment_intersects_rect(s1, s2, rect):
                    hits.append(drawing)
                    break

        return hits

    # ────────────────────────────────────────────────────────────────────────

    def _find_smartline_at_event(self, event):
        drawings = self._find_smartlines_at_event(event)
        return drawings[0] if drawings else None

    def _is_selectable_cl_line(self, drawing):
        if not drawing:
            return False
        source = drawing.get("source", "")
        if source == "snt_cl":
            return drawing.get("snt_open_line") is True
        return (
            drawing.get("type") in ("smartline", "smart_line")
            and source != "block_creation"
        )

    def _has_selected_cl_line(self, drawing):
        return any(selected is drawing for selected in self.selected_cl_lines)

    def _is_shift_selection(self, event):
        try:
            if event is not None and event.modifiers() & Qt.ShiftModifier:
                return True
        except Exception:
            pass
        return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)

    def _find_smartlines_at_event(self, event, tolerance=25.0):
        digitizer = getattr(self.app, "digitizer", None)
        if digitizer is None:
            return []

        renderer = getattr(digitizer, "renderer", None)
        distance_fn = getattr(digitizer, "_distance_point_to_segment_2d", None)
        if renderer is None or distance_fn is None:
            x, y = self._event_display_pos(event)
            drawing = digitizer._get_drawing_under_cursor(x, y, tolerance=tolerance)
            return [drawing] if self._is_selectable_cl_line(drawing) else []

        x, y = self._event_display_pos(event)
        mouse_p = np.array([x, y])
        hits = []

        all_candidates = list(getattr(digitizer, "drawings", [])) + self._snt_cl_pseudo_drawings
        for drawing in all_candidates:
            if not self._is_selectable_cl_line(drawing):
                continue

            coords = drawing.get("coords", [])
            if len(coords) < 2:
                continue

            min_dist = None
            for idx in range(len(coords) - 1):
                p1_world = coords[idx]
                p2_world = coords[idx + 1]

                renderer.SetWorldPoint(p1_world[0], p1_world[1], p1_world[2], 1.0)
                renderer.WorldToDisplay()
                s1 = np.array(renderer.GetDisplayPoint()[:2])

                renderer.SetWorldPoint(p2_world[0], p2_world[1], p2_world[2], 1.0)
                renderer.WorldToDisplay()
                s2 = np.array(renderer.GetDisplayPoint()[:2])

                dist = float(distance_fn(mouse_p, s1, s2))
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            if min_dist is not None and min_dist <= tolerance:
                hits.append((min_dist, drawing))

        hits.sort(key=lambda item: item[0])
        return [drawing for _, drawing in hits]

    def _update_hover_cl_line(self, event):
        candidates = self._find_smartlines_at_event(event)
        drawing = next((d for d in candidates if not self._has_selected_cl_line(d)), None)
        if drawing is self._hovered_cl_line:
            return

        self._clear_hover_cl_line()
        if drawing is None:
            return

        self._hovered_cl_line = drawing
        if drawing.get("source") == "snt_cl":
            self._highlight_snt_cl_line(drawing)
        else:
            digitizer = getattr(self.app, "digitizer", None)
            if digitizer is not None and hasattr(digitizer, "_highlight_line"):
                digitizer._highlight_line(drawing)
                self._render_view()

    def _select_cl_drawings(self, drawings):
        """Add the given lines to the current CL selection (additive)."""
        drawings = [drawing for drawing in drawings if self._is_selectable_cl_line(drawing)]
        if not drawings:
            return False

        # Remove the hover actor before promoting to selection actor.
        self._unhighlight_snt_cl_line()

        digitizer = getattr(self.app, "digitizer", None)
        for drawing in drawings:
            if not self._has_selected_cl_line(drawing):
                self.selected_cl_lines.append(drawing)
                if drawing.get("source") == "snt_cl":
                    self._select_snt_cl_line(drawing)
                else:
                    if digitizer is not None and hasattr(digitizer, "_highlight_line"):
                        digitizer._highlight_line(drawing)

        self._hovered_cl_line = None
        self._refresh_cl_selection_label()
        self._render_view()
        return True

    def _select_hovered_cl_line(self, event=None):
        if self._is_shift_selection(event):
            drawings = self._find_smartlines_at_event(event)
        else:
            drawings = [self._hovered_cl_line]

        self._select_cl_drawings(drawings)

    def _refresh_cl_selection_label(self):
        if not self.selected_cl_lines:
            self.cl_selection_label.setText("No CL selected")
            self.cl_selection_label.setVisible(False)
            return

        labels = []
        for idx, d in enumerate(self.selected_cl_lines, 1):
            if d.get("source") == "snt_cl":
                fname = d.get("snt_filename", "SNT")
                labels.append(f"Line {idx} (SNT:{fname})")
            else:
                labels.append(f"Line {idx}")
        self.cl_selection_label.setText(", ".join(labels))
        # Selecting the lines IS the confirmation, so the list is shown live while
        # picking instead of behind a "Done" popup the user had to dismiss.
        self.cl_selection_label.setVisible(True)

    def _clear_hover_cl_line(self):
        drawing = self._hovered_cl_line
        if drawing is None:
            return

        self._hovered_cl_line = None
        if drawing.get("source") == "snt_cl":
            self._unhighlight_snt_cl_line()
        else:
            if not self._has_selected_cl_line(drawing):
                digitizer = getattr(self.app, "digitizer", None)
                if digitizer is not None and hasattr(digitizer, "_unhighlight_line"):
                    digitizer._unhighlight_line(drawing)
                    self._render_view()

    def _render_view(self):
        vtk_widget = getattr(self.app, "vtk_widget", None)
        if vtk_widget is not None and hasattr(vtk_widget, "render"):
            vtk_widget.render()

    def _clear_selected_cl_lines(self):
        digitizer = getattr(self.app, "digitizer", None)
        for drawing in list(self.selected_cl_lines):
            if drawing.get("source") == "snt_cl":
                self._deselect_snt_cl_line(drawing)
            elif digitizer is not None and hasattr(digitizer, "_unhighlight_line"):
                try:
                    digitizer._unhighlight_line(drawing)
                except Exception:
                    pass

        self._clear_all_snt_cl_actors()
        self.selected_cl_lines = []
        self._hovered_cl_line = None
        self._snt_cl_pseudo_drawings = []
        self._refresh_cl_selection_label()

    def closeEvent(self, event):
        """Leave no CL highlight or armed selection behind when closed."""
        self._toggle_cl_selection(False)
        self._clear_selected_cl_lines()
        self._render_view()
        super().closeEvent(event)


def show_cell_creation_settings_dialog(app, parent=None):
    dialog = CellCreationSettingsDialog(app, parent)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
