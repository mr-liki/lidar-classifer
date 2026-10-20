"""
Identification Tool for NakshaAI (native Vulkan; VTK-free).

Gestures arrive through the native input filter (see gui/native_identify.py).
This module keeps the shared, renderer-agnostic logic: index-based identify,
ribbon/statistics updates, class metadata, nearest-point search.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
import numpy as np

try:
    from shiboken6 import isValid as _qt_object_is_valid
except Exception:
    def _qt_object_is_valid(obj):
        return obj is not None


class IdentificationTool(QObject):
    """Tool for identifying point properties on click (native Vulkan)."""

    point_identified = Signal(int, str, tuple)  # class_code, class_name, (x, y, z)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.active = False
        # Compat: the native path never creates pickers/observers.
        self.picker = None
        self._click_observer = None
        self._section_observers = {}
        print("IdentificationTool initialized (native)")

    def activate(self):
        """Activate identification mode (native Vulkan; no VTK observers)."""
        if self.active:
            print("Identification tool already active")
            return
        self.active = True
        print("Identification tool ACTIVATED (native)")

    def deactivate(self):
        """Deactivate identification mode."""
        if not self.active:
            return
        self.active = False
        print("Identification tool DEACTIVATED")
        self.deactivate_all_sections()
        self.clear_highlight()

    def _on_left_click(self, obj, event):
        """Retired VTK entry point. Native clicks arrive via NativeIdentify."""
        return None

    def activate_for_section(self, section_widget, view_index):
        """Retired VTK entry point (native canvases own their gestures)."""
        return None

    def deactivate_for_section(self, view_index):
        """Retired VTK entry point."""
        return None

    def deactivate_all_sections(self):
        """Retired VTK entry point."""
        try:
            self._section_observers = {}
        except Exception:
            pass
        return None

    def _on_section_click(self, obj, event, section_widget, view_index):
        """Retired VTK entry point."""
        return None

    def _get_section_xyz(self, view_index):
        """Retired VTK helper (native reads the section store instead)."""
        return None

    def _get_original_index(self, view_index, section_local_idx):
        """Retired VTK helper."""
        return None

    def _cut_section_is_taking(self):
        """
        Return True if a cut section is currently being placed/taken (consuming
        left-click input). While in this state the IdentificationTool must NOT
        identify points, otherwise it visibly interferes with cut placement.

        CutSectionState: IDLE=0, WAITING_CENTER=1, WAITING_DEPTH=2, FINALIZED=3.
        We compare integer values to avoid importing the enum (keeps this tool
        free of a circular dependency on the cross_section package).
        """
        cut = getattr(self.app, "cut_section_controller", None)
        if cut is None:
            return False
        state = getattr(cut, "_state", 0)
        if state in (1, 2):  # WAITING_CENTER / WAITING_DEPTH
            return True
        return False

    def identify_point_index(self, point_index):
        """
        Identify an already-known global point index using the same backend as
        an interactive 3D click.

        This is intentionally a public, index-based entry point so dialogs such
        as View Fields do not have to fake a VTK pick or duplicate identify
        ribbon/statistics logic.

        Returns a small result dict on success, otherwise None.
        """
        try:
            data = getattr(self.app, "data", None) or {}
            xyz = data.get("xyz")
            classification = data.get("classification")
            if xyz is None or classification is None:
                print("   ⚠️ identify_point_index: xyz/classification unavailable")
                return None

            idx = int(point_index)
            if idx < 0 or idx >= len(xyz) or idx >= len(classification):
                print(f"   ⚠️ identify_point_index: index out of range: {idx}")
                return None

            class_code = int(classification[idx])
            class_name = self.get_class_name(class_code)
            actual_pos = tuple(float(v) for v in xyz[idx])

            self.highlight_class(class_code)
            self._update_ribbon_info(
                class_code,
                class_name,
                actual_pos,
                point_index=idx,
            )
            self.point_identified.emit(class_code, class_name, actual_pos)

            print(
                "   ✅ Identified point index %d: class %d (%s) at "
                "(%.3f, %.3f, %.3f)" % (
                    idx, class_code, class_name,
                    actual_pos[0], actual_pos[1], actual_pos[2],
                )
            )
            return {
                "point_index": idx,
                "class_code": class_code,
                "class_name": class_name,
                "xyz": actual_pos,
            }
        except Exception as exc:
            print(f"   ❌ identify_point_index failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def _collect_field_values(self, class_code, class_name, xyz, point_index=None):
        """
        Build an ordered {field_label: display_value} dict for the picked point,
        restricted to the fields checked in the View Fields dialog (Display ▸
        Fields) and to data actually kept in memory for the loaded cloud.
        """
        from gui.dialogs.view_fields_dialog import (
            DEFAULT_CHECKED,
            FIELD_SPECS,
            find_image_dimension_name,
        )

        selected = getattr(self.app, "selected_view_fields", None)
        selected = set(selected) if selected is not None else set(DEFAULT_CHECKED)

        x, y, z = xyz
        class_lvl = self.get_class_lvl(class_code)
        raw_values = {
            "Class": str(class_code),
            "Description": class_lvl if class_lvl else class_name,
            "Easting": f"{x:.2f}",
            "Northing": f"{y:.2f}",
            "Elevation": f"{z:.2f}",
        }

        data = getattr(self.app, "data", None) or {}
        if point_index is not None:
            image_key = find_image_dimension_name(data.keys())
            image_values = data.get(image_key) if image_key is not None else None
            if image_values is not None and 0 <= point_index < len(image_values):
                raw_values["Image"] = str(image_values[point_index])

            intensity = data.get("intensity")
            if intensity is not None and 0 <= point_index < len(intensity):
                raw_values["Intensity"] = f"{float(intensity[point_index]):.1f}"

            rgb = data.get("rgb")
            if rgb is not None and 0 <= point_index < len(rgb):
                r, g, b = (float(v) for v in rgb[point_index])
                if max(r, g, b) <= 1.0:
                    # The native section store keeps the colours as 0..1 floats.
                    r, g, b = r * 255.0, g * 255.0, b * 255.0
                raw_values["Color RGB"] = f"{int(r)}, {int(g)}, {int(b)}"

        return {
            label: raw_values[label]
            for label, _dims, _always in FIELD_SPECS
            if label in selected and label in raw_values
        }

    def _update_ribbon_info(self, class_code, class_name, xyz, point_index=None):
        """Update the identification ribbon with point info"""
        try:
            # Sync the View Fields table (if open) with the identified point.
            if point_index is not None:
                table_dlg = getattr(self.app, "_view_fields_table_dialog", None)
                if table_dlg is not None and table_dlg.isVisible():
                    try:
                        print(f"   🔗 Syncing View Fields table (point {point_index})")
                        table_dlg.highlight_point(point_index)
                    except Exception as exc:
                        print(f"   ⚠️ View Fields table highlight failed: {exc}")
            # Get color and level regardless of ribbon availability
            class_color = self.get_class_color(class_code)
            class_lvl = self.get_class_lvl(class_code)
            display_name = class_lvl if class_lvl else class_name
            fields = self._collect_field_values(class_code, class_name, xyz, point_index)

            if hasattr(self.app, 'ribbon_manager'):
                identify_ribbon = self.app.ribbon_manager.ribbons.get('identify')
                if identify_ribbon:
                    identify_ribbon.update_info(
                        class_code,
                        class_name,
                        xyz,
                        color=class_color,
                        lvl=class_lvl,
                        fields=fields,
                    )

            # Update footer label with color dot + "Class: code : name"
            if hasattr(self.app, 'snt_layer_pick_footer_label'):
                lbl = self.app.snt_layer_pick_footer_label
                from PySide6.QtCore import Qt

                # Build color strings
                if class_color and len(class_color) >= 3:
                    r, g, b = int(class_color[0]), int(class_color[1]), int(class_color[2])
                else:
                    r, g, b = 160, 160, 160

                hex_color = f"#{r:02X}{g:02X}{b:02X}"

                # Use rich-text HTML: colored ● dot + class text
                plain_text = f"Class: {class_code} : {display_name}"
                html = (
                    f'<span style="color:{hex_color}; font-size:14px;">&#9679;</span>'
                    f'&nbsp;<span style="font-weight:600;">{plain_text}</span>'
                )

                lbl.setTextFormat(Qt.RichText)
                lbl.setText(html)

                # Colored left-border accent + subtle tinted background
                lbl.setStyleSheet(
                    f"""
                    QLabel {{
                        border-left: 3px solid {hex_color};
                        padding-left: 5px;
                        background: rgba({r},{g},{b},30);
                        border-radius: 3px;
                        font-size: 11px;
                    }}
                    """
                )
                lbl.setToolTip(plain_text)
                lbl.setStatusTip(plain_text)
                lbl.setVisible(True)

        except Exception as e:
            print(f"   ⚠️ Failed to update ribbon: {e}")

    def highlight_class(self, class_code):
        """
        ✅ NEW: Highlight a class in the Point Statistics widget.
        
        Args:
            class_code: The classification code to highlight
        """
        if not hasattr(self.app, 'point_count_widget') or not self.app.point_count_widget:
            print("   ⚠️ Point Statistics widget not available")
            return
        
        try:
            # Get the widget's stats container
            stats_container = self.app.point_count_widget.stats_container
            layout = self.app.point_count_widget.stats_layout
            
            # Iterate through all stat widgets to find and highlight the matching class
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget and hasattr(widget, 'property'):
                    # Check if this widget represents our class
                    stored_code = widget.property('class_code')
                    if stored_code == class_code:
                        # Apply highlight style
                        self._apply_highlight_style(widget, class_code)
                        
                        # Scroll to make it visible
                        if hasattr(self.app.point_count_widget, 'parent') and \
                           hasattr(self.app.point_count_widget.parent(), 'ensureWidgetVisible'):
                            self.app.point_count_widget.parent().ensureWidgetVisible(widget)
                        
                        print(f"   ✅ Highlighted class {class_code} in Point Statistics")
                        return
            
            print(f"   ⚠️ Class {class_code} not found in Point Statistics widget")
            
        except Exception as e:
            print(f"   ⚠️ Failed to highlight class: {e}")
            import traceback
            traceback.print_exc()

    def _apply_highlight_style(self, widget, class_code):
        """Apply a highlight animation/style to a widget."""
        try:
            from PySide6.QtCore import QPropertyAnimation, QEasingCurve
            from PySide6.QtGui import QColor
            
            # Get the original style
            original_style = widget.styleSheet()
            
            # Create a pulsing highlight effect
            def pulse_highlight():
                if widget is None or not _qt_object_is_valid(widget):
                    return
                # Bright highlight
                highlight_style = original_style.replace(
                    'background-color: rgba(',
                    'background-color: rgba(255, 255, 100, 0.4); /* background-color: rgba('
                )
                widget.setStyleSheet(highlight_style)
                
                # After 500ms, fade back
                from PySide6.QtCore import QTimer
                def _restore_style():
                    try:
                        if widget is None or not _qt_object_is_valid(widget):
                            return
                        widget.setStyleSheet(original_style)
                    except Exception:
                        pass
                QTimer.singleShot(500, _restore_style)
            
            # Trigger pulse
            pulse_highlight()
            
        except Exception as e:
            print(f"   ⚠️ Failed to apply highlight animation: {e}")

    def clear_highlight(self):
        """Clear all highlights from the Point Statistics widget."""
        if not hasattr(self.app, 'point_count_widget') or not self.app.point_count_widget:
            return
        
        try:
            # Simply refresh the widget to restore original styles
            self.app.point_count_widget.update_statistics()
        except Exception as e:
            print(f"   ⚠️ Failed to clear highlight: {e}")
            
        if hasattr(self.app, '_set_snt_layer_pick_footer_text'):
            self.app._set_snt_layer_pick_footer_text(None)
        # Also reset footer stylesheet and text format
        if hasattr(self.app, 'snt_layer_pick_footer_label'):
            from PySide6.QtCore import Qt
            lbl = self.app.snt_layer_pick_footer_label
            lbl.setTextFormat(Qt.PlainText)
            lbl.setStyleSheet("")

    def get_class_color(self, class_code):
        """
        Get the RGB color for a classification code from the app's class palette.
        Returns tuple (r, g, b) or None if not found.
        """
        # Try to get from app's class palette first
        if hasattr(self.app, 'class_palette') and self.app.class_palette:
            class_info = self.app.class_palette.get(class_code)
            if class_info and 'color' in class_info:
                return class_info['color']
        
        # Try Display Mode dialog palette as backup
        if hasattr(self.app, 'display_mode_dialog') and self.app.display_mode_dialog:
            dialog = self.app.display_mode_dialog
            if hasattr(dialog, 'view_palettes'):
                for view_palette in dialog.view_palettes.values():
                    if class_code in view_palette:
                        color = view_palette[class_code].get('color')
                        if color:
                            return color
        
        # Try Display Mode table directly
        if hasattr(self.app, 'display_dialog') and self.app.display_dialog:
            dialog = self.app.display_dialog
            if hasattr(dialog, 'table'):
                table = dialog.table
                for row in range(table.rowCount()):
                    try:
                        code = int(table.item(row, 1).text())
                        if code == class_code:
                            color_item = table.item(row, 5)
                            if color_item:
                                color = color_item.background().color().getRgb()[:3]
                                return color
                    except Exception:
                        continue
        
        # Fallback to default gray
        return (160, 160, 160)

    def get_class_name(self, class_code):
        """
        Get the name for a classification code from the app's class palette.
        Falls back to generic name if not found in active palettes.
        """
        if hasattr(self.app, 'display_mode_dialog') and self.app.display_mode_dialog:
            table = self.app.display_mode_dialog.table
            for row in range(table.rowCount()):
                try:
                    code = int(table.item(row, 1).text())
                    if code == class_code:
                        desc = table.item(row, 2).text()  # Description column
                        return desc
                except Exception:
                    continue
        # Try to get from app's class palette first
        if hasattr(self.app, 'class_palette') and self.app.class_palette:
            class_info = self.app.class_palette.get(class_code)
            if class_info and 'description' in class_info:
                return class_info['description']
        
        # Try Display Mode dialog palette as backup
        if hasattr(self.app, 'display_mode_dialog') and self.app.display_mode_dialog:
            dialog = self.app.display_mode_dialog
            if hasattr(dialog, 'view_palettes'):
                for view_palette in dialog.view_palettes.values():
                    if class_code in view_palette:
                        return view_palette[class_code].get('description', f"Class {class_code}")
        
        return f"Class {class_code}"

    def _is_class_visible(self, class_code, view_index=0):
        """Check if a classification code is visible in the given view/slot."""
        # Main view
        if view_index == 0:
            if hasattr(self.app, 'class_palette') and self.app.class_palette:
                class_info = self.app.class_palette.get(class_code)
                if class_info is not None:
                    return class_info.get('show', True)
        
        # Check specific view_palettes
        if hasattr(self.app, 'view_palettes') and self.app.view_palettes:
            slot_palette = self.app.view_palettes.get(view_index)
            if slot_palette is not None:
                class_info = slot_palette.get(class_code)
                if class_info is not None:
                    return class_info.get('show', True)
                    
        # Check display_mode_dialog backup
        if hasattr(self.app, 'display_mode_dialog') and self.app.display_mode_dialog:
            dialog = self.app.display_mode_dialog
            if hasattr(dialog, 'view_palettes') and dialog.view_palettes:
                slot_palette = dialog.view_palettes.get(view_index)
                if slot_palette is not None:
                    class_info = slot_palette.get(class_code)
                    if class_info is not None:
                        return class_info.get('show', True)
        
        # Default is visible
        return True

    def _nearest_global_index(self, picked_pos, xyz, view_index=0):
        """Resolve nearest global index of a visible class using spatial index when available."""
        classification = self.app.data.get('classification') if hasattr(self.app, 'data') and self.app.data is not None else None
        
        try:
            spatial_index = getattr(self.app, "spatial_index", None)
            tree = getattr(spatial_index, "tree", None)
            if tree is not None:
                # Query k nearest points (e.g. k=100) to find one that is visible
                k_val = min(100, len(xyz))
                distances, indices = tree.query(np.asarray(picked_pos, dtype=float), k=k_val)
                
                if k_val == 1:
                    distances = [distances]
                    indices = [indices]
                
                for idx, dist in zip(indices, distances):
                    if idx >= len(xyz):
                        continue
                    if classification is not None:
                        class_code = int(classification[idx])
                        if not self._is_class_visible(class_code, view_index=view_index):
                            continue
                    return int(idx), float(dist)
                
                print("   ⚠️ No visible class points found in spatial index search window")
                return None, None
        except Exception as e:
            print(f"   ⚠️ Spatial nearest query failed, falling back: {e}")

        try:
            distances = np.linalg.norm(xyz - np.asarray(picked_pos, dtype=float), axis=1)
            if classification is not None:
                unique_classes = np.unique(classification)
                hidden_classes = [c for c in unique_classes if not self._is_class_visible(int(c), view_index=view_index)]
                if hidden_classes:
                    hidden_mask = np.isin(classification, hidden_classes)
                    distances[hidden_mask] = np.inf
            
            index = int(np.argmin(distances))
            dist = float(distances[index])
            if dist == np.inf:
                return None, None
            return index, dist
        except Exception as e:
            print(f"   ⚠️ Numpy nearest query failed: {e}")
            return None, None

    def get_class_lvl(self, class_code):
        """
        Get the Level (Lvl) for a classification code from Display Mode table.
        Returns string or empty string if not found.
        """
        # Read from Display Mode dialog table (Column 4 = Lvl)
        if hasattr(self.app, 'display_mode_dialog') and self.app.display_mode_dialog:
            table = self.app.display_mode_dialog.table
            for row in range(table.rowCount()):
                try:
                    code = int(table.item(row, 1).text())
                    if code == class_code:
                        lvl_item = table.item(row, 4)  # Column 4 = Lvl
                        return lvl_item.text() if lvl_item else ""
                except Exception:
                    continue
        
        # Fallback
        return ""
