"""Shared selection rules for source-class lists in both rendering backends."""


def bind_any_class_selection(widget):
    """Keep the Any Class sentinel exclusive, including Ctrl/Shift selection.

    Connect before the dialog's status/preview slots so they always read the
    effective selection. Choosing specific classes replaces Any; choosing Any
    replaces the specific set. Clearing remains an empty selection.
    """
    from PySide6.QtCore import Qt, QSignalBlocker

    previous = set()

    def normalize():
        nonlocal previous
        items = widget.selectedItems()
        codes = {item.data(Qt.UserRole) for item in items}
        if None in codes and len(codes) > 1:
            added = codes - previous
            keep_any = None in added
            blocker = QSignalBlocker(widget)
            for item in items:
                if (item.data(Qt.UserRole) is None) != keep_any:
                    item.setSelected(False)
            del blocker
            codes = {item.data(Qt.UserRole) for item in widget.selectedItems()}
        previous = codes

    widget.itemSelectionChanged.connect(normalize)
    normalize()
