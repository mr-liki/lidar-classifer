from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from PySide6.QtWidgets import QCheckBox

from gui.update_manager import APP_VERSION, UpdateInfo, UpdateManager, UpdateWorker

class GlobalSettingsDialog(QDialog):
    """Category-based global settings window."""

    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.settings = QSettings("NakshaAI", "LidarApp")
        self.update_manager = UpdateManager(self.settings)
        self._update_worker = None
        self._available_update = None
        self._downloaded_update = None
        self._pending_auto_download = False

        self._dirty_categories = set()
        self._loading = False
        self._built_pages = set()
        self._loaded_pages = set()

        self.setWindowTitle("Global Settings")
        self.setModal(False)
        self.resize(900, 620)

        self._init_colors()
        self._build_ui()

    def _init_colors(self):
        from gui.theme_manager import ThemeColors as _C

        def _g(key, fallback):
            try:
                val = _C.get(key)
            except Exception:
                val = None
            return val or fallback

        is_light = False
        try:
            is_light = bool(_C.is_light()) if hasattr(_C, "is_light") else False
        except Exception:
            is_light = False

        if is_light:
            self.c_bg          = _g("bg_primary", "#f0f2f5")
            self.c_bg_alt      = _g("bg_secondary", "#ffffff")
            self.c_bg_sidebar  = _g("bg_secondary", "#ffffff")
            self.c_sidebar_sel = _g("dialog_selection", "#edf1f4")
            self.c_sidebar_txt = _g("text_secondary", "#607d8b")
            self.c_sidebar_txt_sel = _g("text_primary", "#1b2838")
            self.c_border      = _g("border", "#d0d5dd")
            self.c_text        = _g("text_primary", "#1b2838")
            self.c_text_muted  = _g("text_secondary", "#607d8b")
            self.c_accent      = _g("accent", "#0077b6")
            self.c_primary     = _g("dialog_primary_bg", "#e7ebef")
            self.c_primary_hover = _g("dialog_primary_hover", "#dde3e9")
            self.c_primary_border = _g("dialog_primary_border", "#c3ccd5")
            self.c_primary_text = _g("dialog_primary_text", "#26313a")
            self.c_table_sel   = _g("dialog_selection", "#edf1f4")
            self.c_hover       = _g("bg_button_hover", "#e0e4ea")
        else:
            # Match the main application dark theme (ThemeColors.DARK).
            self.c_bg          = _g("bg_primary", "#121212")
            self.c_bg_alt      = _g("bg_secondary", "#1a1a1a")
            self.c_bg_sidebar  = _g("bg_secondary", "#1a1a1a")
            self.c_sidebar_sel = _g("dialog_selection", "#2b3138")
            self.c_sidebar_txt = _g("text_secondary", "#aaaaaa")
            self.c_sidebar_txt_sel = _g("text_primary", "#e0e0e0")
            self.c_border      = _g("border_light", "#3a3a3a")
            self.c_text        = _g("text_primary", "#e0e0e0")
            self.c_text_muted  = _g("text_secondary", "#aaaaaa")
            self.c_accent      = _g("accent", "#007acc")
            self.c_primary     = _g("dialog_primary_bg", "#30363c")
            self.c_primary_hover = _g("dialog_primary_hover", "#394148")
            self.c_primary_border = _g("dialog_primary_border", "#505c66")
            self.c_primary_text = _g("dialog_primary_text", "#f5f7fa")
            self.c_table_sel   = _g("dialog_selection", "#2b3138")
            self.c_hover       = _g("bg_button_hover", "#3c3c3c")

    def _apply_style(self):
        self._applied_theme = tuple(sorted((key, value) for key, value in vars(self).items()
                                          if key.startswith('c_')))
        primary = getattr(self, "c_primary", self.c_accent)
        primary_hover = getattr(self, "c_primary_hover", self.c_accent)
        primary_border = getattr(self, "c_primary_border", self.c_border)
        primary_text = getattr(self, "c_primary_text", "#ffffff")
        table_sel = getattr(self, "c_table_sel", self.c_bg_alt)
        hover_bg = getattr(self, "c_hover", self.c_bg_alt)
        self._title_bar.setStyleSheet(
            f"background:{self.c_bg_alt}; color:{self.c_text}; font-weight:bold;"
            f"font-size:11pt; padding-left:10px; border-bottom:1px solid {self.c_border};"
        )
        self.category_list.setStyleSheet(f"""
            QListWidget#gsdSidebar {{
                background:{self.c_bg_sidebar};
                border:none; outline:none;
                font-size:9.5pt;
            }}
            QListWidget#gsdSidebar::item {{
                color:{self.c_sidebar_txt};
                padding:6px 4px;
                border-left:3px solid transparent;
            }}
            QListWidget#gsdSidebar::item:selected {{
                background:{self.c_sidebar_sel};
                color:{self.c_sidebar_txt_sel};
                border-left:3px solid {self.c_accent};
                font-weight:bold;
            }}
            QListWidget#gsdSidebar::item:hover:!selected {{
                background:{hover_bg};
            }}
        """)
        self.setStyleSheet(f"""
            QDialog {{ background:{self.c_bg}; }}
            QWidget {{ background:{self.c_bg}; color:{self.c_text}; }}
            QGroupBox {{
                font-size:9.5pt; font-weight:bold;
                border:1px solid {self.c_border}; border-radius:4px;
                margin-top:10px; color:{self.c_text};
            }}
            QGroupBox::title {{
                subcontrol-origin:margin; left:10px; padding:0 4px;
            }}
            QLabel {{ background:transparent; color:{self.c_text}; }}
            QLabel#dialogCaption {{ color:{self.c_text_muted}; font-size:8.5pt; }}
            QComboBox, QDoubleSpinBox, QLineEdit {{
                font-size:9pt; min-height:22px; padding:2px 6px;
                border:1px solid {self.c_border}; border-radius:3px;
                background:{self.c_bg}; color:{self.c_text};
            }}
            QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
                border-color:{self.c_accent};
            }}
            QSlider::groove:horizontal {{
                height:4px; background:{self.c_border}; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                background:{self.c_accent}; width:14px; height:14px;
                margin:-5px 0; border-radius:7px;
            }}
            QPushButton {{
                font-size:9pt; padding:3px 12px; min-height:24px;
                border:1px solid {self.c_border}; border-radius:3px;
                background:{self.c_bg_alt}; color:{self.c_text};
            }}
            QPushButton:hover {{ background:{hover_bg}; }}
            QPushButton:disabled {{ color:{self.c_text_muted}; }}
            QPushButton#primaryBtn {{
                background:{primary}; color:{primary_text};
                border:1px solid {primary_border}; font-weight:bold;
            }}
            QPushButton#primaryBtn:hover {{ background:{primary_hover}; }}
            QPushButton#primaryBtn:disabled {{
                background:{self.c_bg_alt}; color:{self.c_text_muted};
                border:1px solid {self.c_border};
            }}
            QPushButton#secondaryBtn {{
                background:{self.c_bg_alt}; color:{self.c_text};
            }}
            QTableWidget {{
                background:{self.c_bg}; alternate-background-color:{self.c_bg_alt};
                color:{self.c_text}; gridline-color:transparent;
                font-size:9pt; border:none; outline:none;
            }}
            QTableWidget::item:selected {{
                background:{table_sel}; color:{self.c_text};
            }}
            QHeaderView::section {{
                background:{self.c_bg_alt}; color:{self.c_text_muted};
                border:none; border-bottom:1px solid {self.c_border};
                padding:2px 4px; font-size:8.5pt;
            }}
            QFrame[frameShape="4"], QFrame[frameShape="5"] {{
                color:{self.c_border}; background:{self.c_border}; max-height:1px;
            }}
            QScrollBar:vertical {{
                width:8px; background:{self.c_bg_alt};
            }}
            QScrollBar::handle:vertical {{
                background:{self.c_border}; border-radius:4px; min-height:30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

    def refresh_theme(self):
        self._init_colors()
        signature = tuple(sorted((key, value) for key, value in vars(self).items()
                                 if key.startswith('c_')))
        if hasattr(self, "_title_bar") and signature != getattr(self, '_applied_theme', None):
            self._apply_style()

    def showEvent(self, event):
        self.refresh_theme()
        self._dirty_categories.clear()
        self._loaded_pages.clear()
        self._load_values()
        super().showEvent(event)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar
        self._title_bar = QLabel("  Global Settings")
        self._title_bar.setFixedHeight(28)
        root.addWidget(self._title_bar)

        # ── Body: sidebar | stack
        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(0)
        root.addWidget(body, 1)

        # Sidebar
        self.category_list = QListWidget()
        self.category_list.setObjectName("gsdSidebar")
        self.category_list.setFixedWidth(160)
        for label in ("Theme", "Navigation", "Updates"):
            item = QListWidgetItem(f"  {label}")
            item.setSizeHint(QSize(160, 32))
            self.category_list.addItem(item)
        self.category_list.currentRowChanged.connect(self._on_category_changed)
        body.addWidget(self.category_list)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.category_stack = QStackedWidget()
        self._page_builders = (
            self._build_theme_page,
            self._build_navigation_page, self._build_updates_page,
        )
        for _ in self._page_builders:
            self.category_stack.addWidget(QWidget())
        right_layout.addWidget(self.category_stack, 1)

        # Footer separator + bar
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        right_layout.addWidget(sep)

        footer_widget = QWidget()
        footer_widget.setFixedHeight(46)
        footer_layout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(14, 10, 14, 10)

        self.footer_note = QLabel("Save All applies only changed categories.")
        self.footer_note.setObjectName("dialogCaption")
        footer_layout.addWidget(self.footer_note)
        footer_layout.addStretch()

        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setObjectName("primaryBtn")
        self.save_all_btn.setFixedWidth(82)
        self.save_all_btn.clicked.connect(self.save_all_settings)
        footer_layout.addWidget(self.save_all_btn)

        right_layout.addWidget(footer_widget)
        body.addWidget(right)
        body.setSizes([160, 740])

        self._apply_style()
        self.category_list.setCurrentRow(0)

    def _build_theme_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        group = QGroupBox("Theme")
        form = QFormLayout(group)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        form.addRow("Application theme:", self.theme_combo)

        self.theme_summary = QLabel()
        self.theme_summary.setObjectName("dialogCaption")
        self.theme_summary.setWordWrap(True)

        layout.addWidget(group)
        layout.addWidget(self.theme_summary)
        layout.addStretch()
        return page

    def _build_navigation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        group = QGroupBox("Navigation")
        form = QFormLayout(group)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.zoom_behavior_combo = QComboBox()
        self.zoom_behavior_combo.addItem("Zoom to Center", "center")
        self.zoom_behavior_combo.addItem("Zoom to Cursor", "cursor")
        self.zoom_behavior_combo.addItem("Zoom to Picked Point (MicroStation-like)", "picked_point")
        self.zoom_behavior_combo.currentIndexChanged.connect(self._update_navigation_summary)
        form.addRow("Mouse wheel zoom:", self.zoom_behavior_combo)

        note = QLabel(
            "Center keeps the current behavior. Cursor follows the live mouse position. "
            "Picked Point behaves closer to MicroStation zoom-and-recenter: click a model point first, then wheel zoom keeps bringing that point to the view center."
        )
        note.setObjectName("dialogCaption")
        note.setWordWrap(True)
        form.addRow("", note)

        # Panning button setting
        self.panning_button_combo = QComboBox()
        self.panning_button_combo.addItem("Scroll Button (Middle Click)", "scroll")
        self.panning_button_combo.addItem("Left Mouse Button", "left")
        self.panning_button_combo.addItem("Tap-Tap Pan", "tap")
        self.panning_button_combo.currentIndexChanged.connect(self._update_navigation_summary)
        form.addRow("Panning button:", self.panning_button_combo)

        note_pan = QLabel(
            "Scroll Button uses the scroll wheel press for panning. "
            "Left Mouse Button pans by left-dragging (MicroStation Pan-tool style). "
            "Tap-Tap Pan pans like MicroStation's dynamic pan: tap once to start "
            "panning, move the mouse without holding any button, and tap again to stop. "
            "Pan stays disabled while another tool owns the left click."
        )
        note_pan.setObjectName("dialogCaption")
        note_pan.setWordWrap(True)
        form.addRow("", note_pan)

        self.navigation_summary = QLabel()
        self.navigation_summary.setObjectName("dialogCaption")
        self.navigation_summary.setWordWrap(True)

        layout.addWidget(group)
        layout.addWidget(self.navigation_summary)
        layout.addStretch()
        return page

    def _build_updates_page(self):
        """Build the presentation-only shell for the future updater."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        device_group = QGroupBox("This Device")
        device_form = QFormLayout(device_group)
        device_form.setHorizontalSpacing(12)
        device_form.setVerticalSpacing(8)

        self.update_system_number = QLabel("Not registered")
        self.update_device_id = QLabel("Not available")
        self.update_device_status = QLabel("Update service not configured")
        self.update_device_status.setObjectName("dialogCaption")

        device_form.addRow("System number:", self.update_system_number)
        device_form.addRow("Device ID:", self.update_device_id)
        device_form.addRow("Status:", self.update_device_status)

        preferences_group = QGroupBox("Update Preferences")
        preferences_layout = QVBoxLayout(preferences_group)
        preferences_layout.setSpacing(8)

        self.auto_check_updates_cb = QCheckBox("Automatically check for updates")
        self.auto_download_updates_cb = QCheckBox(
            "Automatically download available updates"
        )
        preferences_layout.addWidget(self.auto_check_updates_cb)
        preferences_layout.addWidget(self.auto_download_updates_cb)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Update channel:"))
        self.update_channel_combo = QComboBox()
        self.update_channel_combo.addItem("Stable", "stable")
        self.update_channel_combo.addItem("Testing", "testing")
        self.update_channel_combo.setMinimumWidth(180)
        channel_row.addWidget(self.update_channel_combo)
        channel_row.addStretch()
        preferences_layout.addLayout(channel_row)

        update_group = QGroupBox("Software Update")
        update_layout = QVBoxLayout(update_group)
        update_layout.setSpacing(9)

        version_form = QFormLayout()
        version_form.setHorizontalSpacing(12)
        version_form.setVerticalSpacing(7)
        self.current_version_label = QLabel("Version information unavailable")
        self.available_version_label = QLabel("Not available")
        self.last_checked_label = QLabel("Never")
        version_form.addRow("Current version:", self.current_version_label)
        version_form.addRow("Available version:", self.available_version_label)
        version_form.addRow("Last checked:", self.last_checked_label)
        update_layout.addLayout(version_form)

        self.update_status_label = QLabel(
            "Automatic updates are not configured yet. No update actions are active."
        )
        self.update_status_label.setObjectName("dialogCaption")
        self.update_status_label.setWordWrap(True)
        update_layout.addWidget(self.update_status_label)

        self.update_progress = QProgressBar()
        self.update_progress.setRange(0, 100)
        self.update_progress.setValue(0)
        self.update_progress.setTextVisible(True)
        self.update_progress.setFormat("No download in progress")
        update_layout.addWidget(self.update_progress)

        release_notes_title = QLabel("Release notes")
        release_notes_title.setStyleSheet("font-weight:bold;")
        update_layout.addWidget(release_notes_title)

        self.update_release_notes = QLabel(
            "Release details will appear here when an update is available for this system."
        )
        self.update_release_notes.setObjectName("dialogCaption")
        self.update_release_notes.setWordWrap(True)
        self.update_release_notes.setMinimumHeight(36)
        update_layout.addWidget(self.update_release_notes)

        actions = QHBoxLayout()
        self.check_updates_btn = QPushButton("Check for Updates")
        self.download_update_btn = QPushButton("Download Update")
        self.install_restart_btn = QPushButton("Install and Restart")
        self.install_restart_btn.setObjectName("primaryBtn")
        for button in (self.download_update_btn, self.install_restart_btn):
            button.setEnabled(False)
            actions.addWidget(button)
        self.check_updates_btn.clicked.connect(self._check_for_updates)
        self.download_update_btn.clicked.connect(self._download_update)
        self.install_restart_btn.clicked.connect(self._install_update)
        actions.insertWidget(0, self.check_updates_btn)
        actions.addStretch()
        update_layout.addLayout(actions)

        layout.addWidget(device_group)
        layout.addWidget(preferences_group)
        layout.addWidget(update_group)
        layout.addStretch()
        return page

    def _on_category_changed(self, index):
        if index < 0:
            return
        if index not in self._built_pages:
            placeholder = self.category_stack.widget(index)
            page = self._page_builders[index]()
            self.category_stack.removeWidget(placeholder)
            placeholder.deleteLater()
            self.category_stack.insertWidget(index, page)
            self._built_pages.add(index)
            self._connect_dirty_signals(index)
        self.category_stack.setCurrentIndex(index)
        self._load_category(index)
        if index == 2:
            self._refresh_update_page()

    def _load_category(self, index):
        if index in self._loaded_pages:
            return
        loaders = (self._load_theme_settings,
                   self._load_navigation_settings, self._load_update_settings)
        self._loading = True
        try:
            loaders[index]()
            self._loaded_pages.add(index)
        finally:
            self._loading = False

    def _mark_dirty(self, category):
        if not self._loading:
            self._dirty_categories.add(category)

    def _connect_dirty_signals(self, category):
        names = {
            0: ('theme_combo',),
            1: ('zoom_behavior_combo', 'panning_button_combo'),
            2: ('auto_check_updates_cb', 'auto_download_updates_cb', 'update_channel_combo'),
        }
        for name in names[category]:
            widget = getattr(self, name)
            if isinstance(widget, QComboBox):
                signal = widget.currentIndexChanged
            elif isinstance(widget, QCheckBox):
                signal = widget.toggled
            else:
                signal = widget.valueChanged
            signal.connect(lambda _, c=category: self._mark_dirty(c))

    def _load_values(self):
        self._on_category_changed(max(0, self.category_list.currentRow()))

    def _load_theme_settings(self):
        from gui.theme_manager import ThemeManager

        theme_name = ThemeManager.current() or ThemeManager.load_saved_theme()
        index = self.theme_combo.findData(theme_name)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)

        self.theme_summary.setText(
            f"Current theme is {self.theme_combo.currentText().lower()}."
        )

    def _load_navigation_settings(self):
        zoom_behavior = self.settings.value(
            "view_zoom_behavior",
            getattr(self.app, "zoom_behavior", "center"),
            type=str,
        )
        index = self.zoom_behavior_combo.findData(zoom_behavior)
        if index < 0:
            index = 0
        self.zoom_behavior_combo.setCurrentIndex(index)

        panning_button = self.settings.value(
            "panning_button",
            getattr(self.app, "panning_button", "scroll"),
            type=str,
        )
        index = self.panning_button_combo.findData(panning_button)
        if index < 0:
            index = 0
        self.panning_button_combo.setCurrentIndex(index)
        self._update_navigation_summary()

    def _load_update_settings(self):
        auto_check = self.settings.value("updates/auto_check", True, type=bool)
        auto_download = self.settings.value(
            "updates/auto_download", False, type=bool
        )
        self.auto_check_updates_cb.setChecked(auto_check)
        self.auto_download_updates_cb.setChecked(auto_download)
        channel_index = self.update_channel_combo.findData(self.update_manager.channel)
        self.update_channel_combo.setCurrentIndex(max(0, channel_index))
        self._refresh_update_page()

    def _refresh_update_page(self):
        self.current_version_label.setText(APP_VERSION)
        self.update_device_id.setText(self.update_manager.device_id)
        self.update_system_number.setText(
            self.update_manager.system_number or "Not assigned"
        )
        configured = self.update_manager.configured
        self.update_device_status.setText(
            "Ready" if configured else "Update service credentials not configured"
        )
        self.check_updates_btn.setEnabled(configured and not self._worker_running())
        self.check_updates_btn.setToolTip(
            "" if configured else "Configure the update URL and per-device token."
        )

    def _worker_running(self):
        return self._update_worker is not None and self._update_worker.isRunning()

    def _start_update_worker(self, operation, info=None):
        if self._worker_running():
            return
        self._update_worker = UpdateWorker(
            self.update_manager, operation, info, self
        )
        self._update_worker.failed.connect(self._update_operation_failed)
        self._update_worker.finished.connect(self._update_worker_finished)
        if operation == "check":
            self._update_worker.succeeded.connect(self._update_check_finished)
            self.update_status_label.setText("Checking for updates...")
        else:
            self._update_worker.succeeded.connect(self._update_download_finished)
            self._update_worker.progress_changed.connect(
                self._update_download_progress
            )
            self.update_progress.setValue(0)
            self.update_progress.setFormat("Downloading: %p%")
            self.update_status_label.setText("Downloading and verifying update...")
        self.check_updates_btn.setEnabled(False)
        self.download_update_btn.setEnabled(False)
        self._update_worker.start()

    def _check_for_updates(self):
        self._start_update_worker("check")

    def _update_check_finished(self, info):
        from datetime import datetime

        self.last_checked_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.settings.setValue("updates/last_checked", self.last_checked_label.text())
        if info is None:
            self._available_update = None
            self.available_version_label.setText("Up to date")
            self.update_release_notes.setText("No newer update is assigned to this system.")
            self.update_status_label.setText("This application is up to date.")
            return
        self._available_update = info
        self.available_version_label.setText(info.version)
        self.update_release_notes.setText(info.release_notes)
        self.update_status_label.setText(
            f"Version {info.version} is available for this system."
        )
        self.download_update_btn.setEnabled(True)
        if self.auto_download_updates_cb.isChecked():
            self._pending_auto_download = True

    def _download_update(self):
        if self._available_update is not None:
            self._start_update_worker("download", self._available_update)

    def _update_download_progress(self, value):
        self.update_progress.setValue(value)

    def _update_download_finished(self, installer):
        self._downloaded_update = installer
        self.update_progress.setValue(100)
        self.update_progress.setFormat("Download verified and ready")
        self.update_status_label.setText(
            "The signed update is ready to install. Save your work before continuing."
        )
        self.install_restart_btn.setEnabled(True)

    def _update_operation_failed(self, message):
        self.update_status_label.setText(message)
        self.update_progress.setFormat("Update operation failed")
        QMessageBox.warning(self, "Update Failed", message)

    def _update_worker_finished(self):
        worker = self._update_worker
        self._update_worker = None
        if worker is not None:
            worker.deleteLater()
        self._refresh_update_page()
        self.download_update_btn.setEnabled(self._available_update is not None)
        self.install_restart_btn.setEnabled(self._downloaded_update is not None)
        if self._pending_auto_download and self._available_update is not None:
            self._pending_auto_download = False
            self._download_update()

    def _install_update(self):
        if self._downloaded_update is None:
            return
        answer = QMessageBox.question(
            self,
            "Install Update",
            "Save your open work now. The application will close and Windows "
            "will request permission to install the verified update. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            if self.app is not None and not self.app.close():
                return
            self.update_manager.launch_installer(self._downloaded_update)
            QApplication.quit()
        except Exception as exc:
            QMessageBox.warning(self, "Installation Failed", str(exc))

    def _format_class_value(self, value):
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    def _update_navigation_summary(self):
        mode = self.zoom_behavior_combo.currentData() or "center"
        if mode == "cursor":
            zoom_text = "Mouse wheel zoom will stay anchored under the cursor in supported views."
        elif mode == "picked_point":
            zoom_text = "Click a point to set the zoom target, then mouse wheel zoom will recenter and zoom around that picked point."
        else:
            zoom_text = "Mouse wheel zoom will continue using the viewport center."

        panning = self.panning_button_combo.currentData() or "scroll"
        if panning == "left":
            panning_text = "Left-drag pans in the main view (when no tool owns left click); middle-click drag also pans."
        elif panning == "tap":
            panning_text = "Tap once to start panning, move to pan without holding, tap again to stop (MicroStation dynamic pan)."
        else:
            panning_text = "Scroll Button (Middle Click) is used for panning."

        self.navigation_summary.setText(f"{zoom_text}\n{panning_text}")

    def _save_theme_settings(self):
        from gui.theme_manager import ThemeManager

        theme_name = self.theme_combo.currentData()
        ThemeManager.apply_theme(self.app, theme_name)
        self.refresh_theme()
        if hasattr(self.app, "_update_theme_icon"):
            self.app._update_theme_icon()
        if hasattr(self.app, "_update_settings_icon"):
            self.app._update_settings_icon()

    def _save_navigation_settings(self):
        zoom_behavior = self.zoom_behavior_combo.currentData() or "center"
        self.settings.setValue("view_zoom_behavior", zoom_behavior)
        self.app.zoom_behavior = zoom_behavior

        panning_button = self.panning_button_combo.currentData() or "scroll"
        self.settings.setValue("panning_button", panning_button)
        self.app.panning_button = panning_button
        # An explicit settings save re-arms Left/Tap-Tap pan if a previous
        # Escape had suspended it (the combo is the user's chosen mode).
        if hasattr(self.app, "_pan_nav_suspended_button"):
            if getattr(self.app, "_left_pan_shortcut_active", False):
                # A Pan-tool shortcut is currently armed with a temporary
                # mode override. Keep the suspension slot so disarming
                # restores THIS freshly-saved value, not a stale one.
                self.app._pan_nav_suspended_button = panning_button
            else:
                self.app._pan_nav_suspended_button = None

    def _save_update_settings(self):
        self.update_manager.set_preferences(
            self.auto_check_updates_cb.isChecked(),
            self.auto_download_updates_cb.isChecked(),
            self.update_channel_combo.currentData() or "stable",
        )

    def save_all_settings(self):
        try:
            if not self._dirty_categories:
                if hasattr(self.app, "statusBar") and self.app.statusBar():
                    self.app.statusBar().showMessage("No changes to save.", 2500)
                return

            if 0 in self._dirty_categories:
                self._save_theme_settings()
            if 1 in self._dirty_categories:
                self._save_navigation_settings()
            if 2 in self._dirty_categories:
                self._save_update_settings()

            self.settings.sync()
            self._dirty_categories.clear()

            for index, refresh in ((0, self._load_theme_settings),
                                   (1, self._update_navigation_summary)):
                if index in self._built_pages:
                    refresh()

            if hasattr(self.app, "statusBar") and self.app.statusBar():
                self.app.statusBar().showMessage("Settings saved successfully.", 2500)
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))

    def refresh_summaries(self):
        """Compatibility hook used by the main window before showing the dialog."""
        self._load_values()
