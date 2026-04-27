from typing import Any, Dict
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, 
                               QLineEdit, QPushButton, QCheckBox, QProgressBar, QComboBox)
from PySide6.QtCore import Qt

from src.app.config import DEFAULT_UI_MARGIN, DEFAULT_UI_SPACING

class GeneratorPanelView(QWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self.setMinimumWidth(320)
        self.setMaximumWidth(350)
        self.setup_ui()

    def setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        self.main_layout.setSpacing(DEFAULT_UI_SPACING)

        self._build_config_group()
        self._build_preview_hint()
        self._build_action_group()
        self.main_layout.addStretch()

    def _build_config_group(self) -> None:
        group = QGroupBox("Generation Settings")
        form = QFormLayout(group)
        form.setContentsMargins(DEFAULT_UI_MARGIN, 10, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        form.setSpacing(DEFAULT_UI_SPACING)

        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_dir = QLineEdit()
        self.txt_dir.setPlaceholderText("Leave empty for default")
        self.txt_dir.setReadOnly(True)
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self._request_browse)
        self.btn_clear = QPushButton("X")
        self.btn_clear.setFixedWidth(30)
        self.btn_clear.clicked.connect(self.txt_dir.clear)
        
        path_layout.addWidget(self.txt_dir)
        path_layout.addWidget(self.btn_browse)
        path_layout.addWidget(self.btn_clear)
        form.addRow("Output:", path_layout)

        res_layout = QHBoxLayout()
        res_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cmb_res_presets = QComboBox()
        self.cmb_res_presets.addItems(["Custom", "640x640", "1024x1024", "1280x720", "1920x1080", "2560x1440"])
        self.cmb_res_presets.setCurrentIndex(2)
        self.cmb_res_presets.currentTextChanged.connect(self._on_preset_changed)
        
        self.spn_w = QSpinBox()
        self.spn_w.setRange(64, 8192)
        self.spn_w.setValue(1024)
        self.spn_w.setKeyboardTracking(False)
        self.spn_w.valueChanged.connect(self._on_spinbox_changed)
        
        self.spn_h = QSpinBox()
        self.spn_h.setRange(64, 8192)
        self.spn_h.setValue(1024)
        self.spn_h.setKeyboardTracking(False)
        self.spn_h.valueChanged.connect(self._on_spinbox_changed)
        
        res_layout.addWidget(self.cmb_res_presets)
        res_layout.addWidget(self.spn_w)
        res_layout.addWidget(QLabel("x"))
        res_layout.addWidget(self.spn_h)
        
        form.addRow("Resolution:", res_layout)

        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 240)
        self.spn_fps.setValue(24)
        form.addRow("FPS:", self.spn_fps)

        dur_layout = QHBoxLayout()
        dur_layout.setContentsMargins(0, 0, 0, 0)
        self.spn_duration = QDoubleSpinBox()
        self.spn_duration.setRange(0.1, 3600.0)
        self.spn_duration.setSingleStep(0.5)
        self.spn_duration.setValue(5.0)
        self.spn_duration.setDecimals(2)
        self.btn_auto_dur = QPushButton("Auto")
        self.btn_auto_dur.setMaximumWidth(40)
        self.btn_auto_dur.clicked.connect(self._request_auto_duration)
        dur_layout.addWidget(self.spn_duration)
        dur_layout.addWidget(self.btn_auto_dur)
        form.addRow("Duration (s):", dur_layout)

        self.chk_rand_light = QCheckBox("Randomize Light (Time of Day)")
        self.chk_rand_light.setChecked(True)
        form.addRow("", self.chk_rand_light)

        self.chk_rand_cam = QCheckBox("Randomize Camera (Rotation Jitter)")
        self.chk_rand_cam.setChecked(True)
        form.addRow("", self.chk_rand_cam)

        self.main_layout.addWidget(group)

    def _build_preview_hint(self) -> None:
        self.lbl_preview_status = QLabel("Ready.")
        self.lbl_preview_status.setStyleSheet("color: #888; font-style: italic;")
        self.lbl_preview_status.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.lbl_preview_status)

    def _build_action_group(self) -> None:
        action_layout = QVBoxLayout()
        self.btn_preview = QPushButton("▶ LIVE PREVIEW")
        self.btn_preview.setMinimumHeight(35)
        self.btn_preview.setStyleSheet("background-color: #2E5C8A; font-weight: bold; color: white; border-radius: 4px;")
        self.btn_preview.clicked.connect(self._toggle_preview)
        action_layout.addWidget(self.btn_preview)

        self.btn_start = QPushButton("START BATCH GENERATION")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #218838; }")
        self.btn_start.clicked.connect(self._request_generation)
        action_layout.addWidget(self.btn_start)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.hide()
        
        self.lbl_progress_status = QLabel("")
        self.lbl_progress_status.setAlignment(Qt.AlignCenter)
        self.lbl_progress_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self.lbl_progress_status.hide()
        
        action_layout.addWidget(self.progress_bar)
        action_layout.addWidget(self.lbl_progress_status)
        
        self.main_layout.addLayout(action_layout)

    def _on_preset_changed(self, text: str) -> None:
        if text == "Custom":
            return
        try:
            w_str, h_str = text.split('x')
            w, h = int(w_str), int(h_str)
            self.spn_w.blockSignals(True)
            self.spn_h.blockSignals(True)
            self.spn_w.setValue(w)
            self.spn_h.setValue(h)
            self.spn_w.blockSignals(False)
            self.spn_h.blockSignals(False)
            if hasattr(self._controller, 'handle_preview_once'):
                self._controller.handle_preview_once()
        except Exception:
            pass

    def _on_spinbox_changed(self, val: int) -> None:
        self.cmb_res_presets.blockSignals(True)
        self.cmb_res_presets.setCurrentIndex(0)
        self.cmb_res_presets.blockSignals(False)
        if hasattr(self._controller, 'handle_preview_once'):
            self._controller.handle_preview_once()

    def _request_browse(self) -> None: 
        self._controller.handle_browse_directory()

    def _request_auto_duration(self) -> None: 
        self._controller.handle_auto_duration()

    def _request_generation(self) -> None: 
        self._controller.handle_start_generation()

    def _toggle_preview(self) -> None: 
        self._controller.toggle_preview_playback()

    def get_settings(self) -> Dict[str, Any]:
        fps = self.spn_fps.value()
        duration = self.spn_duration.value()
        return {
            "output_dir": self.txt_dir.text().strip(),
            "res_w": self.spn_w.value(),
            "res_h": self.spn_h.value(),
            "num_frames": int(duration * fps),
            "dt": 1.0 / fps,
            "use_rand_light": self.chk_rand_light.isChecked(),
            "use_rand_cam": self.chk_rand_cam.isChecked()
        }

    def set_directory(self, path: str) -> None: 
        self.txt_dir.setText(path)

    def set_duration(self, duration: float) -> None: 
        self.spn_duration.setValue(duration)

    def set_status(self, text: str) -> None: 
        self.lbl_preview_status.setText(text)
    
    def set_progress(self, value: int, maximum: int, text: str) -> None:
        if maximum > 0:
            self.progress_bar.show()
            self.lbl_progress_status.show()
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
            self.lbl_progress_status.setText(text)
        else:
            self.progress_bar.hide()
            self.lbl_progress_status.hide()
    
    def set_preview_state(self, is_playing: bool) -> None:
        if is_playing:
            self.btn_preview.setText("■ STOP PREVIEW")
            self.btn_preview.setStyleSheet("background-color: #8A2E2E; font-weight: bold; color: white; border-radius: 4px;")
            self.btn_start.setEnabled(False)
        else:
            self.btn_preview.setText("▶ LIVE PREVIEW")
            self.btn_preview.setStyleSheet("background-color: #2E5C8A; font-weight: bold; color: white; border-radius: 4px;")
            self.btn_start.setEnabled(True)

    def set_ui_locked(self, locked: bool) -> None:
        self.btn_browse.setEnabled(not locked)
        self.btn_start.setEnabled(not locked)
        self.btn_preview.setEnabled(not locked)
        self.cmb_res_presets.setEnabled(not locked)
        self.spn_w.setEnabled(not locked)
        self.spn_h.setEnabled(not locked)
        self.chk_rand_light.setEnabled(not locked)
        self.chk_rand_cam.setEnabled(not locked)