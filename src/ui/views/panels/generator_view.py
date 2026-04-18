from typing import Any, Dict
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, 
                               QLineEdit, QPushButton, QCheckBox)
from PySide6.QtCore import Qt

from src.app.config import DEFAULT_UI_MARGIN, DEFAULT_UI_SPACING

class GeneratorDialogView(QDialog):
    """
    Floating UI Dialog for configuring and initiating the Synthetic Data Batch Generation.
    Replaces the legacy dockable panel to prevent cluttering the main editor interface.
    """
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self.setWindowTitle("Synthetic Data Generator")
        self.setMinimumWidth(480)
        
        # Ensure the dialog floats above the main window
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setup_ui()

    def setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        self.main_layout.setSpacing(DEFAULT_UI_SPACING)

        self._build_config_group()
        self._build_action_group()

    def _build_config_group(self) -> None:
        group = QGroupBox("Generation Settings")
        form = QFormLayout(group)
        form.setContentsMargins(DEFAULT_UI_MARGIN, 10, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        form.setSpacing(DEFAULT_UI_SPACING)

        # 1. Output Path (Now Optional)
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        
        self.txt_dir = QLineEdit()
        self.txt_dir.setPlaceholderText("Leave empty for default (datasets/export_...)")
        self.txt_dir.setReadOnly(True)
        
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self._request_browse)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(lambda: self.txt_dir.clear())
        
        path_layout.addWidget(self.txt_dir)
        path_layout.addWidget(self.btn_browse)
        path_layout.addWidget(self.btn_clear)
        form.addRow("Output Path:", path_layout)

        # 2. Resolution (Width x Height)
        res_layout = QHBoxLayout()
        res_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spn_w = QSpinBox()
        self.spn_w.setRange(64, 4096)
        self.spn_w.setValue(640)
        
        self.spn_h = QSpinBox()
        self.spn_h.setRange(64, 4096)
        self.spn_h.setValue(640)
        
        res_layout.addWidget(QLabel("W:"))
        res_layout.addWidget(self.spn_w)
        res_layout.addWidget(QLabel("H:"))
        res_layout.addWidget(self.spn_h)
        form.addRow("Resolution:", res_layout)

        # 3. Framerate
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 240)
        self.spn_fps.setValue(30)
        form.addRow("Framerate (FPS):", self.spn_fps)

        # 4. Duration
        dur_layout = QHBoxLayout()
        dur_layout.setContentsMargins(0, 0, 0, 0)

        self.spn_duration = QDoubleSpinBox()
        self.spn_duration.setRange(0.1, 3600.0)
        self.spn_duration.setSingleStep(0.5)
        self.spn_duration.setValue(5.0)
        self.spn_duration.setDecimals(2)

        self.btn_auto_dur = QPushButton("Auto-Detect")
        self.btn_auto_dur.setCursor(Qt.PointingHandCursor)
        self.btn_auto_dur.clicked.connect(self._request_auto_duration)

        dur_layout.addWidget(self.spn_duration)
        dur_layout.addWidget(self.btn_auto_dur)
        form.addRow("Duration (s):", dur_layout)

        # 5. Domain Randomization [NEW]
        self.chk_randomize = QCheckBox("Enable Domain Randomization (Lighting & Jitter)")
        self.chk_randomize.setChecked(True)
        self.chk_randomize.setToolTip("Adds noise to lighting and camera angles to prevent AI Overfitting.")
        form.addRow("", self.chk_randomize)

        self.main_layout.addWidget(group)

    def _build_action_group(self) -> None:
        self.btn_start = QPushButton("START GENERATION")
        self.btn_start.setMinimumHeight(35)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; color: #d3d3d3; }
        """)
        self.btn_start.clicked.connect(self._request_generation)
        self.main_layout.addWidget(self.btn_start)

    def _request_browse(self) -> None:
        if self._controller:
            self._controller.handle_browse_directory()

    def _request_auto_duration(self) -> None:
        if self._controller:
            self._controller.handle_auto_duration()

    def _request_generation(self) -> None:
        if self._controller:
            self._controller.handle_start_generation()

    def get_settings(self) -> Dict[str, Any]:
        fps = self.spn_fps.value()
        duration = self.spn_duration.value()
        return {
            "output_dir": self.txt_dir.text().strip(),
            "res_w": self.spn_w.value(),
            "res_h": self.spn_h.value(),
            "num_frames": int(duration * fps),
            "dt": 1.0 / fps,
            "use_randomization": self.chk_randomize.isChecked()
        }
        
    def set_directory(self, path: str) -> None:
        self.txt_dir.setText(path)

    def set_duration(self, duration: float) -> None:
        self.spn_duration.setValue(duration)
        
    def set_ui_locked(self, locked: bool) -> None:
        self.btn_browse.setEnabled(not locked)
        self.btn_clear.setEnabled(not locked)
        self.spn_fps.setEnabled(not locked)
        self.spn_w.setEnabled(not locked)
        self.spn_h.setEnabled(not locked)
        self.spn_duration.setEnabled(not locked)
        self.btn_auto_dur.setEnabled(not locked)
        self.chk_randomize.setEnabled(not locked)
        self.btn_start.setEnabled(not locked)
        
        if locked:
            self.btn_start.setText("GENERATING... PLEASE WAIT")
            self.btn_start.setStyleSheet("QPushButton { background-color: #f0ad4e; color: black; font-weight: bold; border-radius: 4px; }")
        else:
            self.btn_start.setText("START GENERATION")
            self.btn_start.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #218838; }")