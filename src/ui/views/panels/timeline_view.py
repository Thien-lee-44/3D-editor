from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QDoubleSpinBox, QLabel)
from PySide6.QtCore import Qt
from typing import List

from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.timeline.timeline_track import TimelineTrackWidget
from src.ui.widgets.timeline.ruler_widget import RulerWidget
from src.ui.widgets.timeline.track_header import TrackHeaderWidget
from src.app.config import DEFAULT_UI_MARGIN, DEFAULT_UI_SPACING

class TimelinePanelView(BasePanel):
    """
    Global Timeline Panel docked at the bottom of the editor.
    Features a professional NLE (Non-Linear Editor) layout with separated headers, rulers, and tracks.
    """
    PANEL_TITLE = "Animation Timeline"
    PANEL_DOCK_AREA = Qt.BottomDockWidgetArea
    PANEL_MIN_HEIGHT = 160

    def setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(DEFAULT_UI_MARGIN, 2, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        self.main_layout.setSpacing(DEFAULT_UI_SPACING)

        self._build_top_controls()
        self._build_nle_layout()

    def _build_top_controls(self) -> None:
        layout = QHBoxLayout()
        
        self.btn_rewind = QPushButton("|<")
        self.btn_rewind.setMaximumWidth(40)
        self.btn_rewind.setCursor(Qt.PointingHandCursor)
        
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.setStyleSheet("QPushButton:checked { background-color: #2b5797; color: white; }")
        
        layout.addWidget(self.btn_rewind)
        layout.addWidget(self.btn_play)
        
        self.btn_add_key = QPushButton("Set Key")
        self.btn_add_key.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_add_key.setCursor(Qt.PointingHandCursor)
        
        self.btn_clear_key = QPushButton("Clear Keys")
        self.btn_clear_key.setCursor(Qt.PointingHandCursor)
        
        layout.addWidget(self.btn_add_key)
        layout.addWidget(self.btn_clear_key)
        
        layout.addStretch()
        
        self.spn_time = QDoubleSpinBox()
        self.spn_time.setRange(0.0, 3600.0)
        self.spn_time.setDecimals(2)
        self.spn_time.setSuffix(" s")
        self.spn_time.setSingleStep(0.05)
        self.spn_time.setMinimumWidth(80)
        
        self.spn_max_time = QDoubleSpinBox()
        self.spn_max_time.setRange(0.1, 3600.0)
        self.spn_max_time.setDecimals(2)
        self.spn_max_time.setValue(10.0)
        self.spn_max_time.setSuffix(" s Max")
        self.spn_max_time.setMinimumWidth(80)
        
        layout.addWidget(self.spn_time)
        layout.addWidget(QLabel("/"))
        layout.addWidget(self.spn_max_time)
        
        self.main_layout.addLayout(layout)

        self.btn_play.clicked.connect(self._request_play_toggle)
        self.btn_rewind.clicked.connect(self._request_rewind)
        self.btn_add_key.clicked.connect(self._request_add_key)
        self.btn_clear_key.clicked.connect(self._request_clear_key)
        self.spn_time.valueChanged.connect(self._on_spin_changed)
        self.spn_max_time.valueChanged.connect(self._on_max_time_changed)

    def _build_nle_layout(self) -> None:
        nle_layout = QHBoxLayout()
        nle_layout.setSpacing(2)
        
        self.track_header = TrackHeaderWidget()
        
        right_container = QVBoxLayout()
        right_container.setSpacing(0)
        
        self.ruler = RulerWidget()
        self.track = TimelineTrackWidget()
        
        right_container.addWidget(self.ruler)
        right_container.addWidget(self.track, stretch=1)
        
        nle_layout.addWidget(self.track_header)
        nle_layout.addLayout(right_container, stretch=1)
        
        self.main_layout.addLayout(nle_layout, stretch=1)
        
        self.ruler.time_scrubbed.connect(self._on_scrubbed)
        self.track.time_scrubbed.connect(self._on_scrubbed)
        self.track.keyframe_selected.connect(self._on_kf_selected)
        self.track.keyframes_mutated.connect(self._on_kfs_mutated)

    # =========================================================================
    # DELEGATES
    # =========================================================================

    def _request_play_toggle(self) -> None:
        if self._controller:
            self._controller.toggle_playback(self.btn_play.isChecked())
            self.btn_play.setText("Pause" if self.btn_play.isChecked() else "Play")

    def _request_rewind(self) -> None:
        if self._controller:
            self._controller.rewind_timeline()

    def _request_add_key(self) -> None:
        if self._controller:
            self._controller.add_keyframe_at_current()

    def _request_clear_key(self) -> None:
        if self._controller:
            self._controller.clear_keyframes()

    def _on_scrubbed(self, val: float) -> None:
        if self._controller and not self._controller.is_updating_ui:
            self._controller.set_time(val)

    def _on_spin_changed(self, val: float) -> None:
        if self._controller and not self._controller.is_updating_ui:
            self._controller.set_time(val)

    def _on_max_time_changed(self, val: float) -> None:
        self.ruler.set_max_time(val)
        self.track.set_max_time(val)

    def _on_kfs_mutated(self, payload: dict) -> None:
        if self._controller:
            self._controller.mutate_keyframes(payload)

    def _on_kf_selected(self, index: int) -> None:
        if self._controller:
            self._controller.select_keyframe(index)

    # =========================================================================
    # PUBLIC UPDATERS
    # =========================================================================

    def deselect_keyframe_ui(self) -> None:
        self.track.selected_indices.clear()
        self.track.update()

    def update_ui_time(self, time_sec: float) -> None:
        if self._controller:
            self._controller.is_updating_ui = True
            
            max_limit = self.spn_max_time.value()
            if time_sec > max_limit:
                self.spn_max_time.setValue(time_sec)
                self.ruler.set_max_time(time_sec)
                self.track.set_max_time(time_sec)
                
            self.spn_time.setValue(time_sec)
            self.ruler.set_time(time_sec)
            self.track.set_time(time_sec)
            
            self._controller.is_updating_ui = False

    def update_keyframes_display(self, kf_times: List[float], entity_name: str = "") -> None:
        self.track.set_keyframes(kf_times)
        if entity_name:
            self.track_header.set_target_name(entity_name)