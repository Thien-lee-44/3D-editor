from typing import Any, Dict
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QCheckBox, QWidget, QFrame, QDoubleSpinBox
from PySide6.QtCore import Qt

from src.ui.widgets.inspector.base_widget import BaseComponentWidget

class AnimationWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__("Animation & Keyframes", controller)
        
        self.kf_frame = QFrame()
        kf_layout = QHBoxLayout(self.kf_frame)
        kf_layout.setContentsMargins(8, 6, 8, 6)
        
        self.lbl_kf_title = QLabel("KEYFRAME EDIT")
        self.spn_kf_time = QDoubleSpinBox()
        self.spn_kf_time.setStyleSheet("border: 1px solid #777; background: #222; border-radius: 2px; color: white;")
        self.spn_kf_time.setRange(0.0, 3600.0)
        self.spn_kf_time.setDecimals(2)
        self.spn_kf_time.setSuffix(" s")
        self.spn_kf_time.valueChanged.connect(self._on_kf_time_changed)
        
        kf_layout.addWidget(self.lbl_kf_title)
        kf_layout.addStretch()
        kf_layout.addWidget(QLabel("Time:"))
        kf_layout.addWidget(self.spn_kf_time)
        self.layout.addWidget(self.kf_frame)

        form = QFormLayout()
        form.setContentsMargins(0, 5, 0, 5)

        self.chk_loop = QCheckBox("Loop Playback")
        self.chk_loop.toggled.connect(self._on_loop_toggled)
        form.addRow("", self.chk_loop)
        
        self.lbl_duration = QLabel("Duration: 0.0s")
        self.lbl_duration.setStyleSheet("color: #888; font-style: italic;")
        form.addRow("Total Time:", self.lbl_duration)

        self.layout.addLayout(form)
        
        self.kf_container = QVBoxLayout()
        self.kf_container.setSpacing(2)
        self.kf_container.setContentsMargins(0, 5, 0, 0)
        self.layout.addLayout(self.kf_container)

    def _on_kf_time_changed(self, val: float) -> None:
        if self._controller and self.kf_frame.isVisible():
            if hasattr(self._controller, 'set_active_keyframe_time'):
                self._controller.set_active_keyframe_time(val)

    def update_data(self, data: Dict[str, Any]) -> None:
        kf_idx = data.get("active_keyframe_index", -1)
        
        if kf_idx >= 0:
            self.kf_frame.setStyleSheet("""
                QFrame { background-color: #4A2B00; border: 1px solid #FFA500; border-radius: 4px; }
                QLabel { border: none; background: transparent; color: white; font-weight: bold; }
            """)
            self.spn_kf_time.blockSignals(True)
            self.spn_kf_time.setValue(data.get("active_keyframe_time", 0.0))
            self.spn_kf_time.setEnabled(kf_idx > 0)
            self.spn_kf_time.blockSignals(False)
            self.lbl_kf_title.setText("BASE STATE EDIT" if kf_idx == 0 else f"KEYFRAME #{kf_idx} EDIT")
        else:
            self.kf_frame.setStyleSheet("""
                QFrame { background-color: #1A3A1A; border: 1px solid #4CAF50; border-radius: 4px; }
                QLabel { border: none; background: transparent; color: white; font-weight: bold; }
            """)
            self.spn_kf_time.blockSignals(True)
            self.spn_kf_time.setValue(data.get("active_keyframe_time", 0.0))
            self.spn_kf_time.setEnabled(False)
            self.spn_kf_time.blockSignals(False)
            self.lbl_kf_title.setText("AUTO-KEYING (NO FOCUS)")

        self.chk_loop.blockSignals(True)
        self.chk_loop.setChecked(data.get("loop", False))
        self.chk_loop.blockSignals(False)
        
        keyframes = data.get("keyframes", [])
        duration = keyframes[-1].get("time", 0.0) if keyframes else 0.0
        self.lbl_duration.setText(f"{duration:.2f} s")
        
        # [CHỐNG GIẬT UI]: Chỉ vẽ lại danh sách nút bấm nếu cấu trúc Keyframe bị thay đổi
        kf_signature = f"idx:{kf_idx}-" + "-".join([f"{k.get('time', 0.0):.2f}" for k in keyframes])
        if getattr(self, "_last_kf_sig", "") == kf_signature:
            return 
            
        self._last_kf_sig = kf_signature
        
        while self.kf_container.count():
            item = self.kf_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        is_active = (kf_idx == 0)
        prefix = "▶ " if is_active else "  "
        color = "#FFA500" if is_active else "#DDDDDD"
        weight = "bold" if is_active else "normal"
        
        btn_kf_base = QPushButton(f"{prefix}Base State [0]  ->  t = 0.00s")
        btn_kf_base.setStyleSheet(f"text-align: left; font-size: 11px; color: {color}; font-weight: {weight}; border: none; background: transparent;")
        btn_kf_base.setCursor(Qt.PointingHandCursor)
        btn_kf_base.clicked.connect(lambda checked=False, idx=0: self._select_kf(idx))
        
        btn_del = QPushButton("x")
        btn_del.setFixedWidth(20)
        btn_del.setFixedHeight(20)
        btn_del.setStyleSheet("color: #555555; font-weight: bold; border: 1px solid #555555; border-radius: 2px;")
        btn_del.setEnabled(False) 
        
        row.addWidget(btn_kf_base)
        row.addWidget(btn_del)
        
        w = QWidget()
        w.setLayout(row)
        self.kf_container.addWidget(w)
                
        for i, kf in enumerate(keyframes):
            ui_idx = i + 1
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            
            is_active = (ui_idx == kf_idx)
            prefix = "▶ " if is_active else "  "
            color = "#FFA500" if is_active else "#DDDDDD"
            weight = "bold" if is_active else "normal"
            
            btn_kf = QPushButton(f"{prefix}Keyframe [{ui_idx}]  ->  t = {kf.get('time', 0.0):.2f}s")
            btn_kf.setStyleSheet(f"text-align: left; font-size: 11px; color: {color}; font-weight: {weight}; border: none; background: transparent;")
            btn_kf.setCursor(Qt.PointingHandCursor)
            btn_kf.clicked.connect(lambda checked=False, idx=ui_idx: self._select_kf(idx))
            
            btn_del = QPushButton("x")
            btn_del.setFixedWidth(20)
            btn_del.setFixedHeight(20)
            btn_del.setStyleSheet("color: #ff4444; font-weight: bold; border: 1px solid #ff4444; border-radius: 2px;")
            btn_del.setCursor(Qt.PointingHandCursor)
            btn_del.clicked.connect(lambda checked=False, idx=ui_idx: self._remove_kf(idx))
            
            row.addWidget(btn_kf)
            row.addWidget(btn_del)
            
            w = QWidget()
            w.setLayout(row)
            self.kf_container.addWidget(w)
            
    def _select_kf(self, idx: int) -> None:
        if self._controller and hasattr(self._controller, 'select_keyframe_from_inspector'):
            self._controller.select_keyframe_from_inspector(idx)
            
    def _on_loop_toggled(self, checked: bool) -> None:
        if self._controller:
            self._controller.set_property("Animation", "loop", checked)
            
    def _remove_kf(self, idx: int) -> None:
        if self._controller:
            if hasattr(self._controller, 'remove_keyframe'):
                self._controller.remove_keyframe(idx)