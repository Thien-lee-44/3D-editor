from typing import Any, Dict
from PySide6.QtWidgets import (QFormLayout, QLabel, QPushButton, QHBoxLayout, 
                               QVBoxLayout, QCheckBox, QWidget, QFrame, 
                               QDoubleSpinBox, QComboBox, QMessageBox)
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
        
        self.spn_kf_time.setSingleStep(0.1)
        self.spn_kf_time.setKeyboardTracking(False)
        
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

        copy_group = QWidget()
        copy_layout = QVBoxLayout(copy_group)
        copy_layout.setContentsMargins(0, 15, 0, 0)
        
        lbl_copy_title = QLabel("DUPLICATE KEYFRAME RANGE")
        lbl_copy_title.setStyleSheet("font-weight: bold; color: #E0E0E0;")
        copy_layout.addWidget(lbl_copy_title)

        copy_form = QFormLayout()
        copy_form.setContentsMargins(0, 5, 0, 5)

        self.cmb_copy_start = QComboBox()
        self.cmb_copy_start.setToolTip("Select the starting keyframe to duplicate.")
        
        self.cmb_copy_end = QComboBox()
        self.cmb_copy_end.setToolTip("Select the ending keyframe to duplicate.")
        
        self.spn_copy_dst = QDoubleSpinBox()
        self.spn_copy_dst.setRange(0.0, 9999.0)
        self.spn_copy_dst.setDecimals(2)
        self.spn_copy_dst.setSingleStep(0.1)
        self.spn_copy_dst.setValue(1.0)
        self.spn_copy_dst.setSuffix(" s")
        
        copy_form.addRow("From KF:", self.cmb_copy_start)
        copy_form.addRow("To KF:", self.cmb_copy_end)
        copy_form.addRow("Insert At:", self.spn_copy_dst)
        copy_layout.addLayout(copy_form)

        btn_apply_copy = QPushButton("Duplicate Range")
        btn_apply_copy.setStyleSheet("background-color: #2D5A27; color: white; border-radius: 3px; padding: 4px;")
        btn_apply_copy.clicked.connect(self.apply_copy_range)
        copy_layout.addWidget(btn_apply_copy)

        self.layout.addWidget(copy_group)

    def _on_kf_time_changed(self, val: float) -> None:
        if self._controller and self.kf_frame.isVisible():
            if hasattr(self._controller, 'set_active_keyframe_time'):
                self._controller.set_active_keyframe_time(val)

    def update_data(self, data: Dict[str, Any]) -> None:
        kf_idx = data.get("active_keyframe_index", -1)
        
        if kf_idx == 0:
            self.kf_frame.setStyleSheet("""
                QFrame { background-color: #4A2B00; border: 1px solid #FFA500; border-radius: 4px; }
                QLabel { border: none; background: transparent; color: white; font-weight: bold; }
            """)
            self.spn_kf_time.blockSignals(True)
            self.spn_kf_time.setValue(0.0)
            self.spn_kf_time.setEnabled(False)
            self.spn_kf_time.blockSignals(False)
            self.lbl_kf_title.setText("BASE STATE [0] EDIT")
            
        elif kf_idx > 0:
            self.kf_frame.setStyleSheet("""
                QFrame { background-color: #4A2B00; border: 1px solid #FFA500; border-radius: 4px; }
                QLabel { border: none; background: transparent; color: white; font-weight: bold; }
            """)
            self.spn_kf_time.blockSignals(True)
            self.spn_kf_time.setValue(data.get("active_keyframe_time", 0.0))
            self.spn_kf_time.setEnabled(True)
            self.spn_kf_time.blockSignals(False)
            self.lbl_kf_title.setText(f"KEYFRAME [{kf_idx}] EDIT")
            
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
        
        kf_signature = f"idx:{kf_idx}-" + "-".join([f"{k.get('time', 0.0):.2f}" for k in keyframes])
        if getattr(self, "_last_kf_sig", "") == kf_signature:
            return 
            
        self._last_kf_sig = kf_signature
        
        while self.kf_container.count():
            item = self.kf_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.cmb_copy_start.blockSignals(True)
        self.cmb_copy_end.blockSignals(True)
        self.cmb_copy_start.clear()
        self.cmb_copy_end.clear()
        
        if not keyframes:
            self.cmb_copy_start.addItem("No Keyframes", -1)
            self.cmb_copy_end.addItem("No Keyframes", -1)
                
        for i, kf in enumerate(keyframes):
            t = kf.get('time', 0.0)
            
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            
            is_active = (i == kf_idx)
            prefix = "▶ " if is_active else "  "
            color = "#FFA500" if is_active else "#DDDDDD"
            weight = "bold" if is_active else "normal"
            
            title = f"{prefix}Base State [0]  ->  t = 0.00s" if i == 0 else f"{prefix}Keyframe [{i}]  ->  t = {t:.2f}s"
            
            btn_kf = QPushButton(title)
            btn_kf.setStyleSheet(f"text-align: left; font-size: 11px; color: {color}; font-weight: {weight}; border: none; background: transparent;")
            btn_kf.setCursor(Qt.PointingHandCursor)
            btn_kf.clicked.connect(lambda checked=False, idx=i: self._select_kf(idx))
            
            btn_del = QPushButton("x")
            btn_del.setFixedWidth(20)
            btn_del.setFixedHeight(20)
            
            if i == 0:
                btn_del.setStyleSheet("color: #555555; font-weight: bold; border: 1px solid #555555; border-radius: 2px;")
                btn_del.setEnabled(False)
            else:
                btn_del.setStyleSheet("color: #ff4444; font-weight: bold; border: 1px solid #ff4444; border-radius: 2px;")
                btn_del.setCursor(Qt.PointingHandCursor)
                btn_del.clicked.connect(lambda checked=False, idx=i: self._remove_kf(idx))
            
            row.addWidget(btn_kf)
            row.addWidget(btn_del)
            
            w = QWidget()
            w.setLayout(row)
            self.kf_container.addWidget(w)

            label_text = f"Base [0] (0.00s)" if i == 0 else f"KF [{i}] ({t:.2f}s)"
            self.cmb_copy_start.addItem(label_text, i)
            self.cmb_copy_end.addItem(label_text, i)
                
        self.cmb_copy_start.blockSignals(False)
        self.cmb_copy_end.blockSignals(False)
            
    def _select_kf(self, idx: int) -> None:
        if self._controller and hasattr(self._controller, 'select_keyframe_from_inspector'):
            self._controller.select_keyframe_from_inspector(idx)
            
    def _on_loop_toggled(self, checked: bool) -> None:
        if self._controller:
            self._controller.set_properties("Animation", {"loop": checked})
            
    def _remove_kf(self, idx: int) -> None:
        if self._controller and hasattr(self._controller, 'remove_keyframe'):
            self._controller.remove_keyframe(idx)

    def apply_copy_range(self) -> None:
        if not self._controller: return
        
        start_idx = self.cmb_copy_start.currentData()
        end_idx = self.cmb_copy_end.currentData()
        
        if start_idx is None or end_idx is None or start_idx < 0 or end_idx < 0:
            QMessageBox.warning(self, "Invalid Selection", "Please select valid keyframes to duplicate.")
            return
            
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            
        dst_time = self.spn_copy_dst.value()
        
        self.request_undo_snapshot()
        
        # Đã bọc lệnh DUPLICATE_KEYFRAME_RANGE vào bên trong MUTATE_KEYFRAMES để Engine nhận diện đúng
        payload = {
            "MUTATE_KEYFRAMES": {
                "mode": "DUPLICATE_KEYFRAME_RANGE",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "target_time": dst_time
            }
        }
        self._controller.set_properties("Animation", payload)