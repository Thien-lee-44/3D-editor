import os
from typing import Any, Dict
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QCheckBox, 
                               QComboBox, QFormLayout, QGroupBox, QVBoxLayout, QPushButton, QFileDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from src.ui.widgets.custom_inputs import SliderSpinBox
from .base_widget import BaseComponentWidget, set_vec3_spinboxes, rgb_to_hex
from src.app import AppEvent

from src.app.config import (
    DEFAULT_MAT_AMB_STRENGTH, DEFAULT_MAT_DIFF_STRENGTH, DEFAULT_MAT_SPEC_STRENGTH, 
    DEFAULT_MAT_SHININESS, DEFAULT_MAT_OPACITY, DEFAULT_MAT_BASE_COLOR, 
    DEFAULT_MAT_AMBIENT, DEFAULT_MAT_DIFFUSE, DEFAULT_MAT_SPECULAR, DEFAULT_MAT_EMISSION,
    MAT_FACTOR_RANGE, MAT_SHININESS_RANGE, MAT_OPACITY_RANGE,
    TEX_THUMB_SIZE, STYLE_TEX_THUMB, STYLE_TEX_EMPTY, STYLE_TEX_LOADED
)

class MeshWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__("Mesh & Material", controller)
        self.tex_labels = {}
        self.pixmap_cache = {}
        
        self.chk_visible = QCheckBox("Visible")
        self.chk_visible.clicked.connect(self.request_undo_snapshot)
        self.chk_visible.toggled.connect(self.apply_mesh)
        self.layout.addWidget(self.chk_visible)
        
        # --- COLOR MODE ---
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Color Mode:"))
        self.cmb_mat_mode = QComboBox()
        self.cmb_mat_mode.addItems(["Basic (Base + Multipliers)", "Advanced (Independent)"])
        self.cmb_mat_mode.activated.connect(self.request_undo_snapshot) 
        self.cmb_mat_mode.currentIndexChanged.connect(self.switch_mat_mode)
        row_mode.addWidget(self.cmb_mat_mode)
        self.layout.addLayout(row_mode)

        self.w_mat_basic = QWidget()
        fb = QFormLayout(self.w_mat_basic)
        fb.setContentsMargins(0, 0, 0, 0)
        
        row_base, self.btn_mat_base, self.sp_mat_base_vec = self._build_color_row('base', self.apply_mat_vec_colors, self.pick_mat_color)
        fb.addRow("Base Color:", row_base)
        
        self.sp_mat_amb = SliderSpinBox(*MAT_FACTOR_RANGE, 0.05, DEFAULT_MAT_AMB_STRENGTH, self.apply_mesh, press_callback=self.request_undo_snapshot)
        self.sp_mat_diff = SliderSpinBox(*MAT_FACTOR_RANGE, 0.05, DEFAULT_MAT_DIFF_STRENGTH, self.apply_mesh, press_callback=self.request_undo_snapshot)
        self.sp_mat_spec = SliderSpinBox(*MAT_FACTOR_RANGE, 0.05, DEFAULT_MAT_SPEC_STRENGTH, self.apply_mesh, press_callback=self.request_undo_snapshot)
        fb.addRow("Ambient Factor:", self.sp_mat_amb)
        fb.addRow("Diffuse Factor:", self.sp_mat_diff)
        fb.addRow("Specular Factor:", self.sp_mat_spec)
        self.layout.addWidget(self.w_mat_basic)

        self.w_mat_adv = QWidget()
        fa = QFormLayout(self.w_mat_adv)
        fa.setContentsMargins(0, 0, 0, 0)
        
        row_amb, self.btn_mat_amb, self.sp_mat_amb_vec = self._build_color_row('amb', self.apply_mat_vec_colors, self.pick_mat_color)
        row_diff, self.btn_mat_diff, self.sp_mat_diff_vec = self._build_color_row('diff', self.apply_mat_vec_colors, self.pick_mat_color)
        row_spec, self.btn_mat_spec, self.sp_mat_spec_vec = self._build_color_row('spec', self.apply_mat_vec_colors, self.pick_mat_color)
        
        fa.addRow("Ambient Color:", row_amb)
        fa.addRow("Diffuse Color:", row_diff)
        fa.addRow("Specular Color:", row_spec)
        self.layout.addWidget(self.w_mat_adv)

        fc = QFormLayout()
        row_emis, self.btn_mat_emis, self.sp_mat_emis_vec = self._build_color_row('emis', self.apply_mat_vec_colors, self.pick_mat_color)
        fc.addRow("Emission:", row_emis)
        
        self.sp_shine = SliderSpinBox(*MAT_SHININESS_RANGE, 0.1, DEFAULT_MAT_SHININESS, self.apply_mesh, press_callback=self.request_undo_snapshot)
        self.sp_opacity = SliderSpinBox(*MAT_OPACITY_RANGE, 0.01, DEFAULT_MAT_OPACITY, self.apply_mesh, press_callback=self.request_undo_snapshot)
        fc.addRow("Shininess:", self.sp_shine)
        fc.addRow("Opacity:", self.sp_opacity)
        self.layout.addLayout(fc)

        self.group_tex = QGroupBox("Texture Maps")
        l_tex = QVBoxLayout(self.group_tex)
        
        map_types = [
            ("Diffuse", "map_diffuse"), ("Specular", "map_specular"), 
            ("Emission", "map_emission"), ("Ambient (AO)", "map_ambient"), 
            ("Shininess", "map_shininess"), ("Opacity", "map_opacity"), 
            ("Bump / Normal", "map_bump"), ("Reflection", "map_reflection")
        ]
        for label, attr in map_types:
            self._build_tex_slot(l_tex, label, attr)
            
        self.layout.addWidget(self.group_tex)

    def _build_tex_slot(self, parent_layout: QVBoxLayout, label_text: str, map_attr: str) -> None:
        row = QHBoxLayout()
        lbl = QLabel(f"{label_text}:")
        lbl.setMinimumWidth(110)
        
        lbl_thumb = QLabel()
        lbl_thumb.setFixedSize(TEX_THUMB_SIZE, TEX_THUMB_SIZE)
        lbl_thumb.setStyleSheet(STYLE_TEX_THUMB)
        lbl_thumb.setAlignment(Qt.AlignCenter)
        
        lbl_status = QLabel("Empty")
        lbl_status.setStyleSheet(STYLE_TEX_EMPTY)
        
        btn_load = QPushButton("Load")
        btn_load.setMaximumWidth(45)
        btn_load.clicked.connect(lambda: self.load_texture_map(map_attr))
        
        btn_del = QPushButton("Clear")
        btn_del.setMaximumWidth(45)
        btn_del.clicked.connect(lambda: self.remove_texture_map(map_attr))
        
        row.addWidget(lbl)
        row.addWidget(lbl_thumb)
        row.addWidget(lbl_status)
        row.addStretch()
        row.addWidget(btn_load)
        row.addWidget(btn_del)
        
        parent_layout.addLayout(row)
        self.tex_labels[map_attr] = (lbl_status, lbl_thumb)

    def update_data(self, mesh_d: Dict[str, Any]) -> None:
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(mesh_d.get("visible", True))
        self.chk_visible.blockSignals(False)
        
        self.cmb_mat_mode.blockSignals(True)
        self.cmb_mat_mode.setCurrentIndex(1 if mesh_d.get("use_adv", False) else 0)
        self.cmb_mat_mode.blockSignals(False)
        
        self.w_mat_basic.setVisible(not mesh_d.get("use_adv", False))
        self.w_mat_adv.setVisible(mesh_d.get("use_adv", False))

        self.sp_mat_amb.setValue(mesh_d.get("amb_s", DEFAULT_MAT_AMB_STRENGTH))
        self.sp_mat_diff.setValue(mesh_d.get("diff_s", DEFAULT_MAT_DIFF_STRENGTH))
        self.sp_mat_spec.setValue(mesh_d.get("spec_s", DEFAULT_MAT_SPEC_STRENGTH))
        self.sp_shine.setValue(mesh_d.get("shine", DEFAULT_MAT_SHININESS))
        self.sp_opacity.setValue(mesh_d.get("opacity", DEFAULT_MAT_OPACITY))
        
        set_vec3_spinboxes(self.sp_mat_base_vec, mesh_d.get("base_c", list(DEFAULT_MAT_BASE_COLOR)))
        set_vec3_spinboxes(self.sp_mat_amb_vec, mesh_d.get("amb_c", list(DEFAULT_MAT_AMBIENT)))
        set_vec3_spinboxes(self.sp_mat_diff_vec, mesh_d.get("diff_c", list(DEFAULT_MAT_DIFFUSE)))
        set_vec3_spinboxes(self.sp_mat_spec_vec, mesh_d.get("spec_c", list(DEFAULT_MAT_SPECULAR)))
        set_vec3_spinboxes(self.sp_mat_emis_vec, mesh_d.get("emis_c", list(DEFAULT_MAT_EMISSION)))

        self.btn_mat_base.setStyleSheet(rgb_to_hex(mesh_d.get("base_c", list(DEFAULT_MAT_BASE_COLOR))))
        self.btn_mat_amb.setStyleSheet(rgb_to_hex(mesh_d.get("amb_c", list(DEFAULT_MAT_AMBIENT))))
        self.btn_mat_diff.setStyleSheet(rgb_to_hex(mesh_d.get("diff_c", list(DEFAULT_MAT_DIFFUSE))))
        self.btn_mat_spec.setStyleSheet(rgb_to_hex(mesh_d.get("spec_c", list(DEFAULT_MAT_SPECULAR))))
        self.btn_mat_emis.setStyleSheet(rgb_to_hex(mesh_d.get("emis_c", list(DEFAULT_MAT_EMISSION))))
        
        tex_dict = mesh_d.get("tex_paths", {})
        for attr, (lbl_status, lbl_thumb) in self.tex_labels.items():
            t_path = tex_dict.get(attr, "")
            if t_path and os.path.exists(t_path):
                lbl_status.setText("Loaded")
                lbl_status.setStyleSheet(STYLE_TEX_LOADED)
                
                if t_path not in self.pixmap_cache: 
                    self.pixmap_cache[t_path] = QPixmap(t_path).scaled(TEX_THUMB_SIZE, TEX_THUMB_SIZE, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                lbl_thumb.setPixmap(self.pixmap_cache[t_path])
            else:
                lbl_status.setText("Empty")
                lbl_status.setStyleSheet(STYLE_TEX_EMPTY)
                lbl_thumb.clear()
                lbl_thumb.setText("")

    def apply_mesh(self) -> None:
        if not self._controller: return
        self._controller.set_property("Mesh", "visible", self.chk_visible.isChecked())
        self._controller.set_property("Mesh", "mat_ambient_strength", self.sp_mat_amb.value())
        self._controller.set_property("Mesh", "mat_diffuse_strength", self.sp_mat_diff.value())
        self._controller.set_property("Mesh", "mat_specular_strength", self.sp_mat_spec.value())
        self._controller.set_property("Mesh", "mat_shininess", self.sp_shine.value())
        self._controller.set_property("Mesh", "mat_opacity", self.sp_opacity.value())

    def apply_mat_vec_colors(self) -> None:
        if not self._controller: return
        base_c = [s.value() for s in self.sp_mat_base_vec]
        amb_c = [s.value() for s in self.sp_mat_amb_vec]
        diff_c = [s.value() for s in self.sp_mat_diff_vec]
        spec_c = [s.value() for s in self.sp_mat_spec_vec]
        emis_c = [s.value() for s in self.sp_mat_emis_vec]
        
        self._controller.set_property("Mesh", "mat_base_color", base_c)
        self._controller.set_property("Mesh", "mat__ambient", amb_c)
        self._controller.set_property("Mesh", "mat__diffuse", diff_c)
        self._controller.set_property("Mesh", "mat__specular", spec_c)
        self._controller.set_property("Mesh", "mat_emission", emis_c)
        
        self.btn_mat_base.setStyleSheet(rgb_to_hex(base_c))
        self.btn_mat_amb.setStyleSheet(rgb_to_hex(amb_c))
        self.btn_mat_diff.setStyleSheet(rgb_to_hex(diff_c))
        self.btn_mat_spec.setStyleSheet(rgb_to_hex(spec_c))
        self.btn_mat_emis.setStyleSheet(rgb_to_hex(emis_c))

    def switch_mat_mode(self, idx: int) -> None:
        if not self._controller: return
        self._controller.set_property("Mesh", "mat_use_advanced_mode", idx == 1)
        self.w_mat_basic.setVisible(idx == 0)
        self.w_mat_adv.setVisible(idx == 1)

    def pick_mat_color(self, c_type: str) -> None:
        from src.app import ctx
        data = ctx.engine.get_selected_entity_data()
        if not data or not data.get("mesh"): return
            
        default_color = list(DEFAULT_MAT_BASE_COLOR) if c_type == 'base' else (list(DEFAULT_MAT_EMISSION) if c_type == 'emis' else list(DEFAULT_MAT_AMBIENT))
        curr_c = data["mesh"].get(f"{c_type}_c", default_color)
        new_c = self._pick_color_with_dialog(curr_c)
        
        if new_c is not None and self._controller:
            self.request_undo_snapshot()
            prop_map = {'base': 'mat_base_color', 'amb': 'mat__ambient', 'diff': 'mat__diffuse', 'spec': 'mat__specular', 'emis': 'mat_emission'}
            self._controller.set_property("Mesh", prop_map[c_type], new_c)
            
            vec_map = {'base': self.sp_mat_base_vec, 'amb': self.sp_mat_amb_vec, 'diff': self.sp_mat_diff_vec, 'spec': self.sp_mat_spec_vec, 'emis': self.sp_mat_emis_vec}
            btn_map = {'base': self.btn_mat_base, 'amb': self.btn_mat_amb, 'diff': self.btn_mat_diff, 'spec': self.btn_mat_spec, 'emis': self.btn_mat_emis}
            
            set_vec3_spinboxes(vec_map[c_type], new_c)
            btn_map[c_type].setStyleSheet(rgb_to_hex(new_c))
            
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def load_texture_map(self, map_attr: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Texture", "", "Images (*.png *.jpg *.jpeg)")
        if path and self._controller:
            self._controller.load_texture(map_attr, path)

    def remove_texture_map(self, map_attr: str) -> None:
        if self._controller: 
            self._controller.remove_texture(map_attr)