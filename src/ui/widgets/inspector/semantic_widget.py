from typing import Any, Dict
from PySide6.QtWidgets import (QFormLayout, QLabel, QComboBox, QPushButton, 
                               QHBoxLayout, QInputDialog, QColorDialog, QCheckBox, QWidget)
from PySide6.QtGui import QColor
from src.ui.widgets.inspector.base_widget import BaseComponentWidget

class SemanticWidget(BaseComponentWidget):
    """
    Refined Inspector UI for Semantic Labeling.
    Dynamically adapts controls based on whether the entity acts as a Unified Object 
    or purely as an Organizational Folder capable of bulk-assigning Class IDs.
    """
    def __init__(self, controller: Any) -> None:
        super().__init__("Semantic Labeling", controller)
        
        self.chk_is_object = QCheckBox("Treat as Unified 3D Object")
        self.chk_is_object.setToolTip("Uncheck to use this entity as a folder. Unified Objects forcefully sync semantics to children.")
        self.chk_is_object.toggled.connect(self.toggle_is_object)
        self.layout.addWidget(self.chk_is_object)

        self.chk_propagate = QCheckBox("Apply Class ID to Children")
        self.chk_propagate.setToolTip("Bulk-assign this Class ID to all nested entities.")
        self.chk_propagate.toggled.connect(self.toggle_propagation)
        self.layout.addWidget(self.chk_propagate)

        self.data_container = QWidget()
        form = QFormLayout(self.data_container)
        form.setContentsMargins(0, 5, 0, 5)

        self.lbl_track_title = QLabel("Track ID:")
        self.lbl_track_id = QLabel("Auto (Dynamic)")
        self.lbl_track_id.setStyleSheet("color: #00BFFF; font-style: italic; font-weight: bold;")
        form.addRow(self.lbl_track_title, self.lbl_track_id)

        self.cmb_class = QComboBox()
        self.cmb_class.currentIndexChanged.connect(self.apply_class_change)
        
        self.btn_color = QPushButton()
        self.btn_color.setFixedWidth(28)
        self.btn_color.clicked.connect(self.change_class_color)

        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(28)
        self.btn_add.clicked.connect(self.add_new_class)
        
        class_layout = QHBoxLayout()
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(self.cmb_class)
        class_layout.addWidget(self.btn_color)
        class_layout.addWidget(self.btn_add)

        form.addRow("Class ID:", class_layout)
        self.layout.addWidget(self.data_container)

    def update_data(self, data: Dict[str, Any]) -> None:
        is_group = data.get("is_group", False)
        is_obj = data.get("is_merged_instance", True)
        should_prop = data.get("propagate_to_children", True)

        self.chk_is_object.setVisible(is_group)
        self.chk_propagate.setVisible(is_group and not is_obj)

        show_track = is_obj if is_group else True
        self.lbl_track_title.setVisible(show_track)
        self.lbl_track_id.setVisible(show_track)

        self.chk_is_object.blockSignals(True)
        self.chk_is_object.setChecked(is_obj)
        self.chk_is_object.blockSignals(False)

        self.chk_propagate.blockSignals(True)
        self.chk_propagate.setChecked(should_prop)
        self.chk_propagate.blockSignals(False)

        self.cmb_class.blockSignals(True)
        self.cmb_class.clear()
        
        if self._controller:
            classes = self._controller.get_semantic_classes()
            for c_id, c_info in classes.items():
                name = c_info.get("name", "Unknown") if isinstance(c_info, dict) else c_info
                self.cmb_class.addItem(f"{c_id}: {name}", c_id)

        target_id = data.get("class_id", 0)
        idx = self.cmb_class.findData(target_id)
        if idx >= 0:
            self.cmb_class.setCurrentIndex(idx)
            c_info = classes.get(target_id, {}) if self._controller else {}
            color = c_info.get("color", [1.0, 1.0, 1.0]) if isinstance(c_info, dict) else [1.0, 1.0, 1.0]
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            self.btn_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;")
            
        self.cmb_class.blockSignals(False)

    def toggle_is_object(self, checked: bool) -> None:
        self.chk_propagate.setVisible(not checked)
        self.lbl_track_title.setVisible(checked)
        self.lbl_track_id.setVisible(checked)
        
        if self._controller:
            self.request_undo_snapshot()
            self._controller.set_properties("Semantic", {"is_merged_instance": checked})

    def toggle_propagation(self, checked: bool) -> None:
        if self._controller:
            self.request_undo_snapshot()
            self._controller.set_properties("Semantic", {"propagate_to_children": checked})

    def apply_class_change(self) -> None:
        if self._controller:
            class_id = self.cmb_class.currentData()
            if class_id is not None:
                self.request_undo_snapshot()
                self._controller.set_properties("Semantic", {"class_id": class_id})

    def add_new_class(self) -> None:
        if not self._controller: return
        name, ok = QInputDialog.getText(self, "New Class", "Enter class name:")
        if ok and name.strip():
            new_id = self._controller.add_semantic_class(name.strip())
            self.update_data({"class_id": new_id})

    def change_class_color(self) -> None:
        if not self._controller: return
        class_id = self.cmb_class.currentData()
        classes = self._controller.get_semantic_classes()
        c_info = classes.get(class_id, {})
        curr_color = c_info.get("color", [1.0, 1.0, 1.0])
        init_color = QColor(int(curr_color[0] * 255), int(curr_color[1] * 255), int(curr_color[2] * 255))
        
        color = QColorDialog.getColor(init_color, self, "Select Class Color")
        if color.isValid():
            rgb = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
            self._controller.update_semantic_class_color(class_id, rgb)