from typing import Any
from PySide6.QtWidgets import QFormLayout, QLineEdit, QMessageBox
from .base_widget import BaseComponentWidget

class HeaderWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__("Entity Info", controller)
        f_layout = QFormLayout()
        
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self.apply_name)
        f_layout.addRow("Name:", self.txt_name)
        self.layout.addLayout(f_layout)

    def update_data(self, name: str) -> None:
        self.txt_name.blockSignals(True)
        self.txt_name.setText(name)
        self.txt_name.blockSignals(False)

    def apply_name(self) -> None:
        if not self._controller: return
        
        new_name = self.txt_name.text().strip()
        if not new_name:
            return
            
        from src.app import ctx, AppEvent
        
        current_id = ctx.engine.get_selected_entity_id()
        entities = ctx.engine.get_scene_entities_list()
        
        for ent in entities:
            if ent["id"] != current_id and ent["name"] == new_name:
                QMessageBox.warning(
                    self, 
                    "Invalid Name", 
                    f"The name '{new_name}' is already in use.\nPlease choose a unique name."
                )
                data = ctx.engine.get_selected_entity_data()
                if data:
                    self.update_data(data.get("name", ""))
                return

        self.request_undo_snapshot() 
        self._controller.set_properties("Entity", {"name": new_name})
        
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)