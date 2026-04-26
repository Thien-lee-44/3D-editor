from typing import Any
from PySide6.QtWidgets import QFormLayout, QLineEdit
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
        self.request_undo_snapshot() 
        self._controller.set_properties("Entity", {"name": self.txt_name.text().strip()})
        
        from src.app import ctx, AppEvent
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)