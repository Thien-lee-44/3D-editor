"""
Header Widget.
Displays and allows editing of the fundamental properties (e.g., Name) of the selected Entity.
"""

from typing import Any
from PySide6.QtWidgets import QFormLayout, QLineEdit
from .base_widget import BaseComponentWidget

class HeaderWidget(BaseComponentWidget):
    """
    Inspector panel for editing the Entity's name.
    Automatically triggers a hierarchy tree rebuild upon modification.
    """
    
    def __init__(self, controller: Any) -> None:
        super().__init__("Entity Info", controller)
        f_layout = QFormLayout()
        
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self.apply_name)
        f_layout.addRow("Name:", self.txt_name)
        self.layout.addLayout(f_layout)

    def update_data(self, name: str) -> None:
        """Populates the text field silently."""
        self.txt_name.blockSignals(True)
        self.txt_name.setText(name)
        self.txt_name.blockSignals(False)

    def apply_name(self) -> None:
        """Validates the input and dispatches the mutation to the Controller."""
        if not self._controller: 
            return
            
        self.request_undo_snapshot() 
        self._controller.set_property("Entity", "name", self.txt_name.text().strip())
        
        # Emit Event to notify the software that the name changed, forcing the Hierarchy tree to rebuild
        from src.app import ctx, AppEvent
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)