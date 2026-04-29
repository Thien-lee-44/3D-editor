"""
Project Browser Dialog.
Provides a standalone UI window allowing users to open or delete saved projects.
"""

import os
from typing import Any, Optional
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QListWidget, QPushButton, QMessageBox)

from src.app.config import PROJECT_MANAGER_TITLE, PROJECT_MANAGER_MIN_SIZE

class ProjectBrowserDialog(QDialog):
    """
    Popup dialog interface for selecting and managing saved projects.
    Handles scanning the project directory and dispatching user choices.
    """
    
    def __init__(self, parent: Any, proj_dir: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(PROJECT_MANAGER_TITLE)
        self.setMinimumSize(PROJECT_MANAGER_MIN_SIZE[0], PROJECT_MANAGER_MIN_SIZE[1])
        
        self.proj_dir: str = proj_dir
        self.selected_path: Optional[str] = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Saved Projects:"))
        
        self.list_widget = QListWidget()
        self.refresh_list()
        layout.addWidget(self.list_widget)
        
        # Setup control buttons
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Project")
        self.btn_del = QPushButton("Delete Project")
        self.btn_cancel = QPushButton("Cancel")
        
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_del)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        # Bind UI logic
        self.btn_open.clicked.connect(self.on_open)
        self.btn_del.clicked.connect(self.on_delete)
        self.btn_cancel.clicked.connect(self.reject)
        self.list_widget.itemDoubleClicked.connect(self.on_open)

    def refresh_list(self) -> None:
        """Scans the local projects directory and populates the UI list."""
        self.list_widget.clear()
        if os.path.exists(self.proj_dir):
            for f in os.listdir(self.proj_dir):
                if f.endswith('.json'):
                    self.list_widget.addItem(f.replace('.json', ''))

    def on_open(self) -> None:
        """Captures the selected project path and closes the dialog with an Accepted state."""
        item = self.list_widget.currentItem()
        if item:
            self.selected_path = os.path.join(self.proj_dir, f"{item.text()}.json")
            self.accept()
            
    def on_delete(self) -> None:
        """Prompts for confirmation before permanently deleting a project file from disk."""
        item = self.list_widget.currentItem()
        if item:
            proj_name = item.text()
            ans = QMessageBox.warning(
                self, "Confirm Delete", 
                f"Are you sure you want to delete the project '{proj_name}'?\nThis action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if ans == QMessageBox.Yes:
                del_path = os.path.join(self.proj_dir, f"{proj_name}.json")
                if os.path.exists(del_path):
                    try:
                        os.remove(del_path)
                        self.refresh_list()
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Could not delete file:\n{e}")