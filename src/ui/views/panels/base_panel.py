"""
Base Panel View.
Defines the foundational class for all dockable tool panels within the editor.
"""

from typing import Any, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from src.ui.views.base_view import BaseView
from src.app.config import PANEL_TITLE_UNKNOWN, PANEL_MIN_WIDTH_DEFAULT

class BasePanel(BaseView):
    """
    Base class for dockable tool panels (Hierarchy, Inspector, Asset Browser, etc.).
    Provides structural metadata allowing the MainController to automatically 
    generate and configure QDockWidgets.
    """
    
    # --- Metadata (Must be overridden by subclasses) ---
    PANEL_TITLE: str = PANEL_TITLE_UNKNOWN
    PANEL_DOCK_AREA: Qt.DockWidgetArea = Qt.RightDockWidgetArea
    PANEL_MIN_WIDTH: int = PANEL_MIN_WIDTH_DEFAULT
    PANEL_MIN_HEIGHT: int = 0
    # ---------------------------------------------------

    def __init__(self, controller: Optional[Any] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(controller, parent)
        self.setMinimumWidth(self.PANEL_MIN_WIDTH)
        if self.PANEL_MIN_HEIGHT > 0:
            self.setMinimumHeight(self.PANEL_MIN_HEIGHT)