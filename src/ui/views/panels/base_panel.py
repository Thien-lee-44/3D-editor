"""
Base Panel View.

Provides the foundational UI class for all dockable editor panels.
"""

from typing import Any, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from src.ui.views.base_view import BaseView
from src.app.config import PANEL_TITLE_UNKNOWN, PANEL_MIN_WIDTH_DEFAULT


class BasePanel(BaseView):
    """
    Base class for dockable tool panels (Hierarchy, Inspector, etc.).
    Provides metadata so the MainController can automatically generate QDockWidgets.
    """
    
    # --- Metadata (Should be overridden by subclasses) ---
    PANEL_TITLE = PANEL_TITLE_UNKNOWN
    PANEL_DOCK_AREA = Qt.RightDockWidgetArea
    PANEL_MIN_WIDTH = PANEL_MIN_WIDTH_DEFAULT
    PANEL_MIN_HEIGHT = 0
    # -----------------------------------------------------

    def __init__(self, controller: Optional[Any] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(controller, parent)
        self.setMinimumWidth(self.PANEL_MIN_WIDTH)
        
        if self.PANEL_MIN_HEIGHT > 0:
            self.setMinimumHeight(self.PANEL_MIN_HEIGHT)