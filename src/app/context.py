"""
Global Application Context.
Implements the Singleton pattern to provide a centralized state repository.
"""

from typing import Any, Optional
from .events import EventBus

class AppContext:
    """
    Global state container bridging UI and Engine layers.
    Prevents tight coupling and circular imports.
    """
    _instance: Optional['AppContext'] = None

    def __new__(cls) -> 'AppContext':
        if cls._instance is None:
            cls._instance = super(AppContext, cls).__new__(cls)
            cls._instance._engine = None
            cls._instance._events = EventBus()
        return cls._instance

    @property
    def engine(self) -> Any:
        """Retrieves the active 3D Engine instance."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Ensure assignment during application bootstrap.")
        return self._engine

    @engine.setter
    def engine(self, value: Any) -> None:
        """Injects the Engine dependency once during startup."""
        self._engine = value

    @property
    def events(self) -> EventBus:
        """Accesses the global Event Bus."""
        return self._events
    
# Global Singleton instance
ctx = AppContext()