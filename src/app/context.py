"""
Global Application Context.
Implements the Singleton Design Pattern to provide a centralized state repository.
"""

from typing import Any
from .events import EventBus

class AppContext:
    """
    Global application state repository.
    Controllers interact with this context to access the 3D Engine Facade or emit/listen to events 
    without requiring tight coupling or circular imports.
    """
    _instance = None

    def __new__(cls) -> 'AppContext':
        if cls._instance is None:
            cls._instance = super(AppContext, cls).__new__(cls)
            cls._instance._engine = None
            cls._instance._events = EventBus()
        return cls._instance

    @property
    def engine(self) -> Any:
        """Accesses the 3D Engine Facade API."""
        if self._engine is None:
            raise RuntimeError("CRITICAL: AppContext.engine is not initialized. Ensure the Engine is assigned during application bootstrap.")
        return self._engine

    @engine.setter
    def engine(self, value: Any) -> None:
        """Injects the Engine instance. This should strictly be called once during startup."""
        self._engine = value

    @property
    def events(self) -> EventBus:
        """Accesses the Event Bus for cross-module decoupled communication."""
        return self._events
    
# Global Singleton Instance exposed to the entire application
ctx = AppContext()