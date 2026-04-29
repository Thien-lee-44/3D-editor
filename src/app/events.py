"""
Event-Driven Architecture Core.
Implements a Pub-Sub mechanism for asynchronous module communication.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Any

class AppEvent(Enum):
    """
    Standardized application events to guarantee type safety.
    """
    # --- Project & Scene State ---
    PROJECT_LOADED = auto()              
    PROJECT_SAVED = auto()               
    SCENE_CHANGED = auto()               
    
    # --- Undo / Redo Memory ---
    ACTION_BEFORE_MUTATION = auto()      
    HISTORY_RECORDED = auto()            
    
    # --- UI Refresh Signals ---
    HIERARCHY_NEEDS_REFRESH = auto()     
    ASSET_BROWSER_NEEDS_REFRESH = auto() 
    COMPONENT_PROPERTY_CHANGED = auto()  
    
    # --- Selection & Interaction ---
    ENTITY_SELECTED = auto()             
    TRANSFORM_FAST_UPDATED = auto()      
    
    # --- Asset Management ---
    ASSET_IMPORTED = auto()              

class EventBus:
    """
    Lightweight event dispatcher managing listener registrations and broadcasts.
    """
    def __init__(self) -> None:
        self._subscribers: Dict[AppEvent, List[Callable]] = {event: [] for event in AppEvent}

    def subscribe(self, event_type: AppEvent, callback: Callable) -> None:
        """Registers a listener for a specific event."""
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: AppEvent, callback: Callable) -> None:
        """Removes a registered listener."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def emit(self, event_type: AppEvent, *args: Any, **kwargs: Any) -> None:
        """
        Broadcasts an event with optional payloads to all registered callbacks.
        """
        # Iterate over a copy to safely handle modifications during emission
        for callback in list(self._subscribers[event_type]):
            callback(*args, **kwargs)
            
    def clear_all(self) -> None:
        """Flushes all registered callbacks. Used during teardown operations."""
        for event in self._subscribers:
            self._subscribers[event].clear()