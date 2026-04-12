"""
Event-Driven Architecture Core.
Defines the Pub-Sub (Publish-Subscribe) mechanism allowing disjointed 
modules (UI, Engine, I/O) to communicate asynchronously.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Any

class AppEvent(Enum):
    """
    Standardized dictionary of all valid application events.
    Controllers utilize this Enum to guarantee type safety when emitting signals.
    """
    # --- Project & Scene State ---
    PROJECT_LOADED = auto()              # Triggered when a .json project is fully deserialized
    PROJECT_SAVED = auto()               # Triggered upon successful project serialization
    SCENE_CHANGED = auto()               # Signals the 3D Viewport to trigger a hardware re-render
    
    # --- Undo / Redo Memory ---
    ACTION_BEFORE_MUTATION = auto()      # Fired BEFORE data mutation to capture an Undo snapshot
    HISTORY_RECORDED = auto()            # Signals a successful snapshot capture
    
    # --- UI Refresh Signals ---
    HIERARCHY_NEEDS_REFRESH = auto()     # Prompts the Hierarchy Panel to rebuild its tree widget
    ASSET_BROWSER_NEEDS_REFRESH = auto() # Prompts the Asset Browser to rescan local files
    COMPONENT_PROPERTY_CHANGED = auto()  # Signals an inspector value modification
    
    # --- Selection & Interaction ---
    ENTITY_SELECTED = auto()             # Payload: (entity_id: int)
    TRANSFORM_FAST_UPDATED = auto()      # Payload: (transform_data: dict) - Prevents lag during high-frequency Gizmo drags
    
    # --- Asset Management ---
    ASSET_IMPORTED = auto()              # Payload: (asset_type: str, path: str)


class EventBus:
    """
    Lightweight Pub-Sub event dispatcher.
    Allows modules to broadcast (emit) and listen (subscribe) to events anonymously.
    """
    def __init__(self) -> None:
        # Initialize empty subscriber lists mapped to each Enum event type
        self._subscribers: Dict[AppEvent, List[Callable]] = {event: [] for event in AppEvent}

    def subscribe(self, event_type: AppEvent, callback: Callable) -> None:
        """Registers a callback function to listen for a specific event."""
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: AppEvent, callback: Callable) -> None:
        """Removes a registered callback from the listener pool."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def emit(self, event_type: AppEvent, *args: Any, **kwargs: Any) -> None:
        """
        Broadcasts an event with optional payloads to all registered callbacks.
        """
        # Cast to list() to prevent 'dictionary changed size during iteration' runtime errors
        for callback in list(self._subscribers[event_type]):
            callback(*args, **kwargs)
            
    def clear_all(self) -> None:
        """Flushes all event memory (typically used during project teardown/load operations)."""
        for event in self._subscribers:
            self._subscribers[event].clear()