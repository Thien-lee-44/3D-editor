"""
Core Application Namespace.

Exposes the global context and event bus for application-wide imports.
"""

from .context import ctx
from .events import AppEvent, EventBus

__all__ = ["ctx", "AppEvent", "EventBus"]