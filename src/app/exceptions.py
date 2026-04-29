"""
Domain-Specific Exceptions.
Defines a strict exception hierarchy for the 3D Engine subsystem.
"""

class EngineError(Exception):
    """Base class for all Engine-related domain exceptions."""
    pass

class ShaderError(EngineError):
    """Raised when GLSL compilation or program linking fails."""
    pass

class ResourceError(EngineError):
    """Raised when external assets (Models, Textures) fail to load or parse."""
    pass

class RenderError(EngineError):
    """Raised upon failure in OpenGL state configuration or draw calls."""
    pass

class SimulationError(EngineError):
    """Raised upon invalid ECS state transitions or logical constraints violations."""
    pass