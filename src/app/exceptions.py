"""
Domain-Specific Exceptions.
Defines the strict exception hierarchy for the 3D Engine. 
The Engine layer MUST ONLY use these exceptions to report failures.
Generic Exception() or ValueError() should be wrapped into these before reaching the UI.
"""

class EngineError(Exception):
    """Base exception class for all engine-related domain errors."""
    pass

class ShaderError(EngineError):
    """Raised when GLSL shader compilation or hardware linking fails."""
    pass

class ResourceError(EngineError):
    """Raised when an external resource (Model, Texture, Configuration) cannot be parsed or loaded."""
    pass

class RenderError(EngineError):
    """Raised when an OpenGL state setup, FBO generation, or draw call fails."""
    pass

class SimulationError(EngineError):
    """Raised during ECS state violations, logical constraints (e.g., max lights), or math failures."""
    pass