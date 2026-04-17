import glm
import math
from typing import Dict, Any
from src.engine.scene.entity import Component

# Import centralized configurations
from src.app.config import (
    DEFAULT_LIGHT_COLOR, DEFAULT_LIGHT_INTENSITY, 
    DEFAULT_LIGHT_AMBIENT, DEFAULT_LIGHT_DIFFUSE, DEFAULT_LIGHT_SPECULAR,
    DEFAULT_LIGHT_CONSTANT, DEFAULT_LIGHT_LINEAR, DEFAULT_LIGHT_QUADRATIC,
    DEFAULT_SPOT_INNER_ANGLE, DEFAULT_SPOT_OUTER_ANGLE
)

class LightComponent(Component):
    """
    Provides physical illumination parameters mapped directly to the GLSL Forward Renderer.
    Contains mathematical abstractions like Cutoff angles for Spotlights.
    """
    
    def __init__(self, light_type: str = "Point") -> None:
        super().__init__()
        self.type: str = light_type 
        self.on: bool = True         
        self.intensity: float = DEFAULT_LIGHT_INTENSITY         
        self.use_advanced_mode: bool = False 
        
        # Basic mode properties (Scalar multipliers against a base color)
        self.color = glm.vec3(*DEFAULT_LIGHT_COLOR)
        self.ambient_strength: float = DEFAULT_LIGHT_AMBIENT
        self.diffuse_strength: float = DEFAULT_LIGHT_DIFFUSE
        self.specular_strength: float = DEFAULT_LIGHT_SPECULAR
        
        # Advanced mode properties (Independent RGB control per channel)
        self.explicit_ambient = glm.vec3(*DEFAULT_LIGHT_COLOR)
        self.explicit_diffuse = glm.vec3(*DEFAULT_LIGHT_COLOR)
        self.explicit_specular = glm.vec3(*DEFAULT_LIGHT_COLOR)
        
        # Pre-calculated cosine values for Spotlight cones to avoid calculating acos() in the shader
        self.cutOff: float = math.cos(math.radians(DEFAULT_SPOT_INNER_ANGLE))
        self.outerCutOff: float = math.cos(math.radians(DEFAULT_SPOT_OUTER_ANGLE))
        
        self.constant: float = DEFAULT_LIGHT_CONSTANT
        self.linear: float = DEFAULT_LIGHT_LINEAR
        self.quadratic: float = DEFAULT_LIGHT_QUADRATIC
                
    @property
    def ambient(self) -> glm.vec3:
        """Evaluates final ambient color taking into account the global intensity scalar."""
        if not self.on: 
            return glm.vec3(0)
        base = self.explicit_ambient if self.use_advanced_mode else (self.color * self.ambient_strength)
        return base * self.intensity
                
    @property
    def diffuse(self) -> glm.vec3:
        """Evaluates final diffuse scattering color."""
        if not self.on: 
            return glm.vec3(0)
        base = self.explicit_diffuse if self.use_advanced_mode else (self.color * self.diffuse_strength)
        return base * self.intensity
                
    @property
    def specular(self) -> glm.vec3:
        """Evaluates final specular highlight color."""
        if not self.on: 
            return glm.vec3(0)
        base = self.explicit_specular if self.use_advanced_mode else (self.color * self.specular_strength)
        return base * self.intensity

    def to_dict(self) -> Dict[str, Any]:
        """Strict JSON mapping. Keys match exact variable names."""
        return {
            "type": self.type, 
            "on": self.on, 
            "intensity": float(self.intensity),
            "use_advanced_mode": self.use_advanced_mode,
            "ambient_strength": float(self.ambient_strength), 
            "diffuse_strength": float(self.diffuse_strength), 
            "specular_strength": float(self.specular_strength),
            "color": list(self.color), 
            "explicit_ambient": list(self.explicit_ambient),
            "explicit_diffuse": list(self.explicit_diffuse), 
            "explicit_specular": list(self.explicit_specular),
            "cutOff": float(self.cutOff), 
            "outerCutOff": float(self.outerCutOff), 
            "constant": float(self.constant),
            "linear": float(self.linear),
            "quadratic": float(self.quadratic)
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.type = data.get("type", "Point")
        self.on = bool(data.get("on", True))
        self.intensity = float(data.get("intensity", DEFAULT_LIGHT_INTENSITY))
        self.use_advanced_mode = bool(data.get("use_advanced_mode", False))
        self.ambient_strength = float(data.get("ambient_strength", DEFAULT_LIGHT_AMBIENT))
        self.diffuse_strength = float(data.get("diffuse_strength", DEFAULT_LIGHT_DIFFUSE))
        self.specular_strength = float(data.get("specular_strength", DEFAULT_LIGHT_SPECULAR))
        
        self.color = glm.vec3(*data.get("color", list(DEFAULT_LIGHT_COLOR)))
        self.explicit_ambient = glm.vec3(*data.get("explicit_ambient", list(DEFAULT_LIGHT_COLOR)))
        self.explicit_diffuse = glm.vec3(*data.get("explicit_diffuse", list(DEFAULT_LIGHT_COLOR)))
        self.explicit_specular = glm.vec3(*data.get("explicit_specular", list(DEFAULT_LIGHT_COLOR)))
        
        self.cutOff = float(data.get("cutOff", math.cos(math.radians(DEFAULT_SPOT_INNER_ANGLE))))
        self.outerCutOff = float(data.get("outerCutOff", math.cos(math.radians(DEFAULT_SPOT_OUTER_ANGLE))))
        
        self.constant = float(data.get("constant", DEFAULT_LIGHT_CONSTANT))
        self.linear = float(data.get("linear", DEFAULT_LIGHT_LINEAR))
        self.quadratic = float(data.get("quadratic", DEFAULT_LIGHT_QUADRATIC))