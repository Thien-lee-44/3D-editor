import os
import glm
from OpenGL.GL import *
from typing import Tuple, Dict, Any, Optional

# Import SSOT configuration
from src.app.config import (
    DEFAULT_MAT_AMBIENT, DEFAULT_MAT_DIFFUSE, 
    DEFAULT_MAT_SPECULAR, DEFAULT_MAT_SHININESS, TEXTURE_CHANNELS
)
TEXTURE_MAP_ATTRS = tuple(TEXTURE_CHANNELS.values())

class RenderState:
    """Encapsulates OpenGL render state settings for a specific material."""
    def __init__(self) -> None:
        self.cull_face: bool = True
        self.cull_mode: int = GL_BACK
        self.depth_test: bool = True
        self.depth_write: bool = True
        self.depth_func: int = GL_LESS
        self.blend: bool = False
        self.wireframe_override: bool = False

class Material:
    """
    Defines the surface characteristics and multi-texturing mappings for an entity.
    Bridges the gap between the ECS data structures and the GLSL shader uniforms.
    """
    
    def __init__(self, 
                 ambient: Tuple[float, float, float] = DEFAULT_MAT_AMBIENT, 
                 diffuse: Tuple[float, float, float] = DEFAULT_MAT_DIFFUSE, 
                 specular: Tuple[float, float, float] = DEFAULT_MAT_SPECULAR, 
                 shininess: float = DEFAULT_MAT_SHININESS) -> None:
                     
        # Toggle between scalar multiplier-based coloring and independent RGB channels
        self.use_advanced_mode = False 
        
        # Base attributes designed for simplified UI manipulation
        self.base_color = glm.vec3(*diffuse) 
        self.ambient_strength = 0.5
        self.diffuse_strength = 1.0
        self.specular_strength = 1.0
        
        # Physical lighting reflection parameters
        self._ambient = glm.vec3(*ambient)
        self._diffuse = glm.vec3(*diffuse)
        self._specular = glm.vec3(*specular)
        self.emission = glm.vec3(0.0) 
        self.shininess = shininess
        self.opacity = 1.0            # 1.0 equates to fully opaque, 0.0 to fully transparent
        self.ior = 1.0                # Index of Refraction for translucent materials
        self.illum = 2                # Standard Illumination model identifier
        
        # NEW: Render State and Custom Shader Injection
        self.render_state = RenderState()
        self.custom_shader_name: str = ""
        
        # Multi-texturing Slots holding OpenGL Texture Object IDs
        self.map_diffuse: int = 0
        self.map_specular: int = 0
        self.map_bump: int = 0
        self.map_ambient: int = 0
        self.map_emission: int = 0
        self.map_shininess: int = 0
        self.map_opacity: int = 0
        self.map_reflection: int = 0
        
        # Centralized dictionary tracking the absolute file paths of all active texture maps.
        self.tex_paths: Dict[str, str] = {}

    # Dynamic properties evaluate the final physical channels based on the current UI mode
    @property
    def ambient(self) -> glm.vec3: 
        return self._ambient if self.use_advanced_mode else self.base_color * self.ambient_strength
        
    @ambient.setter
    def ambient(self, val: glm.vec3) -> None: 
        self._ambient = val

    @property
    def diffuse(self) -> glm.vec3: 
        return self._diffuse if self.use_advanced_mode else self.base_color * self.diffuse_strength
        
    @diffuse.setter
    def diffuse(self, val: glm.vec3) -> None: 
        self._diffuse = val

    @property
    def specular(self) -> glm.vec3: 
        return self._specular if self.use_advanced_mode else self.base_color * self.specular_strength
        
    @specular.setter
    def specular(self, val: glm.vec3) -> None: 
        self._specular = val

    def clear_texture_slots(self) -> None:
        for attr_name in TEXTURE_MAP_ATTRS:
            setattr(self, attr_name, 0)

    def get_tex_paths_snapshot(self) -> Dict[str, str]:
        return {
            k: v
            for k, v in self.tex_paths.items()
            if k in TEXTURE_MAP_ATTRS and isinstance(v, str) and v.strip()
        }

    def apply_texture_paths(self, tex_paths: Dict[str, Any]) -> None:
        """
        Applies a texture-path payload deterministically:
        1. Clears all texture slots
        2. Keeps only supported texture map keys
        3. Rebuilds texture IDs for existing files
        """
        from src.engine.resources.resource_manager import ResourceManager

        self.clear_texture_slots()
        sanitized_paths: Dict[str, str] = {}

        for attr_name in TEXTURE_MAP_ATTRS:
            raw_path = tex_paths.get(attr_name, "")
            if not isinstance(raw_path, str):
                continue
            t_path = raw_path.strip()
            if not t_path:
                continue

            sanitized_paths[attr_name] = t_path
            if os.path.exists(t_path):
                tid = ResourceManager.load_texture(t_path)
                if tid != 0:
                    setattr(self, attr_name, tid)

        self.tex_paths = sanitized_paths

    def apply(self, shader: Any) -> None:
        """
        Transmits scalar/vector properties and binds active texture units to the currently executing Shader.
        Called implicitly by the Forward Renderer during the draw loop.
        """
        shader.set_vec3("material.ambient", self.ambient)
        shader.set_vec3("material.diffuse", self.diffuse)
        shader.set_vec3("material.specular", self.specular)
        shader.set_vec3("material.emission", self.emission)
        shader.set_float("material.shininess", self.shininess)
        shader.set_float("material.opacity", self.opacity)
        
        def bind_tex(tex_id: int, unit: int, name: str) -> None:
            """Internal helper to automatically allocate hardware texture units and toggle GLSL logic flags."""
            if tex_id != 0:
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                shader.set_int(name, unit)
                shader.set_int(f"has{name[0].upper() + name[1:]}", 1)
            else:
                shader.set_int(f"has{name[0].upper() + name[1:]}", 0)

        # Distribute texture maps across Texture Units 0 through 7
        bind_tex(self.map_diffuse, 0, "mapDiffuse")
        bind_tex(self.map_specular, 1, "mapSpecular")
        bind_tex(self.map_bump, 2, "mapBump")
        bind_tex(self.map_ambient, 3, "mapAmbient")
        bind_tex(self.map_emission, 4, "mapEmission")
        bind_tex(self.map_shininess, 5, "mapShininess")
        bind_tex(self.map_opacity, 6, "mapOpacity")
        bind_tex(self.map_reflection, 7, "mapReflection")
        
        # Reset hardware active texture state to prevent spillover effects on subsequent draw calls
        glActiveTexture(GL_TEXTURE0) 

    def setup_from_dict(self, mtl_data: Dict[str, Any]) -> None:
        """
        Reconstructs the material state from a parsed dictionary.
        Typically utilized during .obj/.mtl loading or project deserialization.
        """
        self.use_advanced_mode = True
        self._ambient = glm.vec3(*mtl_data.get('ambient', DEFAULT_MAT_AMBIENT))
        self._diffuse = glm.vec3(*mtl_data.get('diffuse', DEFAULT_MAT_DIFFUSE))
        self._specular = glm.vec3(*mtl_data.get('specular', DEFAULT_MAT_SPECULAR))
        self.emission = glm.vec3(*mtl_data.get('emission', [0.0, 0.0, 0.0]))
        self.shininess = mtl_data.get('shininess', DEFAULT_MAT_SHININESS)
        self.opacity = mtl_data.get('opacity', 1.0)
        
        tex_payload = {attr_name: mtl_data.get(attr_name, "") for attr_name in TEXTURE_MAP_ATTRS}
        self.apply_texture_paths(tex_payload)
