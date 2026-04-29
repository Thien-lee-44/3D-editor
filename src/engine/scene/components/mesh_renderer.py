"""
Mesh Renderer Component.
Couples geometric vertex data with surface material properties for the rendering pipeline.
"""

import os
import glm
import copy
from typing import Dict, Any, Optional
from src.engine.scene.entity import Component
from src.engine.graphics.material import Material

from src.app.config import (
    DEFAULT_MATH_RANGE, DEFAULT_MATH_RESOLUTION,
    DEFAULT_MAT_AMBIENT, DEFAULT_MAT_DIFFUSE, DEFAULT_MAT_SPECULAR, 
    DEFAULT_MAT_EMISSION, DEFAULT_MAT_BASE_COLOR, DEFAULT_MAT_SHININESS, 
    DEFAULT_MAT_OPACITY, DEFAULT_MAT_AMB_STRENGTH, DEFAULT_MAT_DIFF_STRENGTH, 
    DEFAULT_MAT_SPEC_STRENGTH
)

class MeshRenderer(Component):
    """
    Primary identifier for rendering objects. 
    Handles deep serialization of mesh origins and material states.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.geometry: Optional[Any] = None    
        self.material: Material = Material() 
        self.visible: bool = True
        self.is_proxy: bool = False

    def __deepcopy__(self, memo: dict) -> 'MeshRenderer':
        """
        Custom deep clone implementation for prefab instantiation.
        Duplicates materials uniquely while sharing Geometry pointers in memory.
        """
        new_obj = type(self)()
        memo[id(self)] = new_obj
        
        new_obj.visible = self.visible
        new_obj.is_proxy = self.is_proxy
        new_obj.geometry = self.geometry 
        new_obj.material = copy.deepcopy(self.material, memo)
        
        return new_obj

    def to_dict(self) -> Dict[str, Any]:
        """Serializes mesh references and material configuration."""
        from src.engine.resources.resource_manager import ResourceManager
        from src.engine.geometry.primitives import PrimitivesManager
        
        data: Dict[str, Any] = {
            "visible": self.visible, 
            "is_proxy": getattr(self, 'is_proxy', False)
        }
        
        # --- Geometry Origin Serialization ---
        if self.geometry:
            if hasattr(self.geometry, 'filepath') and self.geometry.filepath and not self.is_proxy:
                data["geometry_path"] = self.geometry.filepath
                data["submesh_name"] = getattr(self.geometry, 'name', '')
            elif hasattr(self.geometry, 'formula_str'):
                data["math_formula"] = self.geometry.formula_str
                data["math_ranges"] = [
                    getattr(self.geometry, 'x_range', list(DEFAULT_MATH_RANGE)),
                    getattr(self.geometry, 'y_range', list(DEFAULT_MATH_RANGE)),
                    getattr(self.geometry, 'resolution', DEFAULT_MATH_RESOLUTION)
                ]
            elif self.is_proxy:
                data["proxy_path"] = getattr(self.geometry, 'filepath', '') or getattr(self.geometry, 'name', '')
            else:
                prim_name = None
                for k, v in PrimitivesManager.get_3d_paths().items():
                    if v == self.geometry: 
                        prim_name = k
                        break
                if not prim_name:
                    for k, v in PrimitivesManager.get_2d_paths().items():
                        if v == self.geometry: 
                            prim_name = k
                            break
                if prim_name: 
                    data["primitive_name"] = prim_name

        # --- Material State Serialization ---
        mat = self.material
        data.update({
            "use_adv": getattr(mat, 'use_advanced_mode', False),
            "amb_s": getattr(mat, 'ambient_strength', DEFAULT_MAT_AMB_STRENGTH),
            "diff_s": getattr(mat, 'diffuse_strength', DEFAULT_MAT_DIFF_STRENGTH),
            "spec_s": getattr(mat, 'specular_strength', DEFAULT_MAT_SPEC_STRENGTH),
            "shine": getattr(mat, 'shininess', DEFAULT_MAT_SHININESS),
            "opacity": getattr(mat, 'opacity', DEFAULT_MAT_OPACITY),
            "base_c": list(mat.base_color),
            "amb_c": list(mat._ambient),
            "diff_c": list(mat._diffuse),
            "spec_c": list(mat._specular),
            "emis_c": list(mat.emission),
            "tex_paths": getattr(mat, 'tex_paths', {})
        })
        
        return data

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserializes configurations and dynamically rebuilds resources via ResourceManager."""
        from src.engine.resources.resource_manager import ResourceManager
        from src.engine.geometry.primitives import PrimitivesManager
        
        self.visible = bool(data.get("visible", True))
        self.is_proxy = bool(data.get("is_proxy", False))
        
        # --- Geometry Restitution Logic ---
        if "geometry_path" in data:
            path = data["geometry_path"]
            sub_name = data.get("submesh_name", "")
            if os.path.exists(path):
                mesh_list = ResourceManager.get_model(path)
                if mesh_list:
                    self.geometry = next((m for m in mesh_list if getattr(m, 'name', '') == sub_name), mesh_list[0])
        elif "primitive_name" in data:
            geom = PrimitivesManager.get_primitive(data["primitive_name"], False)
            if not geom: 
                geom = PrimitivesManager.get_primitive(data["primitive_name"], True)
            self.geometry = geom
        elif "math_formula" in data:
            try:
                from src.engine.geometry.math_surface import MathSurface
                f = data["math_formula"]
                r = data.get("math_ranges", [list(DEFAULT_MATH_RANGE), list(DEFAULT_MATH_RANGE), DEFAULT_MATH_RESOLUTION])
                self.geometry = MathSurface(f, (r[0][0], r[0][1]), (r[1][0], r[1][1]), r[2])
                self.geometry.formula_str = f
            except Exception: 
                pass
        elif "proxy_path" in data:
            path = data["proxy_path"]
            if "proxy" in path: 
                self.geometry = PrimitivesManager.get_proxy(os.path.basename(path))

        # --- Material Restitution Logic ---
        mat = self.material
        mat.use_advanced_mode = data.get("use_adv", False)
        mat.ambient_strength = data.get("amb_s", DEFAULT_MAT_AMB_STRENGTH)
        mat.diffuse_strength = data.get("diff_s", DEFAULT_MAT_DIFF_STRENGTH)
        mat.specular_strength = data.get("spec_s", DEFAULT_MAT_SPEC_STRENGTH)
        mat.shininess = data.get("shine", DEFAULT_MAT_SHININESS)
        mat.opacity = data.get("opacity", DEFAULT_MAT_OPACITY)
        
        mat.base_color = glm.vec3(*data.get("base_c", list(DEFAULT_MAT_BASE_COLOR)))
        mat._ambient = glm.vec3(*data.get("amb_c", list(DEFAULT_MAT_AMBIENT)))
        mat._diffuse = glm.vec3(*data.get("diff_c", list(DEFAULT_MAT_DIFFUSE)))
        mat._specular = glm.vec3(*data.get("spec_c", list(DEFAULT_MAT_SPECULAR)))
        mat.emission = glm.vec3(*data.get("emis_c", list(DEFAULT_MAT_EMISSION)))
        
        mat.tex_paths = data.get("tex_paths", {})
        
        for attr_name, path in mat.tex_paths.items():
            if path and os.path.exists(path):
                tid = ResourceManager.load_texture(path)
                if tid != 0:
                    setattr(mat, attr_name, tid)