import os
import glm
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
    Couples raw geometric vertex data (BufferObject) with surface physical properties (Material).
    Serves as the primary identifier for objects meant to be pushed into the rendering pipeline.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.geometry: Optional[Any] = None    
        self.material = Material() 
        self.visible: bool = True
        
        # Identifies proxy meshes used only by the Editor (e.g. Light/Camera visual markers)
        self.is_proxy: bool = False

    def __deepcopy__(self, memo: dict) -> 'MeshRenderer':
        """
        Custom deep clone implementation required for prefab instantiation.
        Ensures materials are duplicated uniquely while Geometry pointers (BufferObjects) 
        can be shared across instances to save VRAM.
        """
        import copy
        new_obj = type(self)()
        memo[id(self)] = new_obj
        
        new_obj.visible = self.visible
        new_obj.is_proxy = self.is_proxy
        new_obj.geometry = self.geometry 
        new_obj.material = copy.deepcopy(self.material, memo)
        
        return new_obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes mesh references and material configuration.
        Implements strict Geometry Type routing to survive Undo/Redo JSON serialization.
        """
        data = {
            "visible": self.visible, 
            "is_proxy": getattr(self, 'is_proxy', False)
        }
        
        # =====================================================================
        # 1. ROBUST GEOMETRY ORIGIN SERIALIZATION
        # =====================================================================
        data["geom_type"] = "none"
        if self.geometry:
            geom_name = getattr(self.geometry, 'name', '')
            
            if hasattr(self.geometry, 'formula_str'):
                data["geom_type"] = "math"
                data["math_formula"] = self.geometry.formula_str
                data["math_ranges"] = [
                    getattr(self.geometry, 'x_range', list(DEFAULT_MATH_RANGE)),
                    getattr(self.geometry, 'y_range', list(DEFAULT_MATH_RANGE)),
                    getattr(self.geometry, 'resolution', DEFAULT_MATH_RESOLUTION)
                ]
            elif self.is_proxy:
                data["geom_type"] = "proxy"
                data["proxy_path"] = getattr(self.geometry, 'filepath', '') or geom_name
                
            elif hasattr(self.geometry, 'filepath') and getattr(self.geometry, 'filepath', ''):
                data["geom_type"] = "model"
                data["geometry_path"] = self.geometry.filepath
                data["submesh_name"] = geom_name
                
            elif geom_name in ["Cube", "Sphere", "Cylinder", "Cone", "Plane", "Quad", "Torus", "Grid"]:
                data["geom_type"] = "primitive"
                data["primitive_name"] = geom_name
            else:
                data["geom_type"] = "primitive"
                data["primitive_name"] = geom_name or "Cube"

        # =====================================================================
        # 2. MATERIAL STATE SERIALIZATION (Golden Rule Names)
        # =====================================================================
        mat = self.material
        data["mat_use_advanced_mode"] = getattr(mat, 'use_advanced_mode', False)
        data["mat_ambient_strength"] = float(getattr(mat, 'ambient_strength', DEFAULT_MAT_AMB_STRENGTH))
        data["mat_diffuse_strength"] = float(getattr(mat, 'diffuse_strength', DEFAULT_MAT_DIFF_STRENGTH))
        data["mat_specular_strength"] = float(getattr(mat, 'specular_strength', DEFAULT_MAT_SPEC_STRENGTH))
        data["mat_shininess"] = float(getattr(mat, 'shininess', DEFAULT_MAT_SHININESS))
        data["mat_opacity"] = float(getattr(mat, 'opacity', DEFAULT_MAT_OPACITY))
        
        data["mat_base_color"] = list(mat.base_color)
        data["mat__ambient"] = list(mat._ambient)
        data["mat__diffuse"] = list(mat._diffuse)
        data["mat__specular"] = list(mat._specular)
        data["mat_emission"] = list(mat.emission)
        
        data["mat_tex_paths"] = mat.get_tex_paths_snapshot() if hasattr(mat, 'get_tex_paths_snapshot') else {}
        
        return data

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Deserializes layout configurations and dynamically requests the ResourceManager 
        to rebuild missing geometry/texture objects into VRAM.
        """
        from src.engine.geometry.primitives import PrimitivesManager
        
        self.visible = bool(data.get("visible", True))
        self.is_proxy = bool(data.get("is_proxy", False))
        
        # =====================================================================
        # 1. ROBUST GEOMETRY RESTITUTION
        # =====================================================================
        geom_type = data.get("geom_type", "none")

        if geom_type == "model":
            path = data.get("geometry_path", "")
            sub_name = data.get("submesh_name", "")
            if os.path.exists(path):
                # Retrieve from VRAM Cache rather than re-uploading
                from src.engine.resources.resource_manager import ResourceManager
                mesh_list = ResourceManager.get_model(path)
                if mesh_list:
                    self.geometry = next((m for m in mesh_list if getattr(m, 'name', '') == sub_name), mesh_list[0])
        
        elif geom_type == "primitive":
            p_name = data.get("primitive_name", "Cube")
            geom = PrimitivesManager.get_primitive(p_name, False)
            if not geom: 
                geom = PrimitivesManager.get_primitive(p_name, True)
            self.geometry = geom
            
        elif geom_type == "math":
            try:
                from src.engine.geometry.math_surface import MathSurface
                f = data["math_formula"]
                r = data.get("math_ranges", [list(DEFAULT_MATH_RANGE), list(DEFAULT_MATH_RANGE), DEFAULT_MATH_RESOLUTION])
                self.geometry = MathSurface(f, (r[0][0], r[0][1]), (r[1][0], r[1][1]), r[2])
                self.geometry.formula_str = f
            except Exception: 
                pass
                
        elif geom_type == "proxy":
            path = data.get("proxy_path", "")
            if path:
                self.geometry = PrimitivesManager.get_proxy(os.path.basename(path))

        # =====================================================================
        # 2. MATERIAL RESTITUTION
        # =====================================================================
        mat = self.material
        mat.use_advanced_mode = bool(data.get("mat_use_advanced_mode", False))
        mat.ambient_strength = float(data.get("mat_ambient_strength", DEFAULT_MAT_AMB_STRENGTH))
        mat.diffuse_strength = float(data.get("mat_diffuse_strength", DEFAULT_MAT_DIFF_STRENGTH))
        mat.specular_strength = float(data.get("mat_specular_strength", DEFAULT_MAT_SPEC_STRENGTH))
        mat.shininess = float(data.get("mat_shininess", DEFAULT_MAT_SHININESS))
        mat.opacity = float(data.get("mat_opacity", DEFAULT_MAT_OPACITY))
        
        mat.base_color = glm.vec3(*data.get("mat_base_color", list(DEFAULT_MAT_BASE_COLOR)))
        mat._ambient = glm.vec3(*data.get("mat__ambient", list(DEFAULT_MAT_AMBIENT)))
        mat._diffuse = glm.vec3(*data.get("mat__diffuse", list(DEFAULT_MAT_DIFFUSE)))
        mat._specular = glm.vec3(*data.get("mat__specular", list(DEFAULT_MAT_SPECULAR)))
        mat.emission = glm.vec3(*data.get("mat_emission", list(DEFAULT_MAT_EMISSION)))
        
        mat.apply_texture_paths(data.get("mat_tex_paths", {}))
