"""
Resource Manager.
Centralized caching system for runtime assets to prevent redundant disk I/O.
"""

import json
from pathlib import Path
from typing import Dict, Set, Any
from src.engine.resources.model_loader import ModelLoader
from src.engine.resources.texture_loader import TextureLoader
from src.app.exceptions import ResourceError
from src.app.config import SHADERS_DIR

class ResourceManager:
    """
    Singleton-like interface managing VRAM resources and project file manifests.
    """
    
    _models: Dict[str, Any] = {}
    _textures: Dict[str, int] = {}
    _shaders: Dict[str, Any] = {} 
    
    project_models: Set[str] = set()
    project_textures: Set[str] = set()

    @staticmethod
    def get_shader(name: str) -> Any:
        """Retrieves a cached shader program, compiling defaults on first request."""
        if not ResourceManager._shaders:
            from src.engine.graphics.shader import Shader
            try:
                ResourceManager._shaders = {
                    "mat_standard": Shader(str(SHADERS_DIR / "materials" / "mat_standard.vert"), str(SHADERS_DIR / "materials" / "mat_standard.frag")),
                    "mat_unlit": Shader(str(SHADERS_DIR / "materials" / "mat_unlit.vert"), str(SHADERS_DIR / "materials" / "mat_unlit.frag")),
                    "pass_depth": Shader(str(SHADERS_DIR / "passes" / "pass_depth.vert"), str(SHADERS_DIR / "passes" / "pass_depth.frag")),
                    "pass_picking": Shader(str(SHADERS_DIR / "passes" / "pass_picking.vert"), str(SHADERS_DIR / "passes" / "pass_picking.frag")),
                    "editor_solid": Shader(str(SHADERS_DIR / "editor" / "editor_solid.vert"), str(SHADERS_DIR / "editor" / "editor_solid.frag")),
                    "editor_proxy": Shader(str(SHADERS_DIR / "editor" / "editor_proxy.vert"), str(SHADERS_DIR / "editor" / "editor_proxy.frag"))
                }
            except Exception as e:
                raise ResourceError(f"Failed to initialize core shader programs.\nDetails: {e}")
        
        if name not in ResourceManager._shaders: 
            raise ResourceError(f"Shader program not registered in cache: '{name}'")
            
        return ResourceManager._shaders[name]
    
    @classmethod
    def add_project_model(cls, path: str) -> None:
        """Registers a model path to the active project manifest."""
        cls.project_models.add(path)

    @classmethod
    def add_project_texture(cls, path: str) -> None:
        """Registers a texture path to the active project manifest."""
        cls.project_textures.add(path)
        
    @classmethod
    def clear_project_assets(cls) -> None:
        """Flushes the project manifest without destroying cached VRAM assets."""
        cls.project_models.clear()
        cls.project_textures.clear()

    @staticmethod
    def get_model(filepath: str) -> Any:
        """Retrieves a 3D model from the cache or dispatches a load request."""
        if filepath not in ResourceManager._models:
            ResourceManager._models[filepath] = ModelLoader.load(filepath)
        return ResourceManager._models[filepath]

    @staticmethod
    def load_texture(filepath: str) -> int:
        """Retrieves a texture ID from the VRAM cache or dispatches a load request."""
        if filepath not in ResourceManager._textures:
            tex_id = TextureLoader.load(filepath)
            ResourceManager._textures[filepath] = tex_id
            ResourceManager.add_project_texture(filepath)
                
        return ResourceManager._textures.get(filepath, 0)

    @staticmethod
    def save_project_file(file_path: str, data: Dict[str, Any]) -> None:
        """Serializes workspace state into JSON."""
        try:
            with Path(file_path).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise ResourceError(f"Failed to save project file to '{file_path}'.\nReason: {e}")

    @staticmethod
    def load_project_file(file_path: str) -> Dict[str, Any]:
        """Deserializes workspace state from JSON."""
        try:
            with Path(file_path).open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ResourceError(f"Failed to load project file from '{file_path}'.\nReason: {e}")