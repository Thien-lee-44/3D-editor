import os
from pathlib import Path
from typing import Dict, Optional, Any
from src.engine.resources.resource_manager import ResourceManager
from src.app.config import MODELS_DIR

class PrimitivesManager:
    """
    Acts as a centralized provisioner for foundational geometric meshes (Cube, Sphere, etc.) 
    and Editor proxy models (Camera/Light icons).
    Architecture Note: Employs a Lazy Initialization pattern. Primitive BufferObjects are 
    only loaded into VRAM upon their first request, then cached globally by the ResourceManager.
    """
    
    DIR_2D: Path = MODELS_DIR / "primitives" / "2d"
    DIR_3D: Path = MODELS_DIR / "primitives" / "3d"
    DIR_PROXIES: Path = MODELS_DIR / "proxies"

    @classmethod
    def _scan_dir(cls, directory: Path) -> Dict[str, str]:
        """
        Traverses a target directory and constructs a mapping of formatted, 
        human-readable primitive names to their absolute file paths.
        """
        results: Dict[str, str] = {}
        if directory.exists():
            for f in directory.iterdir():
                if f.suffix in ('.obj', '.ply'):
                    name = f.stem.replace('_', ' ').title()
                    results[name] = str(f.resolve()).replace('\\', '/')
        return results

    @classmethod
    def get_2d_paths(cls) -> Dict[str, str]: 
        """Returns a dictionary of all available 2D flat primitives."""
        return cls._scan_dir(cls.DIR_2D)

    @classmethod
    def get_3d_paths(cls) -> Dict[str, str]: 
        """Returns a dictionary of all available 3D volumetric primitives."""
        return cls._scan_dir(cls.DIR_3D)

    @classmethod
    def get_proxy_path(cls, filename: str) -> str: 
        """Resolves the absolute path for editor-only utility models."""
        return str((cls.DIR_PROXIES / filename).resolve()).replace('\\', '/')

    @classmethod
    def get_primitive(cls, name: str, is_2d: bool = False) -> Optional[Any]:
        """
        Retrieves the parsed BufferObject for a requested primitive.
        Delegates the actual disk I/O and GPU upload to the ResourceManager.
        """
        paths = cls.get_2d_paths() if is_2d else cls.get_3d_paths()
        path = paths.get(name)
        
        if path and os.path.exists(path):
            models = ResourceManager.get_model(path)
            if models: 
                return models[0] 
        return None

    @classmethod
    def get_proxy(cls, filename: str) -> Optional[Any]:
        """
        Retrieves the BufferObject for an editor-only proxy representation.
        Proxies are typically wireframes or unlit solid icons representing invisible entities.
        """
        path = cls.get_proxy_path(filename)
        if os.path.exists(path):
            models = ResourceManager.get_model(path)
            if models: 
                return models[0]
        return None