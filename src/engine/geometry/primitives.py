"""
Geometry Primitives Provisioner.
Implements Lazy Initialization to cache foundational meshes and editor proxies.
"""

from pathlib import Path
from typing import Dict, Optional, Any
from src.engine.resources.resource_manager import ResourceManager
from src.app.config import MODELS_DIR

class PrimitivesManager:
    """
    Centralized loader for 2D/3D primitives and editor utility models.
    Delegates actual VRAM upload to the ResourceManager.
    """
    
    DIR_2D: Path = MODELS_DIR / "primitives" / "2d"
    DIR_3D: Path = MODELS_DIR / "primitives" / "3d"
    DIR_PROXIES: Path = MODELS_DIR / "proxies"

    @classmethod
    def _scan_dir(cls, directory: Path) -> Dict[str, str]:
        """
        Traverses a directory to map human-readable primitive names to absolute paths.
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
        """Retrieves paths for all available 2D flat primitives."""
        return cls._scan_dir(cls.DIR_2D)

    @classmethod
    def get_3d_paths(cls) -> Dict[str, str]: 
        """Retrieves paths for all available 3D volumetric primitives."""
        return cls._scan_dir(cls.DIR_3D)

    @classmethod
    def get_proxy_path(cls, filename: str) -> str: 
        """Resolves the absolute path for editor-only utility models."""
        return str((cls.DIR_PROXIES / filename).resolve()).replace('\\', '/')

    @classmethod
    def get_primitive(cls, name: str, is_2d: bool = False) -> Optional[Any]:
        """
        Retrieves the parsed BufferObject for a requested primitive.
        Loads into VRAM on first request.
        """
        paths = cls.get_2d_paths() if is_2d else cls.get_3d_paths()
        path_str = paths.get(name)
        
        if path_str and Path(path_str).exists():
            models = ResourceManager.get_model(path_str)
            if models: 
                return models[0] 
        return None

    @classmethod
    def get_proxy(cls, filename: str) -> Optional[Any]:
        """
        Retrieves the BufferObject for an editor-only proxy representation 
        (e.g., Camera or Light wireframes).
        """
        path_str = cls.get_proxy_path(filename)
        if Path(path_str).exists():
            models = ResourceManager.get_model(path_str)
            if models: 
                return models[0]
        return None