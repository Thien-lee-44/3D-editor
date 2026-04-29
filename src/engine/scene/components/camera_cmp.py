"""
Camera Component.

Defines the viewing frustum and constructs mathematical matrices 
used to project 3D geometry into 2D screen space.
"""

import glm
from typing import Dict, Any

from src.engine.scene.entity import Component
from src.engine.scene.components.transform_cmp import TransformComponent
from src.app.config import DEFAULT_CAMERA_FOV, DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR, DEFAULT_WINDOW_SIZE


class CameraComponent(Component):
    """
    Provides View and Projection matrices to the graphics pipeline.
    Supports both Perspective and Orthographic projection modes.
    """
    
    def __init__(self, mode: str = "Perspective") -> None:
        super().__init__()
        self.mode: str = mode
        self.fov: float = DEFAULT_CAMERA_FOV
        self.aspect: float = DEFAULT_WINDOW_SIZE[0] / max(DEFAULT_WINDOW_SIZE[1], 1)
        self.ortho_size: float = 5.0 
        self.is_active: bool = False
        self.near: float = DEFAULT_CAMERA_NEAR
        self.far: float = DEFAULT_CAMERA_FAR

    def get_view_matrix(self) -> glm.mat4:
        """
        Constructs the View Matrix using the LookAt algorithm.
        Transforms coordinates from World Space into Camera/Eye Space.
        """
        if not self.entity: 
            return glm.mat4(1.0)
            
        transform = self.entity.get_component(TransformComponent)
        if not transform: 
            return glm.mat4(1.0)
        
        rot_mat = glm.mat3_cast(transform.global_quat_rot)
        return glm.lookAt(transform.global_position, transform.global_position - rot_mat[2], rot_mat[1])

    def get_projection_matrix(self) -> glm.mat4:
        """
        Constructs the Projection Matrix mapped to Normalized Device Coordinates (NDC).
        """
        if self.mode == "Perspective":
            return glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        else:
            s = self.ortho_size
            a = self.aspect
            return glm.ortho(-s * a, s * a, -s, s, self.near, self.far)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes camera frustum parameters."""
        return {
            "mode": self.mode, 
            "fov": float(self.fov), 
            "ortho_size": float(self.ortho_size),
            "is_active": getattr(self, 'is_active', False),
            "near": float(self.near), 
            "far": float(self.far)
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserializes camera parameters from dictionary representation."""
        self.mode = data.get("mode", "Perspective")
        self.fov = float(data.get("fov", DEFAULT_CAMERA_FOV))
        self.ortho_size = float(data.get("ortho_size", data.get("ortho", 5.0)))
        self.is_active = bool(data.get("is_active", data.get("active", False)))
        self.near = float(data.get("near", DEFAULT_CAMERA_NEAR))
        self.far = float(data.get("far", DEFAULT_CAMERA_FAR))