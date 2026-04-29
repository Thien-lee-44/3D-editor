"""
Transform Component.
Manages spatial state and hierarchical transformation matrices within the Scene Graph.
"""

import glm
from typing import Dict, Any, List
from src.engine.core.transform import Transform
from src.engine.scene.entity import Component
from src.app.config import DEFAULT_SPAWN_POSITION, DEFAULT_SPAWN_ROTATION, DEFAULT_SPAWN_SCALE

class TransformComponent(Transform, Component):
    """
    Acts as the primary anchor for spatial operations, integrating local Transform data
    with parent entities to resolve global coordinates.
    """
    
    def __init__(self) -> None:
        Transform.__init__(self)
        Component.__init__(self)
        self.locked_axes: Dict[str, bool] = {"pos": False, "rot": False, "scl": False}

    def get_matrix(self) -> glm.mat4:
        """Calculates the global transformation matrix by propagating through the entity hierarchy."""
        local_mat = super().get_matrix()
        
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                p_mat = parent_tf.get_matrix()
                
                # [SCALE GUARD]: Inherit Position/Rotation, but discard Parent Scale if locked.
                if self.locked_axes.get('scl', False):
                    global_mat = p_mat * local_mat
                    pos = glm.vec3(global_mat[3])
                    global_quat = parent_tf.global_quat_rot * self.quat_rot
                    return glm.translate(glm.mat4(1.0), pos) * glm.mat4_cast(global_quat) * glm.scale(glm.mat4(1.0), self.scale)
                
                return p_mat * local_mat
                
        return local_mat

    @property
    def global_position(self) -> glm.vec3:
        """Extracts absolute world-space position from the 4th column of the global matrix."""
        return glm.vec3(self.get_matrix()[3])

    @property
    def global_scale(self) -> glm.vec3:
        """Extracts absolute world-space scale via directional basis vector lengths."""
        mat = self.get_matrix()
        return glm.vec3(glm.length(mat[0]), glm.length(mat[1]), glm.length(mat[2]))

    @property
    def global_quat_rot(self) -> glm.quat:
        """Extracts absolute world-space rotation cleanly by stripping non-uniform scale data."""
        mat = self.get_matrix()
        sx, sy, sz = glm.length(mat[0]), glm.length(mat[1]), glm.length(mat[2])
        
        if sx == 0.0 or sy == 0.0 or sz == 0.0: 
            return glm.quat()
            
        if glm.determinant(glm.mat3(mat)) < 0: 
            sx = -sx 
            
        rot_mat = glm.mat3(glm.vec3(mat[0])/sx, glm.vec3(mat[1])/sy, glm.vec3(mat[2])/sz)
        return glm.quat_cast(rot_mat)

    def world_to_local_vec(self, world_vec: glm.vec3) -> glm.vec3:
        """Projects a world-space directional vector into this entity's local coordinates."""
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf:
                inv_mat = glm.inverse(parent_tf.get_matrix())
                return glm.vec3(inv_mat * glm.vec4(world_vec, 0.0))
        return world_vec

    def world_to_local_quat(self, world_quat: glm.quat) -> glm.quat:
        """Projects a world-space rotation into this entity's local rotational space."""
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                return glm.inverse(parent_tf.global_quat_rot) * world_quat
        return world_quat

    def set_from_matrix(self, matrix: glm.mat4) -> None:
        """Reconstructs local Position, Rotation, and Scale vectors from a 4x4 matrix."""
        self.position = glm.vec3(matrix[3])
        
        v_x, v_y, v_z = glm.vec3(matrix[0]), glm.vec3(matrix[1]), glm.vec3(matrix[2])
        sx, sy, sz = glm.length(v_x), glm.length(v_y), glm.length(v_z)
        
        if glm.determinant(glm.mat3(matrix)) < 0: 
            sx = -sx  
            
        self.scale = glm.vec3(sx, sy, sz)
        
        if sx == 0.0 or sy == 0.0 or sz == 0.0:
            self.quat_rot = glm.quat()
        else:
            self.quat_rot = glm.quat_cast(glm.mat3(v_x/sx, v_y/sy, v_z/sz))
            
        self.rotation = glm.degrees(glm.eulerAngles(self.quat_rot))

    def to_dict(self) -> Dict[str, List[float]]:
        """Serializes the local transform state."""
        return {
            "pos": [self.position.x, self.position.y, self.position.z],
            "rot": [self.rotation.x, self.rotation.y, self.rotation.z],
            "scl": [self.scale.x, self.scale.y, self.scale.z]
        }

    @staticmethod
    def _read_vec3(data: Dict[str, Any], keys: List[str], default: List[float]) -> glm.vec3:
        """Helper method to robustly deserialize vector data from generic payloads."""
        for key in keys:
            if key not in data:
                continue
            raw_val = data.get(key)
            if isinstance(raw_val, (list, tuple)) and len(raw_val) >= 3:
                return glm.vec3(float(raw_val[0]), float(raw_val[1]), float(raw_val[2]))
            if isinstance(raw_val, dict):
                if all(axis in raw_val for axis in ("x", "y", "z")):
                    return glm.vec3(float(raw_val["x"]), float(raw_val["y"]), float(raw_val["z"]))
        return glm.vec3(*default)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserializes transform state from a JSON payload."""
        self.position = self._read_vec3(data, ["pos", "position", "Position"], list(DEFAULT_SPAWN_POSITION))
        self.rotation = self._read_vec3(data, ["rot", "rotation", "Rotation"], list(DEFAULT_SPAWN_ROTATION))
        self.scale = self._read_vec3(data, ["scl", "scale", "Scale"], list(DEFAULT_SPAWN_SCALE))
        self.quat_rot = glm.quat(glm.radians(self.rotation))
        if hasattr(self, 'sync_from_gui'):
            self.sync_from_gui()