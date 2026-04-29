"""
Transform Component.

Manages spatial state (Position, Rotation, Scale) and computes hierarchical 
transformation matrices relative to parent entities within the Scene Graph.
"""

import glm
from typing import Dict, Any

from src.engine.core.transform import Transform
from src.engine.scene.entity import Component
from src.app.config import DEFAULT_SPAWN_POSITION, DEFAULT_SPAWN_ROTATION, DEFAULT_SPAWN_SCALE


class TransformComponent(Transform, Component):
    """
    Extends base Transform to support hierarchical constraints and aggressive 
    matrix caching to minimize redundant matrix multiplications during render loops.
    """
    
    EPSILON = 1e-6
    
    def __init__(self) -> None:
        Transform.__init__(self)
        Component.__init__(self)
        self.locked_axes: Dict[str, bool] = {"pos": False, "rot": False, "scl": False}
        
        # Optimization Core: Cache State
        self._is_dirty: bool = True
        self._global_matrix_cache: glm.mat4 = glm.mat4(1.0)
        
        # State-Hash: Tracks silent mutations from external systems (Gizmos/Input)
        self._last_local_state: tuple = ()

    @property
    def is_dirty(self) -> bool:
        """Returns the current evaluation state of the matrix cache."""
        return self._is_dirty

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        """
        Intercepts dirty flag assignments to recursively invalidate the cache 
        of all descendants in the Scene Graph.
        """
        self._is_dirty = value
        
        # Propagate the invalidation down the hierarchy tree
        if value and self.entity and hasattr(self.entity, 'children'):
            for child in self.entity.children:
                child_tf = child.get_component(TransformComponent)
                if child_tf and not child_tf._is_dirty:
                    child_tf.is_dirty = True

    def get_matrix(self) -> glm.mat4:
        """
        Calculates the global transformation matrix.
        Bypasses multiplication entirely if the spatial state hasn't mutated.
        """
        # [STATE-HASH GUARD]: Detect silent mutations bypassing setter triggers
        current_state = (
            self.position.x, self.position.y, self.position.z,
            self.quat_rot.w, self.quat_rot.x, self.quat_rot.y, self.quat_rot.z,
            self.scale.x, self.scale.y, self.scale.z
        )
        
        if current_state != self._last_local_state:
            self._last_local_state = current_state
            self.is_dirty = True  
            
        if not self._is_dirty:
            return self._global_matrix_cache

        local_mat = super().get_matrix()
        
        if self.entity and getattr(self.entity, 'parent', None):
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                p_mat = parent_tf.get_matrix()
                raw_global_mat = p_mat * local_mat
                
                if not any(self.locked_axes.values()):
                    final_mat = raw_global_mat
                else:
                    final_pos = self.position if self.locked_axes.get("pos", False) else glm.vec3(raw_global_mat[3])
                    final_quat = self.quat_rot if self.locked_axes.get("rot", False) else parent_tf.global_quat_rot * self.quat_rot
                        
                    if self.locked_axes.get("scl", False):
                        final_scl = self.scale
                    else:
                        sx = glm.length(glm.vec3(raw_global_mat[0]))
                        sy = glm.length(glm.vec3(raw_global_mat[1]))
                        sz = glm.length(glm.vec3(raw_global_mat[2]))
                        if glm.determinant(glm.mat3(raw_global_mat)) < 0:
                            sx = -sx
                        final_scl = glm.vec3(sx, sy, sz)
                        
                    final_mat = glm.translate(glm.mat4(1.0), final_pos) * glm.mat4_cast(final_quat) * glm.scale(glm.mat4(1.0), final_scl)
            else:
                final_mat = local_mat
        else:
            final_mat = local_mat
            
        # Bake to cache
        self._global_matrix_cache = final_mat
        self._is_dirty = False
        return self._global_matrix_cache

    @property
    def global_position(self) -> glm.vec3:
        """Extracts absolute world-space position from the matrix."""
        return glm.vec3(self.get_matrix()[3])

    @property
    def global_scale(self) -> glm.vec3:
        """Extracts absolute world-space scale from basis vectors."""
        mat = self.get_matrix()
        return glm.vec3(glm.length(mat[0]), glm.length(mat[1]), glm.length(mat[2]))

    @property
    def global_quat_rot(self) -> glm.quat:
        """Extracts absolute world-space rotation stripped of scale data."""
        mat = self.get_matrix()
        sx, sy, sz = glm.length(mat[0]), glm.length(mat[1]), glm.length(mat[2])
        
        if sx == 0.0 or sy == 0.0 or sz == 0.0: 
            return glm.quat()
            
        if glm.determinant(glm.mat3(mat)) < 0: sx = -sx 
            
        rot_mat = glm.mat3(glm.vec3(mat[0])/sx, glm.vec3(mat[1])/sy, glm.vec3(mat[2])/sz)
        return glm.quat_cast(rot_mat)

    def world_to_local_vec(self, world_vec: glm.vec3) -> glm.vec3:
        """Projects a directional vector into this entity's local coordinate system."""
        if self.entity and getattr(self.entity, 'parent', None):
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf:
                inv_mat = glm.inverse(parent_tf.get_matrix())
                return glm.vec3(inv_mat * glm.vec4(world_vec, 0.0))
        return world_vec

    def world_to_local_quat(self, world_quat: glm.quat) -> glm.quat:
        """Projects a rotation into this entity's local rotational space."""
        if self.entity and getattr(self.entity, 'parent', None):
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                return glm.inverse(parent_tf.global_quat_rot) * world_quat
        return world_quat

    def set_from_matrix(self, matrix: glm.mat4) -> None:
        """
        Matrix Decomposition: Reconstructs local vectors mathematically from a 4x4 matrix.
        """
        pos = glm.vec3(matrix[3])
        v_x, v_y, v_z = glm.vec3(matrix[0]), glm.vec3(matrix[1]), glm.vec3(matrix[2])
        sx, sy, sz = glm.length(v_x), glm.length(v_y), glm.length(v_z)
        
        if glm.determinant(glm.mat3(matrix)) < 0: sx = -sx  
            
        if not self.locked_axes.get("pos", False):
            self.position = pos
            
        if not self.locked_axes.get("scl", False):
            self.scale = glm.vec3(sx, sy, sz)
       
        if not self.locked_axes.get("rot", False):
            nsx = sx if sx != 0 else self.EPSILON
            nsy = sy if sy != 0 else self.EPSILON
            nsz = sz if sz != 0 else self.EPSILON
            
            self.quat_rot = glm.quat_cast(glm.mat3(v_x/nsx, v_y/nsy, v_z/nsz))
            self.rotation = glm.degrees(glm.eulerAngles(self.quat_rot))
            
        self.is_dirty = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the local transform state."""
        return {
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": [self.rotation.x, self.rotation.y, self.rotation.z],
            "scale": [self.scale.x, self.scale.y, self.scale.z],
            "quat_rot": [self.quat_rot.w, self.quat_rot.x, self.quat_rot.y, self.quat_rot.z],
            "locked_axes": self.locked_axes
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserializes transform state from saved data."""
        self.position = glm.vec3(*data.get("position", list(DEFAULT_SPAWN_POSITION)))
        self.rotation = glm.vec3(*data.get("rotation", list(DEFAULT_SPAWN_ROTATION)))
        self.scale = glm.vec3(*data.get("scale", list(DEFAULT_SPAWN_SCALE)))
        
        if "quat_rot" in data:
            self.quat_rot = glm.quat(*data["quat_rot"])
        else:
            self.quat_rot = glm.quat(glm.radians(self.rotation))

        if "locked_axes" in data:
            self.locked_axes = data["locked_axes"]
            
        if hasattr(self, 'sync_from_gui'):
            self.sync_from_gui()
            
        self.is_dirty = True