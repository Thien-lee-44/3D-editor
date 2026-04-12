import glm
from typing import Dict, Any, List
from src.engine.core.transform import Transform
from src.engine.scene.entity import Component

# Import configuration defaults
from src.app.config import DEFAULT_SPAWN_POSITION, DEFAULT_SPAWN_ROTATION, DEFAULT_SPAWN_SCALE

class TransformComponent(Transform, Component):
    """
    Manages spatial state (Position, Rotation, Scale) and computes hierarchical 
    transformation matrices relative to parent entities within the Scene Graph.
    Acts as the primary anchor for all spatial operations in the ECS.
    """
    
    def __init__(self) -> None:
        Transform.__init__(self)
        Component.__init__(self)

    def get_matrix(self) -> glm.mat4:
        """
        Calculates the global transformation matrix by walking up the entity hierarchy.
        Multiplies the local matrix by the parent's global matrix to inherit spatial state.
        """
        local_mat = super().get_matrix()
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                return parent_tf.get_matrix() * local_mat
        return local_mat

    @property
    def global_position(self) -> glm.vec3:
        """Extracts the absolute world-space position from the 4th column of the global matrix."""
        return glm.vec3(self.get_matrix()[3])

    @property
    def global_scale(self) -> glm.vec3:
        """Extracts the absolute world-space scale by calculating the length of the directional basis vectors."""
        mat = self.get_matrix()
        return glm.vec3(glm.length(mat[0]), glm.length(mat[1]), glm.length(mat[2]))

    @property
    def global_quat_rot(self) -> glm.quat:
        """
        Extracts the absolute world-space rotation. 
        Requires stripping out scale data to construct a pure, orthogonal rotation matrix 
        before converting to a Quaternion to prevent skewing/shearing artifacts.
        """
        mat = self.get_matrix()
        sx = glm.length(mat[0])
        sy = glm.length(mat[1])
        sz = glm.length(mat[2])
        
        if sx == 0.0 or sy == 0.0 or sz == 0.0: 
            return glm.quat()
            
        # Handle mirrored scales (negative determinants)
        if glm.determinant(glm.mat3(mat)) < 0: 
            sx = -sx 
            
        rot_mat = glm.mat3(glm.vec3(mat[0])/sx, glm.vec3(mat[1])/sy, glm.vec3(mat[2])/sz)
        return glm.quat_cast(rot_mat)

    def world_to_local_vec(self, world_vec: glm.vec3) -> glm.vec3:
        """Projects a directional vector from absolute world space into this entity's local coordinate system."""
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf:
                inv_mat = glm.inverse(parent_tf.get_matrix())
                return glm.vec3(inv_mat * glm.vec4(world_vec, 0.0))
        return world_vec

    def world_to_local_quat(self, world_quat: glm.quat) -> glm.quat:
        """Projects a rotation from absolute world space into this entity's local rotational space."""
        if self.entity and self.entity.parent:
            parent_tf = self.entity.parent.get_component(TransformComponent)
            if parent_tf: 
                return glm.inverse(parent_tf.global_quat_rot) * world_quat
        return world_quat

    def set_from_matrix(self, matrix: glm.mat4) -> None:
        """
        Matrix Decomposition Algorithm: Reconstructs the local Position, Rotation, 
        and Scale vectors mathematically from a baked 4x4 transformation matrix.
        Typically used when reparenting entities to maintain their visual world location.
        """
        self.position = glm.vec3(matrix[3])
        
        v_x = glm.vec3(matrix[0])
        v_y = glm.vec3(matrix[1])
        v_z = glm.vec3(matrix[2])
        
        sx = glm.length(v_x)
        sy = glm.length(v_y)
        sz = glm.length(v_z)
        
        if glm.determinant(glm.mat3(matrix)) < 0: 
            sx = -sx  
            
        self.scale = glm.vec3(sx, sy, sz)
        
        if sx == 0.0 or sy == 0.0 or sz == 0.0:
            self.quat_rot = glm.quat()
        else:
            self.quat_rot = glm.quat_cast(glm.mat3(v_x/sx, v_y/sy, v_z/sz))
            
        self.rotation = glm.degrees(glm.eulerAngles(self.quat_rot))

    def to_dict(self) -> Dict[str, List[float]]:
        """Serializes the local transform state for saving to disk."""
        return {
            "pos": [self.position.x, self.position.y, self.position.z],
            "rot": [self.rotation.x, self.rotation.y, self.rotation.z],
            "scl": [self.scale.x, self.scale.y, self.scale.z]
        }

    def from_dict(self, data: Dict[str, List[float]]) -> None:
        """Deserializes transform state from loaded JSON payload."""
        self.position = glm.vec3(*data.get("pos", list(DEFAULT_SPAWN_POSITION)))
        self.rotation = glm.vec3(*data.get("rot", list(DEFAULT_SPAWN_ROTATION)))
        self.scale = glm.vec3(*data.get("scl", list(DEFAULT_SPAWN_SCALE)))
        self.quat_rot = glm.quat(glm.radians(self.rotation))
        if hasattr(self, 'sync_from_gui'):
            self.sync_from_gui()