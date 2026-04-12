import glm
from src.app.config import DEFAULT_SPAWN_POSITION, DEFAULT_SPAWN_ROTATION, DEFAULT_SPAWN_SCALE

class Transform:
    """
    Represents the spatial state of an entity in 3D space.
    Encapsulates position, rotation (Euler and Quaternion), and scale.
    """
    
    def __init__(self) -> None:
        self.position: glm.vec3 = glm.vec3(*DEFAULT_SPAWN_POSITION)
        self.rotation: glm.vec3 = glm.vec3(*DEFAULT_SPAWN_ROTATION) 
        self.scale: glm.vec3 = glm.vec3(*DEFAULT_SPAWN_SCALE)
        self.quat_rot: glm.quat = glm.quat(glm.radians(self.rotation)) 

    def get_matrix(self) -> glm.mat4:
        """
        Computes and returns the local transformation matrix (Model Matrix).
        Applies transformations in TRS order (Translate, Rotate, Scale).
        """
        mat = glm.mat4(1.0)
        mat = glm.translate(mat, self.position)
        mat = mat * glm.mat4_cast(self.quat_rot)
        mat = glm.scale(mat, self.scale)
        return mat

    def rotate_local(self, axis_char: str, angle_degrees: float) -> None:
        """
        Applies a local rotation around a specified primary axis ('X', 'Y', or 'Z').
        Accumulates the rotation using quaternion multiplication to prevent gimbal lock.
        """
        axis = glm.vec3(1, 0, 0)
        if axis_char == 'Y':
            axis = glm.vec3(0, 1, 0)
        elif axis_char == 'Z':
            axis = glm.vec3(0, 0, 1)
            
        q_rot = glm.angleAxis(glm.radians(angle_degrees), axis)
        self.quat_rot = self.quat_rot * q_rot
        self.rotation = glm.degrees(glm.eulerAngles(self.quat_rot))
        
    def sync_from_gui(self) -> None:
        """
        Synchronizes the quaternion rotational state from the current Euler angles.
        Typically invoked after direct Euler manipulation via the Inspector UI.
        """
        self.quat_rot = glm.quat(glm.radians(self.rotation))