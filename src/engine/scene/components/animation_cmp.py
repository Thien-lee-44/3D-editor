import glm
import copy
from typing import List, Dict, Any, Optional

class Keyframe:
    """
    Represents a discrete snapshot of an entity's state at a specific point in time.
    State bag is purely populated via Component native to_dict() serialization, 
    making it implicitly JSON-compatible and completely DRY.
    """
    def __init__(self, time: float) -> None:
        self.time: float = time
        self.state: Dict[str, Dict[str, Any]] = {}

    def clone(self) -> 'Keyframe':
        """Creates a deep copy of this keyframe state for duplication."""
        new_kf = Keyframe(self.time)
        new_kf.state = copy.deepcopy(self.state)
        return new_kf

    def serialize(self) -> Dict[str, Any]:
        """Packages the keyframe data natively."""
        return {
            "time": self.time, 
            "state": copy.deepcopy(self.state)
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'Keyframe':
        """Reconstructs a Keyframe instance directly from a native dictionary."""
        kf = Keyframe(data.get("time", 0.0))
        kf.state = copy.deepcopy(data.get("state", {}))
        return kf


class AnimationComponent:
    """
    Stores keyframe tracks and kinematic properties for dynamic entities.
    Acts as the central timeline for entity-specific state changes over time.
    """
    def __init__(self) -> None:
        self.is_active: bool = True
        self.keyframes: List[Keyframe] = []
        self.current_time: float = 0.0
        self.duration: float = 0.0
        self.loop: bool = False
        
        self.active_keyframe_index: int = -1 
        self.velocity: glm.vec3 = glm.vec3(0.0)
        self.angular_velocity: glm.vec3 = glm.vec3(0.0)
        
        # Temporary RAM cache for the Base State (t=0). 
        # NOT saved to JSON. Used dynamically by Animator during playback.
        self._base_state_cache: Dict[str, Dict[str, Any]] = {}

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self.keyframes.append(keyframe)
        self._sort_and_update_duration()

    def remove_keyframe(self, index: int) -> None:
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
            self.active_keyframe_index = -1 if not self.keyframes else 0
            self._sort_and_update_duration()

    def get_keyframe(self, index: int) -> Optional[Keyframe]:
        if 0 <= index < len(self.keyframes):
            return self.keyframes[index]
        return None

    def set_keyframe_time(self, index: int, new_time: float) -> None:
        if 0 <= index < len(self.keyframes):
            self.keyframes[index].time = new_time
            self._sort_and_update_duration()

    def _sort_and_update_duration(self) -> None:
        if not self.keyframes:
            self.duration = 0.0
            return
            
        active_kf = None
        if hasattr(self, 'active_keyframe_index') and 0 <= self.active_keyframe_index < len(self.keyframes):
            active_kf = self.keyframes[self.active_keyframe_index]
            
        self.keyframes.sort(key=lambda k: k.time)
        self.duration = self.keyframes[-1].time
        
        if active_kf in self.keyframes:
            self.active_keyframe_index = self.keyframes.index(active_kf)

    def serialize(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "current_time": self.current_time,
            "loop": self.loop,
            "keyframes": [k.serialize() for k in self.keyframes],
            "vel": [self.velocity.x, self.velocity.y, self.velocity.z],
            "ang_vel": [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z]
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.is_active = data.get("is_active", True)
        self.current_time = data.get("current_time", 0.0)
        self.loop = data.get("loop", False)
        self.active_keyframe_index = -1
        
        kf_data = data.get("keyframes", [])
        if kf_data:
            self.keyframes = [Keyframe.deserialize(k) for k in kf_data]
            for kf in self.keyframes:
                cam_state = kf.state.get("Camera")
                if isinstance(cam_state, dict):
                    if "active" in cam_state and "is_active" not in cam_state:
                        cam_state["is_active"] = bool(cam_state.pop("active"))
                    if "ortho" in cam_state and "ortho_size" not in cam_state:
                        cam_state["ortho_size"] = float(cam_state.pop("ortho"))
            if self.keyframes:
                self.active_keyframe_index = 0
        else:
            self.keyframes = []
            
        self._sort_and_update_duration()
        self.velocity = glm.vec3(*data.get("vel", [0.0, 0.0, 0.0]))
        self.angular_velocity = glm.vec3(*data.get("ang_vel", [0.0, 0.0, 0.0]))
        
        self._base_state_cache = {}
