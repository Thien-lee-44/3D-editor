import glm
import copy
from typing import Dict, Any, Tuple, Optional, List

from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent, Keyframe
from src.engine.scene.entity import Entity
from src.app.config import TEXTURE_CHANNELS

TIME_TOLERANCE: float = 0.01
BASE_STATE_INDEX: int = -2
UNFOCUSED_INDEX: int = -1
TEXTURE_MAP_ATTRS = tuple(TEXTURE_CHANNELS.values())

class AnimationBackendManager:
    """
    Core manager handling keyframe logic, time interpolation bridging, 
    and strict synchronization between the user interface and the 3D Engine.
    """
    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def _get_active_anim(self) -> Tuple[Optional[Entity], Optional[AnimationComponent]]:
        if self.scene.selected_index < 0:
            return None, None
        ent = self.scene.entities[self.scene.selected_index]
        return ent, ent.get_component(AnimationComponent)

    def _get_or_create_keyframe(self, anim: AnimationComponent, current_time: float) -> Tuple[Optional[Keyframe], bool]:
        if current_time <= TIME_TOLERANCE:
            anim.active_keyframe_index = BASE_STATE_INDEX 
            return None, False

        for i, kf in enumerate(anim.keyframes):
            if abs(kf.time - current_time) < TIME_TOLERANCE:
                anim.active_keyframe_index = i
                return kf, False
                
        new_kf = Keyframe(current_time)
        anim.add_keyframe(new_kf)
        anim.active_keyframe_index = anim.keyframes.index(new_kf)
        return new_kf, True

    def _capture_full_snapshot(self, ent: Entity, kf: Keyframe) -> None:
        components_map = {
            "Transform": TransformComponent, 
            "Mesh": MeshRenderer, 
            "Light": LightComponent, 
            "Camera": CameraComponent
        }
        for comp_name, comp_cls in components_map.items():
            comp = ent.get_component(comp_cls)
            if comp and hasattr(comp, 'to_dict'):
                kf.state[comp_name] = comp.to_dict()
                
        # [DATA GUARD]: Cleanse invalid structural data immediately after capture
        self._enforce_transform_constraints(kf.state, ent)

    def get_animation_info(self) -> Dict[str, Any]:
        _, anim = self._get_active_anim()
        if not anim: 
            return {}
        
        if anim.active_keyframe_index == BASE_STATE_INDEX:
            ui_active_idx = 0     
        elif anim.active_keyframe_index == UNFOCUSED_INDEX:
            ui_active_idx = UNFOCUSED_INDEX     
        else:
            ui_active_idx = anim.active_keyframe_index + 1 
            
        ui_times = [0.0] + [kf.time for kf in anim.keyframes]
        target_time = ui_times[ui_active_idx] if 0 <= ui_active_idx < len(ui_times) else 0.0
        
        return {
            "has_anim": True,
            "active_idx": ui_active_idx,
            "times": ui_times,
            "target_time": target_time
        }

    def set_active_keyframe(self, ui_index: int) -> float:
        _, anim = self._get_active_anim()
        if not anim: 
            return 0.0
        
        if ui_index == UNFOCUSED_INDEX:
            anim.active_keyframe_index = UNFOCUSED_INDEX 
            return 0.0
        elif ui_index == 0:
            anim.active_keyframe_index = BASE_STATE_INDEX 
            return 0.0
        else:
            real_idx = ui_index - 1
            anim.active_keyframe_index = real_idx
            if 0 <= real_idx < len(anim.keyframes):
                return anim.keyframes[real_idx].time
            return 0.0

    def sync_gizmo_to_keyframe(self, current_time: float, is_hud_drag: bool = False) -> Tuple[bool, float]:
        ent, anim = self._get_active_anim()
        if not anim: 
            return False, current_time
        
        tf = ent.get_component(TransformComponent)
        light = ent.get_component(LightComponent)
        
        if not tf and not light: 
            return False, current_time
        
        is_new_kf = False
        if 0 <= anim.active_keyframe_index < len(anim.keyframes):
            target_kf = anim.keyframes[anim.active_keyframe_index]
        elif anim.active_keyframe_index == BASE_STATE_INDEX:
            if current_time <= TIME_TOLERANCE:
                if hasattr(anim, '_base_state_cache'):
                    anim._base_state_cache.clear()
                return False, current_time
            target_kf, is_new_kf = self._get_or_create_keyframe(anim, current_time)
        elif current_time <= TIME_TOLERANCE:
            if hasattr(anim, '_base_state_cache'):
                anim._base_state_cache.clear()
            return False, current_time
        else:
            target_kf, is_new_kf = self._get_or_create_keyframe(anim, current_time)
            
        if target_kf:
            if is_hud_drag and tf:
                if "Transform" not in target_kf.state:
                    target_kf.state["Transform"] = tf.to_dict()
                else:
                    tf_dict = tf.to_dict()
                    target_kf.state["Transform"]["rotation"] = tf_dict.get("rotation", target_kf.state["Transform"].get("rotation"))
                    target_kf.state["Transform"]["quat_rot"] = tf_dict.get("quat_rot", target_kf.state["Transform"].get("quat_rot"))
                self._update_kf_from_component(target_kf, "Light", light, specific_attrs=['yaw', 'pitch'])
            else:
                if tf:
                    if "Transform" not in target_kf.state:
                        target_kf.state["Transform"] = tf.to_dict()
                    else:
                        tf_dict = tf.to_dict()
                        old_scale = target_kf.state["Transform"].get("scale", [1.0, 1.0, 1.0])
                        new_scale = tf_dict.get("scale", [1.0, 1.0, 1.0])
                        if sum(abs(n - o) for n, o in zip(new_scale, old_scale)) < 0.05:
                            tf_dict["scale"] = old_scale
                        target_kf.state["Transform"].update(tf_dict)
                self._update_kf_from_component(target_kf, "Light", light, specific_attrs=['yaw', 'pitch'])
                
            # [DATA GUARD]: Cleanse invalid structural data
            self._enforce_transform_constraints(target_kf.state, ent)
            return is_new_kf, target_kf.time
            
        return False, current_time

    def update_keyframe_property(self, current_time: float, comp_name: str, prop: str, value: Any) -> Tuple[bool, bool, float]:
        ent, anim = self._get_active_anim()
        if not anim: 
            return False, False, 0.0
        
        is_new_kf = False
        if 0 <= anim.active_keyframe_index < len(anim.keyframes):
            target_kf = anim.keyframes[anim.active_keyframe_index]
        elif anim.active_keyframe_index == BASE_STATE_INDEX:
            if current_time <= TIME_TOLERANCE:
                if hasattr(anim, '_base_state_cache'):
                    anim._base_state_cache.clear()
                return False, False, 0.0
            target_kf, is_new_kf = self._get_or_create_keyframe(anim, current_time)
        elif current_time <= TIME_TOLERANCE:
            if hasattr(anim, '_base_state_cache'):
                anim._base_state_cache.clear()
            return False, False, 0.0
        else:
            target_kf, is_new_kf = self._get_or_create_keyframe(anim, current_time)
            
        if not target_kf:
            return False, False, 0.0
            
        self._ensure_kf_component_state(target_kf, ent, comp_name)
        if comp_name == "Mesh" and prop == "mat_tex_paths" and isinstance(value, dict):
            normalized = {
                key: path.strip()
                for key, path in value.items()
                if key in TEXTURE_MAP_ATTRS and isinstance(path, str) and path.strip()
            }
            target_kf.state[comp_name][prop] = normalized
        else:
            target_kf.state[comp_name][prop] = copy.deepcopy(value)
        
        if comp_name == "Transform" and prop == "rotation":
            q = glm.quat(glm.radians(glm.vec3(*value)))
            target_kf.state[comp_name]["quat_rot"] = [q.w, q.x, q.y, q.z]
            
        # [DATA GUARD]: Cleanse invalid structural data
        self._enforce_transform_constraints(target_kf.state, ent)
        return True, is_new_kf, target_kf.time

    def add_and_focus_keyframe(self, time: float) -> int:
        ent, anim = self._get_active_anim()
        if not anim: 
            return -1
        
        if time <= TIME_TOLERANCE:
            anim.active_keyframe_index = BASE_STATE_INDEX
            return 0 
        
        for i, kf in enumerate(anim.keyframes):
            if abs(kf.time - time) < TIME_TOLERANCE:
                self._capture_full_snapshot(ent, kf)
                anim.active_keyframe_index = i
                return i + 1 
        
        kf = Keyframe(time)
        self._capture_full_snapshot(ent, kf)
        
        anim.add_keyframe(kf)
        anim.active_keyframe_index = anim.keyframes.index(kf)
        return anim.active_keyframe_index + 1 

    def handle_animation_property(self, ent: Entity, comp: AnimationComponent, prop: str, value: Any) -> None:
        if prop == "REMOVE_KEYFRAME":
            real_idx = int(value) - 1
            if real_idx >= 0:
                comp.remove_keyframe(real_idx)
        elif prop == "CLEAR_KEYFRAMES":
            comp.keyframes.clear()
            comp.active_keyframe_index = UNFOCUSED_INDEX
            comp.duration = 0.0
        elif prop == "MOVE_KEYFRAME":
            ui_idx = value.get("index", -1)
            real_idx = ui_idx - 1
            if 0 <= real_idx < len(comp.keyframes):
                new_time = value.get("time", 0.0)
                if new_time > TIME_TOLERANCE: 
                    comp.keyframes[real_idx].time = new_time
                    if hasattr(comp, '_sort_and_update_duration'):
                        comp._sort_and_update_duration()
        elif prop == "loop":
            comp.loop = bool(value)

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _update_kf_from_component(self, kf: Keyframe, comp_name: str, comp: Any, specific_attrs: Optional[List[str]] = None) -> None:
        if not comp: 
            return
            
        if comp_name not in kf.state:
            kf.state[comp_name] = comp.to_dict()
        else:
            if specific_attrs:
                for attr in specific_attrs:
                    kf.state[comp_name][attr] = getattr(comp, attr, 0.0)
            else:
                kf.state[comp_name].update(comp.to_dict())

    def _ensure_kf_component_state(self, kf: Keyframe, ent: Entity, comp_name: str) -> None:
        if comp_name not in kf.state:
            comp_cls = {
                "Transform": TransformComponent, 
                "Mesh": MeshRenderer, 
                "Light": LightComponent, 
                "Camera": CameraComponent
            }.get(comp_name)
            
            comp = ent.get_component(comp_cls) if comp_cls else None
            kf.state[comp_name] = comp.to_dict() if comp else {}

    def _enforce_transform_constraints(self, kf_state: Dict[str, Any], ent: Entity) -> None:
        """
        Applies hardcoded mathematical limits to proxy objects within the keyframe data.
        Ensures strict compliance even if the UI or Gizmo is bypassed via scripting or bugs.
        """
        if "Transform" not in kf_state:
            return
            
        light = ent.get_component(LightComponent)
        cam = ent.get_component(CameraComponent)
        tf = ent.get_component(TransformComponent)
        
        locked_pos, locked_rot, locked_scl = False, False, False
        if light:
            locked_scl = True
            l_type = getattr(light, 'type', '')
            if l_type == "Directional": locked_pos = True
            elif l_type == "Point": locked_rot = True
        elif cam or getattr(ent, 'is_group', False):
            locked_scl = True
            
        if tf and hasattr(tf, 'locked_axes'):
            locked_pos |= tf.locked_axes.get('pos', False)
            locked_rot |= tf.locked_axes.get('rot', False)
            locked_scl |= tf.locked_axes.get('scl', False)
            
        if locked_scl:
            kf_state["Transform"]["scale"] = [1.0, 1.0, 1.0]
        if locked_pos:
            kf_state["Transform"]["position"] = [0.0, 0.0, 0.0]
        if locked_rot:
            kf_state["Transform"]["rotation"] = [0.0, 0.0, 0.0]
            kf_state["Transform"]["quat_rot"] = [1.0, 0.0, 0.0, 0.0]
