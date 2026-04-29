"""
Animation Backend Manager.

Coordinates the animation timeline, keyframe interpolation, and state capture 
for entities. Enforces physical and logical constraints across components.
"""

import glm
import math
import copy
from typing import Dict, Any, Tuple, Optional, List

from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent, Keyframe
from src.engine.scene.entity import Entity
from src.app.config import TEXTURE_CHANNELS


# Constants for temporal precision and tracking
TIME_TOLERANCE: float = 0.01
UNFOCUSED_INDEX: int = -1
TEXTURE_MAP_ATTRS = tuple(TEXTURE_CHANNELS.values())

# Defines fields that must be excluded from timeline interpolation
NON_ANIMATABLE_PROPS = {
    "Transform": {"locked_axes"},
    "Camera": {"is_active", "mode"},
    "Light": {"type", "use_advanced_mode"},
    "Mesh": {
        "is_proxy", "geom_type", "geometry_path", "submesh_name", 
        "primitive_name", "math_formula", "math_ranges", "proxy_path", 
        "mat_use_advanced_mode"
    }
}


class AnimationBackendManager:
    """
    Core engine system responsible for managing keyframe states, 
    synchronizing Gizmo manipulations with timeline events, and 
    enforcing mathematical constraints (e.g., Light directional tracking).
    """
    
    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def _get_active_anim(self) -> Tuple[Optional[Entity], Optional[AnimationComponent]]:
        """Retrieves the animation component of the currently selected entity."""
        if self.scene.selected_index < 0:
            return None, None
            
        ent = self.scene.entities[self.scene.selected_index]
        anim = ent.get_component(AnimationComponent)
        
        # Auto-initialize base state if no keyframes exist
        if anim and not anim.keyframes:
            base_kf = Keyframe(0.0)
            self._capture_full_snapshot(ent, base_kf)
            anim.add_keyframe(base_kf)
            anim.active_keyframe_index = UNFOCUSED_INDEX
            
        return ent, anim

    def _filter_animatable_data(self, comp_name: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Strips out non-animatable properties from component data payloads."""
        if comp_name not in NON_ANIMATABLE_PROPS:
            return raw_data
            
        filtered = {}
        blacklist = NON_ANIMATABLE_PROPS[comp_name]
        for k, v in raw_data.items():
            if k not in blacklist:
                filtered[k] = v
        return filtered

    def _capture_full_snapshot(self, ent: Entity, kf: Keyframe) -> None:
        """Captures the current state of all relevant components into a Keyframe."""
        components_map = {
            "Transform": TransformComponent, 
            "Mesh": MeshRenderer, 
            "Light": LightComponent, 
            "Camera": CameraComponent
        }
        
        for comp_name, comp_cls in components_map.items():
            comp = ent.get_component(comp_cls)
            if comp and hasattr(comp, 'to_dict'):
                raw_data = comp.to_dict()
                kf.state[comp_name] = self._filter_animatable_data(comp_name, raw_data)
                
        self._enforce_transform_constraints(kf.state, ent, source_comp="Transform")

    def get_animation_info(self) -> Dict[str, Any]:
        """Provides metadata about the active animation sequence for UI synchronization."""
        _, anim = self._get_active_anim()
        if not anim: 
            return {}
        
        ui_times = [kf.time for kf in anim.keyframes]
        ui_active_idx = anim.active_keyframe_index
        target_time = ui_times[ui_active_idx] if 0 <= ui_active_idx < len(ui_times) else 0.0
        
        return {
            "has_anim": True,
            "active_idx": ui_active_idx,
            "times": ui_times,
            "target_time": target_time
        }

    def set_active_keyframe(self, ui_index: int) -> float:
        """Sets the playhead focus to a specific keyframe index."""
        _, anim = self._get_active_anim()
        if not anim: 
            return 0.0
        
        anim.active_keyframe_index = ui_index
        if 0 <= ui_index < len(anim.keyframes):
            return anim.keyframes[ui_index].time
        return 0.0

    def sync_gizmo_to_keyframe(self, current_time: float, is_hud_drag: bool = False) -> Tuple[bool, float]:
        """
        Bakes spatial modifications made via Viewport Gizmos into the current keyframe.
        Automatically creates a new keyframe if none exists at the current playhead.
        """
        ent, anim = self._get_active_anim()
        if not anim: 
            return False, current_time
        
        tf = ent.get_component(TransformComponent)
        light = ent.get_component(LightComponent)
        
        if not tf and not light: 
            return False, current_time

        is_new_kf = False
        target_kf = None
        
        # Resolve target keyframe
        if anim.active_keyframe_index != UNFOCUSED_INDEX and anim.active_keyframe_index < len(anim.keyframes):
            target_kf = anim.keyframes[anim.active_keyframe_index]
        else:
            existing_idx = next((i for i, kf in enumerate(anim.keyframes) if abs(kf.time - current_time) < TIME_TOLERANCE), -1)
            if existing_idx >= 0:
                target_kf = anim.keyframes[existing_idx]
                anim.active_keyframe_index = existing_idx
            else:
                target_kf = Keyframe(current_time)
                self._capture_full_snapshot(ent, target_kf)
                anim.add_keyframe(target_kf)
                anim.active_keyframe_index = anim.keyframes.index(target_kf)
                is_new_kf = True

        # Sync Transform
        if tf:
            if "Transform" not in target_kf.state:
                target_kf.state["Transform"] = self._filter_animatable_data("Transform", tf.to_dict())
            else:
                target_kf.state["Transform"].update(self._filter_animatable_data("Transform", tf.to_dict()))
        
        # Sync Light Constraints
        specific_attrs = ['yaw', 'pitch'] if is_hud_drag else None
        self._update_kf_from_component(target_kf, "Light", light, specific_attrs)
            
        self._enforce_transform_constraints(target_kf.state, ent, source_comp="Transform")
        return is_new_kf, target_kf.time

    def update_keyframe_property(self, current_time: float, comp_name: str, prop: str, value: Any) -> Tuple[bool, bool, float]:
        """Wrapper for single-property updates."""
        return self.update_keyframe_properties(current_time, comp_name, {prop: value})

    def update_keyframe_properties(self, current_time: float, comp_name: str, payload: Dict[str, Any]) -> Tuple[bool, bool, float]:
        """
        Commits batch property mutations to a keyframe, ensuring engine components 
        reflect the mathematical state immediately.
        """
        if comp_name not in ["Transform", "Mesh", "Light", "Camera"]:
            return False, False, 0.0

        ent, anim = self._get_active_anim()
        if not anim: 
            return False, False, 0.0

        is_new_kf = False
        target_kf = None

        if anim.active_keyframe_index != UNFOCUSED_INDEX and anim.active_keyframe_index < len(anim.keyframes):
            target_kf = anim.keyframes[anim.active_keyframe_index]
        else:
            existing_idx = next((i for i, kf in enumerate(anim.keyframes) if abs(kf.time - current_time) < TIME_TOLERANCE), -1)
            if existing_idx >= 0:
                target_kf = anim.keyframes[existing_idx]
                anim.active_keyframe_index = existing_idx
            else:
                target_kf = Keyframe(current_time)
                self._capture_full_snapshot(ent, target_kf)
                anim.add_keyframe(target_kf)
                anim.active_keyframe_index = anim.keyframes.index(target_kf)
                is_new_kf = True
                
        self._ensure_kf_component_state(target_kf, ent, comp_name)
        
        comp_cls_map = {
            "Transform": TransformComponent, 
            "Mesh": MeshRenderer, 
            "Light": LightComponent, 
            "Camera": CameraComponent
        }
        comp = ent.get_component(comp_cls_map.get(comp_name)) if comp_name in comp_cls_map else None
        
        for prop, value in payload.items():
            # Skip locked transform axes
            if comp_name == "Transform" and comp:
                if prop == "position" and comp.locked_axes.get("pos", False): continue
                if prop in ["rotation", "quat_rot"] and comp.locked_axes.get("rot", False): continue
                if prop == "scale" and comp.locked_axes.get("scl", False): continue

            # Handle non-animatable assignments immediately without injecting into timeline
            if comp_name in NON_ANIMATABLE_PROPS and prop in NON_ANIMATABLE_PROPS[comp_name]:
                if comp:
                    self._apply_property_to_component(comp, prop, value)
                continue
                
            # Material Textures Filtering
            if comp_name == "Mesh" and prop == "mat_tex_paths" and isinstance(value, dict):
                normalized = {
                    key: path.strip() for key, path in value.items()
                    if key in TEXTURE_MAP_ATTRS and isinstance(path, str) and path.strip()
                }
                target_kf.state[comp_name][prop] = normalized
            else:
                target_kf.state[comp_name][prop] = copy.deepcopy(value)
            
            # Special case: Pre-calculate quat_rot if Euler angles are provided
            if comp_name == "Transform" and prop == "rotation":
                q = glm.quat(glm.radians(glm.vec3(*value)))
                target_kf.state[comp_name]["quat_rot"] = [q.w, q.x, q.y, q.z]
                
            if comp:
                if comp_name == "Transform" and prop == "rotation":
                    comp.rotation = glm.vec3(*value)
                    comp.quat_rot = glm.quat(glm.radians(comp.rotation))
                    comp.is_dirty = True
                    if hasattr(comp, 'sync_from_gui'):
                        comp.sync_from_gui()
                else:
                    self._apply_property_to_component(comp, prop, value)

        self._enforce_transform_constraints(target_kf.state, ent, source_comp=comp_name)
        return True, is_new_kf, target_kf.time

    def _apply_property_to_component(self, comp: Any, prop: str, value: Any) -> None:
        """Helper to apply reflection-based properties dynamically."""
        if prop.startswith("mat_") and hasattr(comp, 'material') and comp.material:
            mat_prop = prop[4:]
            if isinstance(value, list) and len(value) == 3:
                setattr(comp.material, mat_prop, glm.vec3(*value))
            else:
                setattr(comp.material, mat_prop, value)
        elif hasattr(comp, prop):
            if isinstance(value, list) and len(value) == 3:
                setattr(comp, prop, glm.vec3(*value))
            else:
                setattr(comp, prop, value)
                
        if hasattr(comp, 'is_dirty'):
            comp.is_dirty = True
        if hasattr(comp, 'sync_from_gui'):
            comp.sync_from_gui()

    def add_and_focus_keyframe(self, time: float) -> int:
        """Creates a keyframe at the target timestamp and shifts UI focus to it."""
        ent, anim = self._get_active_anim()
        if not anim: 
            return -1
        
        for i, kf in enumerate(anim.keyframes):
            if abs(kf.time - time) < TIME_TOLERANCE:
                self._capture_full_snapshot(ent, kf)
                anim.active_keyframe_index = i
                return i
        
        kf = Keyframe(time)
        self._capture_full_snapshot(ent, kf)
        
        anim.add_keyframe(kf)
        anim.active_keyframe_index = anim.keyframes.index(kf)
        return anim.active_keyframe_index

    def handle_animation_property(self, ent: Entity, comp: AnimationComponent, prop: str, value: Any) -> None:
        """Processes high-level bulk operations originating from Timeline UI actions."""
        if prop == "REMOVE_KEYFRAME":
            real_idx = int(value)
            if 0 < real_idx < len(comp.keyframes):
                comp.remove_keyframe(real_idx)
                
        elif prop == "CLEAR_KEYFRAMES":
            if len(comp.keyframes) > 0:
                base_kf = comp.keyframes[0]
                comp.keyframes.clear()
                comp.keyframes.append(base_kf)
            comp.active_keyframe_index = UNFOCUSED_INDEX
            comp.duration = 0.0
            
        elif prop == "MOVE_KEYFRAME":
            real_idx = value.get("index", -1)
            if 0 < real_idx < len(comp.keyframes):
                new_time = max(0.0, value.get("time", 0.0))
                target_kf = comp.keyframes[real_idx]
                target_kf.time = new_time
                if hasattr(comp, '_sort_and_update_duration'):
                    comp._sort_and_update_duration()
                if target_kf in comp.keyframes:
                    comp.active_keyframe_index = comp.keyframes.index(target_kf)
                    
        elif prop == "loop":
            comp.loop = bool(value)
            
        elif prop == "MUTATE_KEYFRAMES":
            mode = value.get("mode")
            if mode == "UPDATE":
                for str_idx, new_time in value.get("data", {}).items():
                    idx = int(str_idx)
                    if 0 < idx < len(comp.keyframes):
                        comp.keyframes[idx].time = new_time
                if hasattr(comp, '_sort_and_update_duration'):
                    comp._sort_and_update_duration()
                    
            elif mode == "COPY":
                offset = value.get("offset", 0.0)
                new_kfs = [comp.keyframes[i].clone() for i in value.get("indices", []) if 0 <= i < len(comp.keyframes)]
                
                for nkf in new_kfs:
                    nkf.time += offset
                    if nkf.time > 0.001:
                        existing = [i for i, k in enumerate(comp.keyframes) if abs(k.time - nkf.time) < TIME_TOLERANCE]
                        if existing:
                            comp.keyframes[existing[0]] = nkf
                        else:
                            comp.keyframes.append(nkf)
                if hasattr(comp, '_sort_and_update_duration'):
                    comp._sort_and_update_duration()
                    
            elif mode == "DELETE_BULK":
                indices_to_remove = sorted([i for i in value.get("indices", []) if 0 < i < len(comp.keyframes)], reverse=True)
                for idx in indices_to_remove:
                    comp.keyframes.pop(idx)
                comp.active_keyframe_index = UNFOCUSED_INDEX
                if hasattr(comp, '_sort_and_update_duration'):
                    comp._sort_and_update_duration()

            elif mode == "DUPLICATE_KEYFRAME_RANGE":
                start_idx = int(value.get("start_idx", -1))
                end_idx = int(value.get("end_idx", -1))
                target_time = max(0.0, float(value.get("target_time", 0.0)))
                
                if 0 <= start_idx <= end_idx < len(comp.keyframes):
                    base_time = comp.keyframes[start_idx].time
                    new_kfs = []
                    
                    for i in range(start_idx, end_idx + 1):
                        new_kf = comp.keyframes[i].clone()
                        new_kf.time = target_time + (new_kf.time - base_time)
                        if new_kf.time > 0.001:
                            new_kfs.append(new_kf)
                    
                    for new_kf in new_kfs:
                        existing_idx = next((j for j, ext in enumerate(comp.keyframes) if abs(ext.time - new_kf.time) < TIME_TOLERANCE), -1)
                        if existing_idx >= 0:
                            comp.keyframes[existing_idx] = new_kf
                        else:
                            comp.keyframes.append(new_kf)
                    
                    if hasattr(comp, '_sort_and_update_duration'):
                        comp._sort_and_update_duration()
                    if new_kfs and new_kfs[-1] in comp.keyframes:
                        comp.active_keyframe_index = comp.keyframes.index(new_kfs[-1])

    def _update_kf_from_component(self, kf: Keyframe, comp_name: str, comp: Any, specific_attrs: Optional[List[str]] = None) -> None:
        """Utility to safely patch component data into an existing keyframe block."""
        if not comp: 
            return
            
        if comp_name not in kf.state:
            kf.state[comp_name] = self._filter_animatable_data(comp_name, comp.to_dict())
        else:
            if specific_attrs:
                for attr in specific_attrs:
                    if comp_name in NON_ANIMATABLE_PROPS and attr in NON_ANIMATABLE_PROPS[comp_name]:
                        continue
                    kf.state[comp_name][attr] = getattr(comp, attr, 0.0)
            else:
                kf.state[comp_name].update(self._filter_animatable_data(comp_name, comp.to_dict()))

    def _ensure_kf_component_state(self, kf: Keyframe, ent: Entity, comp_name: str) -> None:
        """Guarantees a dictionary key exists for a component before manipulation."""
        if comp_name not in kf.state:
            comp_cls = {
                "Transform": TransformComponent, 
                "Mesh": MeshRenderer, 
                "Light": LightComponent, 
                "Camera": CameraComponent
            }.get(comp_name)
            
            comp = ent.get_component(comp_cls) if comp_cls else None
            raw_data = comp.to_dict() if comp else {}
            kf.state[comp_name] = self._filter_animatable_data(comp_name, raw_data)

    def _enforce_transform_constraints(self, kf_state: Dict[str, Any], ent: Entity, source_comp: str = "Transform") -> None:
        """
        Critical Mathematical Synchronization.
        Ensures that if an entity is a Spotlight/Directional light, its Transform 
        Quaternion explicitly matches the logical Yaw/Pitch angles required by the shader, 
        and vice-versa, depending on which component initiated the change.
        """
        if "Transform" not in kf_state:
            return
            
        light = ent.get_component(LightComponent)
        tf = ent.get_component(TransformComponent)

        if light and tf and getattr(light, 'type', '') in ["Directional", "Spot"]:
            if "Light" not in kf_state:
                kf_state["Light"] = self._filter_animatable_data("Light", light.to_dict())

            # Transform drives Light direction (e.g. via Viewport Gizmo rotation)
            if source_comp == "Transform":
                q_vals = kf_state["Transform"].get("quat_rot", [1.0, 0.0, 0.0, 0.0])
                quat_rot = glm.quat(q_vals[0], q_vals[1], q_vals[2], q_vals[3])
                
                # Resolve global orientation
                if getattr(tf, 'entity', None) and getattr(tf.entity, 'parent', None):
                    parent_tf = tf.entity.parent.get_component(TransformComponent)
                    global_quat = (parent_tf.global_quat_rot * quat_rot) if parent_tf else quat_rot
                else:
                    global_quat = quat_rot

                # Derive Euler Yaw/Pitch from Forward vector
                forward = glm.vec3(glm.mat4_cast(global_quat) * glm.vec4(0, 0, -1, 0))
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, forward.y))))
                y_val = math.degrees(math.atan2(-forward.x, -forward.z))
                yaw = y_val if y_val >= 0 else y_val + 360.0
                
                kf_state["Light"]["pitch"] = pitch
                kf_state["Light"]["yaw"] = yaw
                light.pitch = pitch
                light.yaw = yaw

            # Light properties drive Transform (e.g. via UI Inspector input)
            elif source_comp == "Light":
                yaw = kf_state["Light"].get("yaw", getattr(light, 'yaw', 0.0))
                pitch = kf_state["Light"].get("pitch", getattr(light, 'pitch', 0.0))
                
                world_quat = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * \
                             glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
                             
                # Project back to local space if parented
                if getattr(tf, 'entity', None) and getattr(tf.entity, 'parent', None):
                    parent_tf = tf.entity.parent.get_component(TransformComponent)
                    local_quat = (glm.inverse(parent_tf.global_quat_rot) * world_quat) if parent_tf else world_quat
                else:
                    local_quat = world_quat
                    
                rotation_euler = glm.degrees(glm.eulerAngles(local_quat))
                
                kf_state["Transform"]["quat_rot"] = [local_quat.w, local_quat.x, local_quat.y, local_quat.z]
                kf_state["Transform"]["rotation"] = [rotation_euler.x, rotation_euler.y, rotation_euler.z]
                
                tf.quat_rot = local_quat
                tf.rotation = rotation_euler
                tf.is_dirty = True
                
                if hasattr(tf, 'sync_from_gui'):
                    tf.sync_from_gui()