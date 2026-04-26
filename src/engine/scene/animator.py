import glm
import os
from typing import Any, Dict
from src.engine.scene.scene import Scene
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent, Keyframe
from src.engine.resources.resource_manager import ResourceManager
from src.app.config import TEXTURE_CHANNELS

COMP_MAP = {
    "Transform": TransformComponent,
    "Mesh": MeshRenderer,
    "Light": LightComponent,
    "Camera": CameraComponent
}
TEXTURE_MAP_ATTRS = tuple(TEXTURE_CHANNELS.values())

class AnimatorSystem:
    """
    Mathematical interpolator evaluating animation states dynamically across entities.
    Operates strictly on Keyframe data provided by the AnimationComponent.
    """
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._last_eval_time = -1.0 

    def evaluate(self, global_time: float, dt: float, target_entity_id: int = -1) -> None:
        if not self.scene: 
            return

        if dt == 0.0 and getattr(self, '_last_eval_time', -1.0) == global_time and target_entity_id == -1:
            return
        self._last_eval_time = global_time

        for i, ent in enumerate(self.scene.entities):
            if target_entity_id != -1 and i != target_entity_id:
                continue

            anim = ent.get_component(AnimationComponent)
            if not anim or not anim.is_active: 
                continue
                
            tf = ent.get_component(TransformComponent)
            if not tf: 
                continue

            if anim.keyframes:
                if len(anim.keyframes) == 1:
                    # Two-way synchronization: For un-animated entities (only base state exists),
                    # push the live component state back to the keyframe to prevent snap-back.
                    self._sync_base_keyframe_from_component(anim.keyframes[0], ent)
                else:
                    self._process_keyframes(anim, ent, global_time)
            
            is_moving = anim.velocity.x != 0 or anim.velocity.y != 0 or anim.velocity.z != 0
            is_rotating = anim.angular_velocity.x != 0 or anim.angular_velocity.y != 0 or anim.angular_velocity.z != 0

            if is_moving:
                tf.position += anim.velocity * dt
                tf.is_dirty = True
                
            if is_rotating:
                tf.rotation += anim.angular_velocity * dt
                tf.quat_rot = glm.quat(glm.radians(tf.rotation)) 
                tf.is_dirty = True

    # =========================================================================
    # KEYFRAME PROCESSING
    # =========================================================================

    def _sync_base_keyframe_from_component(self, kf: Keyframe, ent: Any) -> None:
        """
        Maintains a continuous live-to-keyframe sync for static entities.
        Ensures that free-flight camera movements or gizmo drags map directly 
        into the base state automatically.
        """
        tf = ent.get_component(TransformComponent)
        if tf and "Transform" in kf.state:
            state_dict = kf.state["Transform"]
            state_dict["position"] = [float(tf.position.x), float(tf.position.y), float(tf.position.z)]
            state_dict["rotation"] = [float(tf.rotation.x), float(tf.rotation.y), float(tf.rotation.z)]
            state_dict["quat_rot"] = [float(tf.quat_rot.w), float(tf.quat_rot.x), float(tf.quat_rot.y), float(tf.quat_rot.z)]
            state_dict["scale"] = [float(tf.scale.x), float(tf.scale.y), float(tf.scale.z)]
            
        light = ent.get_component(LightComponent)
        if light and "Light" in kf.state:
            kf.state["Light"]["yaw"] = float(getattr(light, 'yaw', 0.0))
            kf.state["Light"]["pitch"] = float(getattr(light, 'pitch', 0.0))

        cam = ent.get_component(CameraComponent)
        if cam and "Camera" in kf.state:
            kf.state["Camera"]["fov"] = float(getattr(cam, 'fov', 45.0))
            kf.state["Camera"]["ortho_size"] = float(getattr(cam, 'ortho_size', 5.0))
            kf.state["Camera"]["near"] = float(getattr(cam, 'near', 0.1))
            kf.state["Camera"]["far"] = float(getattr(cam, 'far', 100.0))

    def _process_keyframes(self, anim: AnimationComponent, ent: Any, global_time: float) -> None:
        eval_time = global_time
        first_kf = anim.keyframes[0]
        last_kf = anim.keyframes[-1]

        if eval_time <= first_kf.time:
            self._interpolate_keyframes(first_kf, first_kf, eval_time, ent)
            return

        if eval_time >= last_kf.time:
            if anim.loop and last_kf.time > 0.0:
                eval_time %= last_kf.time
            else:
                self._interpolate_keyframes(last_kf, last_kf, global_time, ent)
                return

        kf_start, kf_end = first_kf, last_kf
        for i in range(len(anim.keyframes) - 1):
            if anim.keyframes[i].time <= eval_time <= anim.keyframes[i+1].time:
                kf_start, kf_end = anim.keyframes[i], anim.keyframes[i+1]
                break

        self._interpolate_keyframes(kf_start, kf_end, eval_time, ent)

    def _interpolate_keyframes(self, kf_start: Keyframe, kf_end: Keyframe, eval_time: float, ent: Any) -> None:
        time_diff = kf_end.time - kf_start.time
        t = 1.0 if time_diff <= 0.0 else max(0.0, min(1.0, (eval_time - kf_start.time) / time_diff))

        for comp_name, props1 in kf_start.state.items():
            if comp_name not in COMP_MAP: 
                continue
                
            comp = ent.get_component(COMP_MAP[comp_name])
            if not comp: 
                continue
            
            props2 = kf_end.state.get(comp_name, {})
            
            for prop_name, val1 in props1.items():
                if prop_name == "rotation" and "quat_rot" in props1:
                    continue

                val2 = props2.get(prop_name, val1)
                new_val = self._calculate_interpolated_value(comp_name, prop_name, val1, val2, t)
                
                self._apply_property(comp, comp_name, prop_name, new_val)
                
            self._post_process_component(comp, comp_name, props1)

    # =========================================================================
    # MATH & ROUTING HELPERS
    # =========================================================================

    def _calculate_interpolated_value(self, comp_name: str, prop_name: str, val1: Any, val2: Any, t: float) -> Any:
        if isinstance(val1, bool):
            return val1 if t < 0.5 else val2
            
        if isinstance(val1, (float, int)):
            return float(val1) + (float(val2) - float(val1)) * t
            
        if isinstance(val1, list):
            if len(val1) == 3:
                v1, v2 = glm.vec3(*val1), glm.vec3(*val2)
                v_mix = glm.mix(v1, v2, t)
                return [v_mix.x, v_mix.y, v_mix.z] 
            if len(val1) == 4 and prop_name == "quat_rot":
                q1, q2 = glm.quat(*val1), glm.quat(*val2)
                q_slerp = glm.slerp(q1, q2, t)
                return [q_slerp.w, q_slerp.x, q_slerp.y, q_slerp.z]
            return val1
            
        return val1 if t < 1.0 else val2

    def _apply_property(self, comp: Any, comp_name: str, prop_name: str, new_val: Any) -> None:
        if comp_name == "Mesh" and prop_name.startswith("mat_"):
            if prop_name == "mat_tex_paths" and isinstance(new_val, dict):
                self._apply_material_textures(comp, new_val)
            else:
                actual_prop = prop_name[4:]
                if hasattr(comp.material, actual_prop):
                    self._set_glm_or_scalar(comp.material, actual_prop, new_val)
        else:
            if hasattr(comp, prop_name):
                self._set_glm_or_scalar(comp, prop_name, new_val)

    def _set_glm_or_scalar(self, target: Any, attr_name: str, value: Any) -> None:
        if isinstance(value, list):
            if len(value) == 3:
                setattr(target, attr_name, glm.vec3(*value))
            elif len(value) == 4 and attr_name == "quat_rot":
                setattr(target, attr_name, glm.quat(*value))
            else:
                setattr(target, attr_name, value)
        else:
            setattr(target, attr_name, value)

    def _apply_material_textures(self, comp: Any, new_val: Dict[str, str]) -> None:
        normalized_paths = {
            key: path.strip() for key, path in new_val.items()
            if key in TEXTURE_MAP_ATTRS and isinstance(path, str) and path.strip()
        }
        current_paths = comp.material.get_tex_paths_snapshot() if hasattr(comp.material, "get_tex_paths_snapshot") else {}
        
        if normalized_paths != current_paths:
            if hasattr(comp.material, "apply_texture_paths"):
                comp.material.apply_texture_paths(normalized_paths)
            else:
                for tex_map in TEXTURE_MAP_ATTRS:
                    if hasattr(comp.material, tex_map):
                        setattr(comp.material, tex_map, 0)
                comp.material.tex_paths = dict(normalized_paths)
                for attr_name, t_path in comp.material.tex_paths.items():
                    if hasattr(comp.material, attr_name) and os.path.exists(t_path):
                        tid = ResourceManager.load_texture(t_path)
                        if tid != 0:
                            setattr(comp.material, attr_name, tid)

    def _post_process_component(self, comp: Any, comp_name: str, props1: Dict[str, Any]) -> None:
        if comp_name == "Transform":
            comp.is_dirty = True
            if "quat_rot" in props1:
                comp.rotation = glm.degrees(glm.eulerAngles(comp.quat_rot))
        elif comp_name == "Light":
            if ("yaw" in props1 or "pitch" in props1) and hasattr(comp, 'update_direction'):
                comp.update_direction(comp.yaw, comp.pitch)