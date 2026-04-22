import glm
import os
from typing import Any
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
    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    def evaluate(self, global_time: float, dt: float) -> None:
        if not self.scene: return

        for ent in self.scene.entities:
            anim = ent.get_component(AnimationComponent)
            if not anim or not anim.is_active: continue
                
            tf = ent.get_component(TransformComponent)
            if not tf: continue

            # Evaluate Keyframe logic based on playhead position
            if global_time <= 0.01:
                if not getattr(anim, "_base_state_cache", {}):
                    self._capture_base_state(anim, ent)
                self._restore_base_state(anim, ent)
            elif anim.keyframes:
                self._process_keyframes(anim, ent, global_time)
            
            # Kinematics Execution
            is_moving = anim.velocity.x != 0 or anim.velocity.y != 0 or anim.velocity.z != 0
            is_rotating = anim.angular_velocity.x != 0 or anim.angular_velocity.y != 0 or anim.angular_velocity.z != 0

            if is_moving:
                tf.position += anim.velocity * dt
                tf.is_dirty = True
                
            if is_rotating:
                tf.rotation += anim.angular_velocity * dt
                tf.quat_rot = glm.quat(glm.radians(tf.rotation)) 
                tf.is_dirty = True

    def _capture_base_state(self, anim: AnimationComponent, ent: Any) -> None:
        """Captures the live component state into the RAM cache."""
        anim._base_state_cache = {}
        for comp_name, comp_cls in COMP_MAP.items():
            comp = ent.get_component(comp_cls)
            if comp and hasattr(comp, 'to_dict'):
                anim._base_state_cache[comp_name] = comp.to_dict()

    def _restore_base_state(self, anim: AnimationComponent, ent: Any) -> None:
        """Reverts the entity to its cached base state. MUST NOT clear the cache here!"""
        if not hasattr(anim, '_base_state_cache') or not anim._base_state_cache:
            return
            
        kf_restore = Keyframe(0.0)
        kf_restore.state = anim._base_state_cache
        self._interpolate_keyframes(kf_restore, kf_restore, 0.0, ent)

    def _process_keyframes(self, anim: AnimationComponent, ent: Any, global_time: float) -> None:
        """Processes animation interpolation starting safely from the Base State."""
        if not hasattr(anim, '_base_state_cache') or not anim._base_state_cache:
            self._capture_base_state(anim, ent)
            
        kf0 = Keyframe(0.0)
        kf0.state = anim._base_state_cache
        
        full_timeline = [kf0] + anim.keyframes

        eval_time = global_time
        last_time = full_timeline[-1].time

        if eval_time >= last_time:
            if anim.loop and last_time > 0.0:
                eval_time %= last_time
            else:
                self._interpolate_keyframes(full_timeline[-1], full_timeline[-1], global_time, ent)
                return

        kf_start, kf_end = full_timeline[0], full_timeline[-1]
        for i in range(len(full_timeline) - 1):
            if full_timeline[i].time <= eval_time <= full_timeline[i+1].time:
                kf_start, kf_end = full_timeline[i], full_timeline[i+1]
                break

        self._interpolate_keyframes(kf_start, kf_end, eval_time, ent)

    def _interpolate_keyframes(self, kf_start: Keyframe, kf_end: Keyframe, eval_time: float, ent: Any) -> None:
        """Handles mathematical interpolation and strictly routes it to live components."""
        time_diff = kf_end.time - kf_start.time
        t = 1.0 if time_diff <= 0.0 else max(0.0, min(1.0, (eval_time - kf_start.time) / time_diff))

        for comp_name, props1 in kf_start.state.items():
            if comp_name not in COMP_MAP: continue
            comp = ent.get_component(COMP_MAP[comp_name])
            if not comp: continue
            
            props2 = kf_end.state.get(comp_name, {})
            
            for prop_name, val1 in props1.items():
                if prop_name == "rotation" and "quat_rot" in props1:
                    continue

                resolved_prop_name = prop_name
                if comp_name == "Camera" and prop_name in ("active", "is_active"):
                    resolved_prop_name = "is_active"
                    val2 = props2.get("is_active", props2.get("active", val1))
                elif comp_name == "Camera" and prop_name in ("ortho", "ortho_size"):
                    resolved_prop_name = "ortho_size"
                    val2 = props2.get("ortho_size", props2.get("ortho", val1))
                else:
                    val2 = props2.get(prop_name, val1)
                
                if isinstance(val1, bool):
                    if comp_name == "Camera" and resolved_prop_name == "is_active":
                        # Camera activation must switch exactly at the keyframe boundary.
                        new_val = val1 if t < 1.0 else val2
                    else:
                        new_val = val1 if t < 0.5 else val2
                elif isinstance(val1, (float, int)):
                    new_val = float(val1) + (float(val2) - float(val1)) * t
                elif isinstance(val1, list):
                    if len(val1) == 3:
                        v1, v2 = glm.vec3(*val1), glm.vec3(*val2)
                        v_mix = glm.mix(v1, v2, t)
                        new_val = [v_mix.x, v_mix.y, v_mix.z] 
                    elif len(val1) == 4 and prop_name == "quat_rot":
                        q1, q2 = glm.quat(*val1), glm.quat(*val2)
                        q_slerp = glm.slerp(q1, q2, t)
                        new_val = [q_slerp.w, q_slerp.x, q_slerp.y, q_slerp.z]
                    else:
                        new_val = val1
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    # Discrete payload (e.g., texture maps): step exactly at keyframe boundary
                    new_val = val1 if t < 1.0 else val2
                else:
                    # Non-interpolable payloads (e.g., strings/enums): step at boundary
                    new_val = val1 if t < 1.0 else val2
                    
                if comp_name == "Mesh" and prop_name.startswith("mat_"):
                    if prop_name == "mat_tex_paths" and isinstance(new_val, dict):
                        normalized_paths = {
                            key: path.strip()
                            for key, path in new_val.items()
                            if key in TEXTURE_MAP_ATTRS and isinstance(path, str) and path.strip()
                        }
                        current_paths = (
                            comp.material.get_tex_paths_snapshot()
                            if hasattr(comp.material, "get_tex_paths_snapshot")
                            else {}
                        )
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
                    else:
                        actual_prop = prop_name[4:]
                        if hasattr(comp.material, actual_prop):
                            # CRITICAL FIX: Ensure Material vectors are strictly glm objects
                            if isinstance(new_val, list):
                                if len(new_val) == 3:
                                    setattr(comp.material, actual_prop, glm.vec3(*new_val))
                                elif len(new_val) == 4:
                                    setattr(comp.material, actual_prop, glm.vec4(*new_val))
                                else:
                                    setattr(comp.material, actual_prop, new_val)
                            else:
                                setattr(comp.material, actual_prop, new_val)
                else:
                    if hasattr(comp, resolved_prop_name):
                        if isinstance(new_val, list):
                            if len(new_val) == 3:
                                setattr(comp, resolved_prop_name, glm.vec3(*new_val))
                            elif len(new_val) == 4 and resolved_prop_name == "quat_rot":
                                setattr(comp, resolved_prop_name, glm.quat(*new_val))
                            else:
                                setattr(comp, resolved_prop_name, new_val)
                        else:
                            setattr(comp, resolved_prop_name, new_val)
                
                if comp_name == "Transform":
                    comp.is_dirty = True
                    if prop_name == "quat_rot":
                        comp.rotation = glm.degrees(glm.eulerAngles(comp.quat_rot))
                elif comp_name == "Light":
                    if prop_name in ["yaw", "pitch"] and hasattr(comp, 'update_direction'):
                        comp.update_direction(comp.yaw, comp.pitch)
