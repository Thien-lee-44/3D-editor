import glm
from typing import Any
from src.engine.scene.scene import Scene
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent
from src.engine.scene.components.animation_cmp import AnimationComponent, Keyframe

class AnimatorSystem:
    """
    Computes and applies kinematic transformations and keyframe interpolations.
    Driven by an absolute global timeline to support playback scrubbing, looping, and clamping.
    """
    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    def evaluate(self, global_time: float, dt: float) -> None:
        if not self.scene:
            return

        for ent in self.scene.entities:
            anim = ent.get_component(AnimationComponent)
            if not anim or not anim.is_active:
                continue
                
            tf = ent.get_component(TransformComponent)
            if not tf:
                continue

            if anim.keyframes:
                self._process_keyframes(anim, ent, tf, global_time)
            
            is_moving = anim.velocity.x != 0 or anim.velocity.y != 0 or anim.velocity.z != 0
            is_rotating = anim.angular_velocity.x != 0 or anim.angular_velocity.y != 0 or anim.angular_velocity.z != 0

            if is_moving:
                tf.position += anim.velocity * dt
                tf.is_dirty = True
                
            if is_rotating:
                tf.rotation += anim.angular_velocity * dt
                tf.quat_rot = glm.quat(glm.radians(tf.rotation)) 
                tf.is_dirty = True

    def _process_keyframes(self, anim: AnimationComponent, ent: Any, tf: TransformComponent, global_time: float) -> None:
        # ---------------------------------------------------------------------
        # VIEWPORT AUTO-KEYING (EDIT MODE)
        # Forcefully populate the selected keyframe with the current Viewport 
        # state. We check if the Entity has the component (tf, mesh, light) 
        # rather than the Keyframe to ensure default/blank keyframes (like 0.0s)
        # are correctly initialized upon viewport interaction.
        # ---------------------------------------------------------------------
        if hasattr(anim, 'active_keyframe_index') and anim.active_keyframe_index >= 0:
            if anim.active_keyframe_index < len(anim.keyframes):
                kf = anim.keyframes[anim.active_keyframe_index]
                
                if tf:
                    kf.set_transform(tf.position, tf.quat_rot, tf.scale)
                    
                mesh = ent.get_component(MeshRenderer)
                if mesh: 
                    kf.set_mesh(mesh.visible)
                        
                light = ent.get_component(LightComponent)
                if light: 
                    kf.set_light(light.on, getattr(light, 'intensity', 1.0), getattr(light, 'color', glm.vec3(1.0)))
                
                return

        # ---------------------------------------------------------------------
        # NORMAL PLAYBACK / SCRUBBING INTERPOLATION
        # ---------------------------------------------------------------------
        if len(anim.keyframes) == 1:
            self._apply_keyframe_state(anim.keyframes[0], ent, tf)
            return

        eval_time = global_time
        last_time = anim.duration

        if eval_time >= last_time:
            if anim.loop and last_time > 0.0:
                eval_time %= last_time
            else:
                self._apply_keyframe_state(anim.keyframes[-1], ent, tf)
                return
                
        elif eval_time <= anim.keyframes[0].time:
            self._apply_keyframe_state(anim.keyframes[0], ent, tf)
            return

        kf_start = anim.keyframes[0]
        kf_end = anim.keyframes[-1]
        
        for i in range(len(anim.keyframes) - 1):
            if anim.keyframes[i].time <= eval_time <= anim.keyframes[i+1].time:
                kf_start = anim.keyframes[i]
                kf_end = anim.keyframes[i+1]
                break

        self._interpolate_keyframes(kf_start, kf_end, eval_time, ent, tf)

    def _interpolate_keyframes(self, kf_start: Keyframe, kf_end: Keyframe, eval_time: float, ent: Any, tf: TransformComponent) -> None:
        time_diff = kf_end.time - kf_start.time
        if time_diff <= 0.0:
            self._apply_keyframe_state(kf_end, ent, tf)
            return

        t = (eval_time - kf_start.time) / time_diff
        t = max(0.0, min(1.0, t)) 

        if kf_start.has_transform and kf_end.has_transform:
            tf.position = glm.mix(kf_start.position, kf_end.position, t)
            tf.scale = glm.mix(kf_start.scale, kf_end.scale, t)
            
            tf.quat_rot = glm.slerp(kf_start.rotation, kf_end.rotation, t)
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            
            tf.is_dirty = True

        if kf_start.has_light and kf_end.has_light:
            light = ent.get_component(LightComponent)
            if light:
                light.intensity = glm.mix(kf_start.light_intensity, kf_end.light_intensity, t)
                light.color = glm.mix(kf_start.light_color, kf_end.light_color, t)

        self._apply_discrete_states(kf_start, ent)

    def _apply_keyframe_state(self, kf: Keyframe, ent: Any, tf: TransformComponent) -> None:
        if kf.has_transform:
            tf.position = glm.vec3(kf.position)
            tf.quat_rot = glm.quat(kf.rotation)
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            tf.scale = glm.vec3(kf.scale)
            tf.is_dirty = True
            
        if kf.has_light:
            light = ent.get_component(LightComponent)
            if light:
                light.on = kf.light_on
                light.intensity = kf.light_intensity
                light.color = glm.vec3(kf.light_color)
                
        self._apply_discrete_states(kf, ent)

    def _apply_discrete_states(self, kf: Keyframe, ent: Any) -> None:
        if kf.has_mesh:
            mesh = ent.get_component(MeshRenderer)
            if mesh:
                mesh.visible = kf.mesh_visible