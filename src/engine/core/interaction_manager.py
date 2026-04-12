import glm
import math
from typing import Tuple, Optional, List, Dict, Any
from src.engine.scene.components import CameraComponent, TransformComponent, MeshRenderer

from src.app.config import CAMERA_MOVE_SPEED, CAMERA_ROTATION_SPEED, HUD_AXIS_PADDING, HUD_AXIS_SCALE

class MathUtils:
    """Contains low-level intersection algorithms for 3D picking and collision."""
    
    @staticmethod
    def ray_intersect_aabb(ro: glm.vec3, rd: glm.vec3, bmin: glm.vec3, bmax: glm.vec3) -> Tuple[bool, float, Optional[glm.vec3]]:
        """Slab method calculation for Ray vs Axis-Aligned Bounding Box intersection."""
        tmin = (bmin.x - ro.x) / (rd.x if rd.x != 0 else 1e-6)
        tmax = (bmax.x - ro.x) / (rd.x if rd.x != 0 else 1e-6)
        nx = -1 if rd.x > 0 else 1
        if tmin > tmax: tmin, tmax, nx = tmax, tmin, -nx
        
        tymin = (bmin.y - ro.y) / (rd.y if rd.y != 0 else 1e-6)
        tymax = (bmax.y - ro.y) / (rd.y if rd.y != 0 else 1e-6)
        ny = -1 if rd.y > 0 else 1
        if tymin > tymax: tymin, tymax, ny = tymax, tymin, -ny
        
        if (tmin > tymax) or (tymin > tmax): return False, -1, None
        
        hit_normal = glm.vec3(nx, 0, 0)
        if tymin > tmin: 
            tmin = tymin
            hit_normal = glm.vec3(0, ny, 0)
        if tymax < tmax: tmax = tymax
        
        tzmin = (bmin.z - ro.z) / (rd.z if rd.z != 0 else 1e-6)
        tzmax = (bmax.z - ro.z) / (rd.z if rd.z != 0 else 1e-6)
        nz = -1 if rd.z > 0 else 1
        if tzmin > tzmax: tzmin, tzmax, nz = tzmax, tzmin, -nz
        
        if (tmin > tzmax) or (tzmin > tmax): return False, -1, None
        if tzmin > tmin: 
            tmin = tzmin
            hit_normal = glm.vec3(0, 0, nz)
            
        return True, tmin, hit_normal

    @staticmethod
    def ray_intersect_ring(ro: glm.vec3, rd: glm.vec3, normal: glm.vec3, radius: float, thick: float) -> Tuple[bool, float]:
        """Approximates a Ray vs Torus intersection used for rotation gizmo evaluation."""
        if abs(normal.x) > 0.5: axis, u, v = 0, 1, 2
        elif abs(normal.y) > 0.5: axis, u, v = 1, 0, 2
        else: axis, u, v = 2, 0, 1
        
        o_u, o_v, o_w = ro[u], ro[v], ro[axis]
        d_u, d_v, d_w = rd[u], rd[v], rd[axis]
        r_out = radius + thick
        r_in = max(0.0, radius - thick)
        half_h = thick * 1.5 
        
        hit, min_t = False, float('inf')
        A = d_u**2 + d_v**2
        B = 2.0 * (o_u * d_u + o_v * d_v)
        
        if A > 1e-6:
            for C in [o_u**2 + o_v**2 - r_out**2, o_u**2 + o_v**2 - r_in**2]:
                delta = B**2 - 4 * A * C
                if delta >= 0:
                    sq_delta = math.sqrt(delta)
                    for t in [(-B - sq_delta) / (2 * A), (-B + sq_delta) / (2 * A)]:
                        if t > 0 and abs(o_w + t * d_w) <= half_h: 
                            hit, min_t = True, min(min_t, t)
        if abs(d_w) > 1e-6:
            for t in [(half_h - o_w) / d_w, (-half_h - o_w) / d_w]:
                if t > 0 and r_in**2 <= (o_u + t * d_u)**2 + (o_v + t * d_v)**2 <= r_out**2: 
                    hit, min_t = True, min(min_t, t)
                    
        return hit, min_t if hit else -1

class InteractionMath:
    """Provides translation functions between 2D screen input vectors and 3D world transformations."""
    
    @staticmethod
    def calc_gizmo_rotate_angle(dx: float, dy: float, active_axis: str, quat_rot: glm.quat, view_matrix: glm.mat4, speed_multiplier: float) -> float:
        w_axis = glm.vec3(glm.mat4_cast(quat_rot) * glm.vec4((1,0,0) if active_axis=='X' else ((0,1,0) if active_axis=='Y' else (0,0,1)), 0))
        view_axis = glm.vec3(view_matrix * glm.vec4(w_axis, 0.0))
        dir_2d = glm.normalize(glm.vec2(-view_axis.y, view_axis.x)) if glm.length(glm.vec2(view_axis.x, view_axis.y)) > 0.001 else glm.vec2(1,0)
        return glm.dot(glm.vec2(dx, dy), dir_2d) * speed_multiplier

    @staticmethod
    def calc_arcball_rotation(dx: float, dy: float, quat_rot: glm.quat, view_matrix: glm.mat4, speed_x: float, speed_y: float) -> Tuple[glm.quat, glm.vec3]:
        cam_right = glm.vec3(glm.inverse(view_matrix)[0])
        cam_up = glm.vec3(glm.inverse(view_matrix)[1])
        q_x = glm.angleAxis(glm.radians(dy * speed_y), cam_right)
        q_y = glm.angleAxis(glm.radians(dx * speed_x), cam_up)
        new_quat = q_y * q_x * quat_rot
        return new_quat, glm.degrees(glm.eulerAngles(new_quat))

    @staticmethod
    def calc_gizmo_move_offset(dx: float, dy: float, active_axis: str, quat_rot: glm.quat, dir_2d: glm.vec2, dist: float, speed: float) -> glm.vec3:
        loc_dir = glm.vec3(1,0,0) if active_axis=='X' else (glm.vec3(0,1,0) if active_axis=='Y' else glm.vec3(0,0,1))
        return glm.vec3(glm.mat4_cast(quat_rot) * glm.vec4(loc_dir * (glm.dot(glm.vec2(dx, dy), dir_2d) * dist * speed), 0.0))

    @staticmethod
    def calc_gizmo_scale_amount(dx: float, dy: float, dir_2d: glm.vec2, speed: float) -> float:
        return glm.dot(glm.vec2(dx, dy), dir_2d) * speed

    @staticmethod
    def calc_free_move_offset(dx: float, dy: float, view_matrix: glm.mat4, dist: float, speed: float) -> glm.vec3:
        cam_right = glm.vec3(glm.inverse(view_matrix)[0])
        cam_up = glm.vec3(glm.inverse(view_matrix)[1])
        return cam_right * (dx * dist * speed) + cam_up * (dy * dist * speed)

class InteractionManager:
    """Manages scene-wide UI interactions including entity picking, gizmo manipulation, and viewport camera control."""
    
    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def _get_active_camera(self) -> Tuple[Optional[TransformComponent], Optional[CameraComponent]]:
        """Resolves the currently designated scene camera for rendering/raycasting."""
        for ent in self.scene.entities:
            cam = ent.get_component(CameraComponent)
            if cam and getattr(cam, 'is_active', False): 
                return ent.get_component(TransformComponent), cam
                
        for ent in self.scene.entities:
            cam = ent.get_component(CameraComponent)
            if cam: 
                return ent.get_component(TransformComponent), cam
                
        return None, None

    def get_ray(self, mx: float, my: float, width: int, height: int, cam_tf: TransformComponent, cam: CameraComponent, custom_view: Optional[glm.mat4] = None, custom_proj: Optional[glm.mat4] = None) -> Tuple[Optional[glm.vec3], Optional[glm.vec3]]:
        """Unprojects screen coordinates into a world-space Ray (Origin and Direction)."""
        x = (2.0 * mx) / max(width, 1.0) - 1.0
        y = 1.0 - (2.0 * my) / max(height, 1.0)
        
        if custom_view is not None and custom_proj is not None:
            inv_proj = glm.inverse(custom_proj)
            inv_view = glm.inverse(custom_view)
            ray_eye = inv_proj * glm.vec4(x, y, -1.0, 1.0)
            ray_dir = glm.normalize(glm.vec3(inv_view * glm.vec4(ray_eye.x, ray_eye.y, -1.0, 0.0)))
            return glm.vec3(inv_view[3]), ray_dir

        if not cam or not cam_tf: 
            return None, None
        
        inv_proj = glm.inverse(cam.get_projection_matrix())
        inv_view = glm.inverse(cam.get_view_matrix())

        if cam.mode == "Perspective":
            ray_eye = inv_proj * glm.vec4(x, y, -1.0, 1.0)
            ray_dir = glm.normalize(glm.vec3(inv_view * glm.vec4(ray_eye.x, ray_eye.y, -1.0, 0.0)))
            ray_origin = cam_tf.global_position
        else:
            near_pt = inv_proj * glm.vec4(x, y, -1.0, 1.0)
            ray_origin = glm.vec3(inv_view * glm.vec4(near_pt.x, near_pt.y, near_pt.z, 1.0))
            ray_dir = -glm.vec3(glm.mat3_cast(cam_tf.global_quat_rot)[2])
            
        return ray_origin, ray_dir

    def project_to_screen(self, pos_3d: glm.vec3, width: int, height: int, cam: CameraComponent) -> Optional[glm.vec2]:
        """Transforms a 3D coordinate into 2D viewport space."""
        if not cam: return None
        clip = cam.get_projection_matrix() * cam.get_view_matrix() * glm.vec4(pos_3d, 1.0)
        if clip.w == 0 or (cam.mode == "Perspective" and clip.w <= 0): 
            return None
        ndc = glm.vec3(clip) / clip.w
        return glm.vec2((ndc.x + 1.0)/2.0 * width, (1.0 - ndc.y)/2.0 * height)

    def get_axis_screen_dir(self, axis_char: str, width: int, height: int, cam: CameraComponent) -> glm.vec2:
        """Calculates the 2D screen-space trajectory of a 3D gizmo axis."""
        if self.scene.selected_index < 0: return glm.vec2(0, 0)
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if not tf: return glm.vec2(0, 0)
        
        local_axis = glm.vec3(1,0,0) if axis_char == 'X' else (glm.vec3(0,1,0) if axis_char == 'Y' else glm.vec3(0,0,1))
        axis_3d = tf.global_position + glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(local_axis, 0.0))
        
        p0 = self.project_to_screen(tf.global_position, width, height, cam)
        p1 = self.project_to_screen(axis_3d, width, height, cam)
        if p0 is None or p1 is None: return glm.vec2(0, 0)
        
        return glm.normalize(p1 - p0) if glm.length(p1 - p0) > 0 else glm.vec2(0, 0)

    def check_gizmo_hover(self, mx: float, my: float, width: int, height: int, custom_tf: Optional[TransformComponent] = None, custom_view: Optional[glm.mat4] = None, custom_proj: Optional[glm.mat4] = None, is_hud: bool = False) -> Optional[str]:
        """Evaluates intersection between screen cursor and translation/rotation/scale handles."""
        cam_tf, cam = self._get_active_camera()
        
        if is_hud:
            if not custom_tf: return 'ALL'
            tf = custom_tf
            mode = "ROTATE"
            ray_origin, ray_dir = self.get_ray(mx, my, width, height, cam_tf, cam, custom_view, custom_proj)
            inv_gizmo = glm.inverse(glm.mat4_cast(tf.quat_rot))
            ring_radius, ring_thick = 0.8, 0.08
        else:
            if self.scene.selected_index < 0: return None
            sel_ent = self.scene.entities[self.scene.selected_index]
            tf = sel_ent.get_component(TransformComponent)
            renderer = sel_ent.get_component(MeshRenderer)
            if not tf or not cam or not cam_tf: return None
            
            # Defense protocol: Avoid selecting the manipulator widget attached to the active view camera
            if sel_ent.get_component(CameraComponent) == cam:
                return None

            cam_g_pos = cam_tf.global_position
            g_pos = tf.global_position
            g_rot = tf.global_quat_rot
            mode = getattr(self.scene, 'manipulation_mode', 'ROTATE')
            
            if renderer and getattr(renderer, 'is_proxy', False):
                if mode == "SCALE": mode = "NONE"
            if mode == "NONE": return None
            
            ray_origin, ray_dir = self.get_ray(mx, my, width, height, cam_tf, cam)
            if not ray_origin: return None

            # Maintain constant pixel size for gizmo relative to camera depth
            pixel_factor = 100.0 / max(height, 1.0)
            if cam.mode == "Perspective":
                dist = glm.length(cam_g_pos - g_pos)
                g_scale = dist * math.tan(math.radians(cam.fov / 2.0)) * pixel_factor
            else:
                g_scale = cam.ortho_size * pixel_factor

            base_model = glm.translate(glm.mat4(1.0), g_pos) * glm.mat4_cast(g_rot) * glm.scale(glm.mat4(1.0), glm.vec3(g_scale))
            inv_gizmo = glm.inverse(base_model)
            
            ring_radius, ring_thick = 1.5, 0.08
            thickness, length = 0.06, 1.4  

        lro = glm.vec3(inv_gizmo * glm.vec4(ray_origin, 1.0))
        lrd = glm.normalize(glm.vec3(inv_gizmo * glm.vec4(ray_dir, 0.0)))
        hit_axis, min_t = None, float('inf')

        if mode in ["SCALE", "MOVE"] and not is_hud:
            if mode == "SCALE":
                hit_c, tc, _ = MathUtils.ray_intersect_aabb(lro, lrd, glm.vec3(-thickness*3.0), glm.vec3(thickness*3.0))
                if hit_c: min_t, hit_axis = tc, 'ALL'
                
            if not hit_axis:
                hit_x, tx, _ = MathUtils.ray_intersect_aabb(lro, lrd, glm.vec3(thickness, -thickness, -thickness), glm.vec3(length, thickness, thickness))
                hit_y, ty, _ = MathUtils.ray_intersect_aabb(lro, lrd, glm.vec3(-thickness, thickness, -thickness), glm.vec3(thickness, length, thickness))
                hit_z, tz, _ = MathUtils.ray_intersect_aabb(lro, lrd, glm.vec3(-thickness, -thickness, thickness), glm.vec3(thickness, thickness, length))
                if hit_x and tx < min_t: min_t, hit_axis = tx, 'X'
                if hit_y and ty < min_t: min_t, hit_axis = ty, 'Y'
                if hit_z and tz < min_t: min_t, hit_axis = tz, 'Z'
                
        elif mode == "ROTATE":
            hit_x, tx = MathUtils.ray_intersect_ring(lro, lrd, glm.vec3(1,0,0), ring_radius, ring_thick)
            hit_y, ty = MathUtils.ray_intersect_ring(lro, lrd, glm.vec3(0,1,0), ring_radius, ring_thick)
            hit_z, tz = MathUtils.ray_intersect_ring(lro, lrd, glm.vec3(0,0,1), ring_radius, ring_thick)
            if hit_x and tx < min_t: min_t, hit_axis = tx, 'X'
            if hit_y and ty < min_t: min_t, hit_axis = ty, 'Y'
            if hit_z and tz < min_t: min_t, hit_axis = tz, 'Z'
            
        if is_hud and hit_axis is None: return 'ALL'
        return hit_axis

    def check_screen_axis_hover(self, mx: float, my: float, width: int, height: int) -> Optional[str]:
        """Checks for interactions with the corner orientation indicator (HUD Axis)."""
        if not getattr(self.scene, 'show_screen_axis', True): return None
        cam_tf, cam = self._get_active_camera()
        if not cam: return None
        
        view_matrix = cam.get_view_matrix()
        axis_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -3.0)) * glm.mat4(glm.mat3(view_matrix))
        axis_proj = glm.perspective(glm.radians(45.0), 1.0, 0.1, 10.0)
        
        clip_origin = axis_proj * (axis_view * glm.vec4(0.0, 0.0, 0.0, 1.0))
        if clip_origin.w <= 0: return None
        ndc_origin = clip_origin / clip_origin.w
        
        pad = HUD_AXIS_PADDING
        scl = HUD_AXIS_SCALE
        p0 = glm.vec2((width - pad) + ndc_origin.x * scl, pad - ndc_origin.y * scl)
        
        pts_3d = [('X', glm.vec3(1.3, 0, 0)), ('Y', glm.vec3(0, 1.3, 0)), ('Z', glm.vec3(0, 0, 1.3)),
                  ('-X', glm.vec3(-1.1, 0, 0)), ('-Y', glm.vec3(0, -1.1, 0)), ('-Z', glm.vec3(0, 0, -1.1))]
        
        closest_axis, min_dist, closest_z = None, 18.0, -float('inf') 
        
        for axis_name, pos in pts_3d:
            clip = axis_proj * (axis_view * glm.vec4(pos, 1.0))
            if clip.w > 0:
                ndc = clip / clip.w
                p1 = glm.vec2((width - pad) + ndc.x * scl, pad - ndc.y * scl)
                line_vec = p1 - p0
                l2 = glm.dot(line_vec, line_vec)
                if l2 == 0: continue
                
                t = max(0.0, min(1.0, glm.dot(glm.vec2(mx, my) - p0, line_vec) / l2))
                proj_p = p0 + t * line_vec
                dist = glm.length(glm.vec2(mx, my) - proj_p)
                
                if dist < min_dist and ndc.z > closest_z:
                    min_dist = dist
                    closest_z = ndc.z
                    closest_axis = axis_name
        return closest_axis

    def check_hud_gizmo_hover(self, mx: float, my: float, width: int, height: int) -> str:
        """Proxies hover state checks for the isolated HUD representation of the active transform."""
        if self.scene.selected_index < 0: return 'ALL'
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        cam_tf, cam = self._get_active_camera()
        hud_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2.5)) * (glm.mat4(glm.mat3(cam.get_view_matrix())) if cam else glm.mat4(1.0))
        proj = glm.perspective(glm.radians(45.0), width / max(height, 1), 0.1, 10.0)
        return self.check_gizmo_hover(mx, my, width, height, custom_tf=tf, custom_view=hud_view, custom_proj=proj, is_hud=True)

    def handle_hud_gizmo_drag(self, dx: float, dy: float, active_axis: str, width: int, height: int) -> None:
        """Applies input deltas to rotate via the corner Arcball UI."""
        if self.scene.selected_index < 0: return
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        cam_tf, cam = self._get_active_camera()
        hud_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2.5)) * (glm.mat4(glm.mat3(cam.get_view_matrix())) if cam else glm.mat4(1.0))
        
        if active_axis in ['X', 'Y', 'Z']:
            tf.rotate_local(active_axis, InteractionMath.calc_gizmo_rotate_angle(-dx, dy, active_axis, tf.quat_rot, hud_view, 1.5))
        else:
            tf.quat_rot, tf.rotation = InteractionMath.calc_arcball_rotation(dx, dy, tf.quat_rot, hud_view, speed_x=0.5, speed_y=0.5)
        
        tf.sync_from_gui()

    def handle_gizmo_drag(self, dx: float, dy: float, active_axis: str, width: int, height: int) -> None:
        """Maps 2D mouse deltas into 3D operations (Translate, Rotate, Scale) on the active entity."""
        if self.scene.selected_index < 0: return
        sel_ent = self.scene.entities[self.scene.selected_index]
        tf = sel_ent.get_component(TransformComponent)
        cam_tf, cam = self._get_active_camera()
        if not tf or not cam_tf: return

        # Defense protocol against manipulating the viewpoint actively rendering the frame
        if sel_ent.get_component(CameraComponent) == cam:
            return

        mode = getattr(self.scene, 'manipulation_mode', 'ROTATE')
        cam_g_pos, g_pos, g_rot = cam_tf.global_position, tf.global_position, tf.global_quat_rot
        
        if active_axis:
            if active_axis == 'ALL' and mode == "SCALE": 
                tf.scale = glm.max(glm.vec3(0.01), tf.scale + glm.vec3((dx - -dy) * 0.005))
            else:
                if mode == "ROTATE":
                    tf.rotate_local(active_axis, InteractionMath.calc_gizmo_rotate_angle(dx, dy, active_axis, g_rot, cam.get_view_matrix(), speed_multiplier=-0.8)) 
                elif mode == "MOVE":
                    dir_2d = self.get_axis_screen_dir(active_axis, width, height, cam)
                    dist = glm.length(cam_g_pos - g_pos) if cam.mode == "Perspective" else cam.ortho_size
                    tf.position += tf.world_to_local_vec(InteractionMath.calc_gizmo_move_offset(dx, -dy, active_axis, g_rot, dir_2d, dist, speed=0.0008))
                elif mode == "SCALE":
                    dir_2d = self.get_axis_screen_dir(active_axis, width, height, cam)
                    amt = InteractionMath.calc_gizmo_scale_amount(dx, -dy, dir_2d, speed=0.005)
                    if active_axis == 'X': tf.scale.x = max(0.01, tf.scale.x + amt)
                    if active_axis == 'Y': tf.scale.y = max(0.01, tf.scale.y + amt)
                    if active_axis == 'Z': tf.scale.z = max(0.01, tf.scale.z + amt)
        else:
            # Freeform manipulations based on view plane
            if mode == "ROTATE": 
                new_global_quat, _ = InteractionMath.calc_arcball_rotation(dx, dy, g_rot, cam.get_view_matrix(), speed_x=0.5, speed_y=-0.5)
                tf.quat_rot = tf.world_to_local_quat(new_global_quat)
                tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            elif mode == "MOVE": 
                dist = glm.length(cam_g_pos - g_pos) if cam.mode == "Perspective" else cam.ortho_size
                tf.position += tf.world_to_local_vec(InteractionMath.calc_free_move_offset(dx, dy, cam.get_view_matrix(), dist, speed=0.0008))
            elif mode == "SCALE": 
                tf.scale = glm.max(glm.vec3(0.01), tf.scale + glm.vec3((dx + dy) * 0.005))
                
        tf.sync_from_gui()

    def update_camera_movement(self, active_commands: List[str], dt: float) -> bool:
        """Processes continuous WASD-style navigation commands."""
        cam_tf, cam = self._get_active_camera()
        if not cam or not cam_tf: return False
        
        speed = CAMERA_MOVE_SPEED * dt
        rot_speed = CAMERA_ROTATION_SPEED * dt
        rot_mat = glm.mat3_cast(cam_tf.quat_rot)
        up, right = rot_mat[1], rot_mat[0]
        moved = False
        
        if "CAM_FORWARD" in active_commands: cam_tf.position += up * speed; moved = True
        if "CAM_BACKWARD" in active_commands: cam_tf.position -= up * speed; moved = True
        if "CAM_LEFT" in active_commands: cam_tf.position -= right * speed; moved = True
        if "CAM_RIGHT" in active_commands: cam_tf.position += right * speed; moved = True
        if "CAM_ROLL_LEFT" in active_commands: cam_tf.rotate_local('Z', rot_speed); moved = True
        if "CAM_ROLL_RIGHT" in active_commands: cam_tf.rotate_local('Z', -rot_speed); moved = True
        
        return moved

    def orbit_camera(self, dx: float, dy: float) -> None:
        """Pivots the camera around its local center point."""
        cam_tf, _ = self._get_active_camera()
        if cam_tf: 
            yaw_quat = glm.angleAxis(glm.radians(dx * 0.1), glm.vec3(0, 1, 0))
            pitch_quat = glm.angleAxis(glm.radians(dy * 0.1), glm.vec3(1, 0, 0))
            cam_tf.quat_rot = yaw_quat * cam_tf.quat_rot * pitch_quat
            cam_tf.rotation = glm.degrees(glm.eulerAngles(cam_tf.quat_rot))

    def pan_camera(self, dx: float, dy: float) -> None:
        """Translates the camera parallel to the viewing plane."""
        cam_tf, cam = self._get_active_camera()
        if cam_tf and cam:
            factor = 0.005 if cam.mode == "Perspective" else cam.ortho_size * 0.001
            rot_mat = glm.mat3_cast(cam_tf.quat_rot)
            cam_tf.position += rot_mat[1] * (dy * factor) - rot_mat[0] * (dx * factor)

    def zoom_camera(self, yoffset: float) -> None:
        """Adjusts depth translation for Perspective cameras, or viewport scale for Orthographic."""
        cam_tf, cam = self._get_active_camera()
        if not cam or not cam_tf: return
        if cam.mode == "Perspective": 
            cam_tf.position += -glm.vec3(glm.mat3_cast(cam_tf.quat_rot)[2]) * (yoffset * 1.5)
        else: 
            cam.ortho_size = max(0.1, cam.ortho_size - yoffset * 0.5)

    def snap_camera_to_axis(self, axis: str) -> None:
        """Aligns the viewpoint to perfectly face cardinal direction planes (Top, Front, Right, etc.)."""
        cam_tf, cam = self._get_active_camera()
        if not cam_tf: return
        
        rot = glm.vec3(0, 0, 0)
        if axis == 'X': rot = glm.vec3(0, 90, 0)      
        elif axis == '-X': rot = glm.vec3(0, -90, 0)  
        elif axis == 'Y': rot = glm.vec3(-90, 0, 0)   
        elif axis == '-Y': rot = glm.vec3(90, 0, 0)   
        elif axis == 'Z': rot = glm.vec3(0, 0, 0)     
        elif axis == '-Z': rot = glm.vec3(0, 180, 0)  
        
        cam_tf.rotation = rot
        cam_tf.quat_rot = glm.quat(glm.radians(rot))
        
        dist = glm.length(cam_tf.position)
        if dist < 0.1: dist = 5.0 
        
        forward = glm.vec3(glm.mat4_cast(cam_tf.quat_rot) * glm.vec4(0, 0, -1, 0))
        cam_tf.position = -forward * dist

    def get_screen_axis_labels_data(self, width: int, height: int) -> List[Dict[str, Any]]:
        """Calculates screen-space positioning for the X/Y/Z orientation labels in the viewport UI."""
        cam_tf, cam = self._get_active_camera()
        if not cam or not cam_tf: return []
        
        axis_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -3.0)) * glm.mat4(glm.mat3(cam.get_view_matrix()))
        axis_proj = glm.perspective(glm.radians(45.0), 1.0, 0.1, 10.0)
        pts_3d = [('X', glm.vec3(1.3, 0, 0)), ('Y', glm.vec3(0, 1.3, 0)), ('Z', glm.vec3(0, 0, 1.3)), 
                  ('-X', glm.vec3(-1.1, 0, 0)), ('-Y', glm.vec3(0, -1.1, 0)), ('-Z', glm.vec3(0, 0, -1.1))]
        
        pad = HUD_AXIS_PADDING
        scl = HUD_AXIS_SCALE
        labels_data = []
        
        for name, pos in pts_3d:
            clip = axis_proj * (axis_view * glm.vec4(pos, 1.0))
            if clip.w > 0:
                ndc = clip / clip.w
                labels_data.append({
                    'name': name, 
                    'z': ndc.z, 
                    'x': int((width - pad) + ndc.x * scl - 5), 
                    'y': int(pad - ndc.y * scl - (10 if name != "" else 4))
                })
                
        # Sort back-to-front for proper label depth rendering
        labels_data.sort(key=lambda item: item['z'], reverse=True)
        return labels_data