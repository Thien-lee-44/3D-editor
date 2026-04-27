import glm
import ctypes
import numpy as np
from OpenGL.GL import *
from typing import Any, List

from src.engine.resources.resource_manager import ResourceManager
from src.engine.graphics.render_queue import RenderQueue
from src.app.config import MAX_LIGHTS

class BaseRenderer:
    def __init__(self) -> None:
        self.mat_standard_shader = ResourceManager.get_shader("mat_standard")
        self.mat_unlit_shader = ResourceManager.get_shader("mat_unlit")
        self.pass_depth_shader = ResourceManager.get_shader("pass_depth")
        self.pass_picking_shader = ResourceManager.get_shader("pass_picking")
        
        try:
            self.pass_shadow_shader = ResourceManager.get_shader("pass_shadow")
        except Exception:
            self.pass_shadow_shader = self.mat_unlit_shader
            
        self.queue = RenderQueue()

        self.shadow_width: int = 4096     
        self.shadow_height: int = 4096      
        self.shadow_ortho_size: float = 75.0              
        self.shadow_min_ortho_extent: float = 25.0         
        self.shadow_max_ortho_extent: float = 130.0       
        self.shadow_fit_padding: float = 5.0
        self.shadow_poly_offset_factor: float = 0.2
        self.shadow_poly_offset_units: float = 0.5
        self.shadow_fbo: int = 0
        self.shadow_texture: int = 0

    def _setup_shadow_fbo(self) -> None:
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        self.shadow_fbo = glGenFramebuffers(1)
        self.shadow_texture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.shadow_texture, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)

    def _is_entity_globally_visible(self, entity: Any) -> bool:
        curr = entity
        while curr is not None:
            if hasattr(curr, 'get_component'):
                from src.engine.scene.components import MeshRenderer
                mesh = curr.get_component(MeshRenderer)
                if mesh and not getattr(mesh, "visible", True):
                    return False
            curr = getattr(curr, "parent", None)
        return True

    def _get_light_space_matrix(self, scene: Any) -> glm.mat4:
        light_dir = None
        for tf, light, ent in scene.cached_lights:
            if getattr(light, 'on', True) and light.type == "Directional":
                light_dir = glm.normalize(glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0)))
                break

        if light_dir is None:
            return glm.mat4(1.0)

        source_items = self.queue.opaque + self.queue.transparent
        if not source_items:
            source_items = list(getattr(scene, 'cached_renderables', []))

        min_corner = glm.vec3(1e9, 1e9, 1e9)
        max_corner = glm.vec3(-1e9, -1e9, -1e9)
        has_bounds = False
        for tf, mesh, ent in source_items:
            if getattr(mesh, 'is_proxy', False):
                continue
            if not self._is_entity_globally_visible(ent):
                continue

            p = tf.global_position
            s = tf.global_scale
            radius = max(0.5, 0.5 * max(abs(s.x), abs(s.y), abs(s.z)))
            r = glm.vec3(radius)
            mn = p - r
            mx = p + r

            min_corner.x = min(min_corner.x, mn.x)
            min_corner.y = min(min_corner.y, mn.y)
            min_corner.z = min(min_corner.z, mn.z)
            max_corner.x = max(max_corner.x, mx.x)
            max_corner.y = max(max_corner.y, mx.y)
            max_corner.z = max(max_corner.z, mx.z)
            has_bounds = True

        if has_bounds:
            focus_center = (min_corner + max_corner) * 0.5
            half_diag = glm.length(max_corner - min_corner) * 0.5
        else:
            focus_center = glm.vec3(0.0)
            half_diag = self.shadow_ortho_size * 0.5

        ortho_extent = half_diag + self.shadow_fit_padding
        ortho_extent = max(self.shadow_min_ortho_extent, ortho_extent)
        ortho_extent = min(self.shadow_max_ortho_extent, ortho_extent)

        light_pos = focus_center - light_dir * (ortho_extent * 2.0)
        up_vec = glm.vec3(0.0, 0.0, 1.0) if abs(light_dir.y) > 0.999 else glm.vec3(0.0, 1.0, 0.0)
        view_matrix = glm.lookAt(light_pos, focus_center, up_vec)
        depth_extent = ortho_extent * 3.0
        proj_matrix = glm.ortho(
            -ortho_extent,
            ortho_extent,
            -ortho_extent,
            ortho_extent,
            -depth_extent,
            depth_extent,
        )
        return proj_matrix * view_matrix

    def _apply_lighting(self, scene: Any, shader: Any) -> None:
        num_dir = num_point = num_spot = 0
        max_dir = MAX_LIGHTS.get("Directional", 8)
        max_point = MAX_LIGHTS.get("Point", 16)
        max_spot = MAX_LIGHTS.get("Spot", 8)
        
        for tf, light, ent in scene.cached_lights:
            if not getattr(light, 'on', True): continue
            
            if light.type == "Directional" and num_dir < max_dir:
                prefix = f"dirLights[{num_dir}]"
                direction = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                shader.set_vec3(f"{prefix}.direction", direction)
                shader.set_vec3(f"{prefix}.ambient", light.ambient)
                shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                shader.set_vec3(f"{prefix}.specular", light.specular)
                num_dir += 1
            elif light.type == "Point" and num_point < max_point:
                prefix = f"pointLights[{num_point}]"
                shader.set_vec3(f"{prefix}.position", tf.global_position)
                shader.set_vec3(f"{prefix}.ambient", light.ambient)
                shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                shader.set_vec3(f"{prefix}.specular", light.specular)
                shader.set_float(f"{prefix}.constant", light.constant)
                shader.set_float(f"{prefix}.linear", light.linear)
                shader.set_float(f"{prefix}.quadratic", light.quadratic)
                num_point += 1
            elif light.type == "Spot" and num_spot < max_spot:
                prefix = f"spotLights[{num_spot}]"
                direction = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                shader.set_vec3(f"{prefix}.position", tf.global_position)
                shader.set_vec3(f"{prefix}.direction", direction)
                shader.set_vec3(f"{prefix}.ambient", light.ambient)
                shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                shader.set_vec3(f"{prefix}.specular", light.specular)
                shader.set_float(f"{prefix}.constant", light.constant)
                shader.set_float(f"{prefix}.linear", light.linear)
                shader.set_float(f"{prefix}.quadratic", light.quadratic)
                shader.set_float(f"{prefix}.cutOff", light.cutOff)
                shader.set_float(f"{prefix}.outerCutOff", light.outerCutOff)
                num_spot += 1

        shader.set_int("numDirLights", num_dir)
        shader.set_int("numPointLights", num_point)
        shader.set_int("numSpotLights", num_spot)

    def _draw_geometry_list(self, item_list: List[Any], shader: Any, is_depth_pass: bool, is_unlit_pass: bool, force_depth_write: bool, is_shadow_pass: bool = False) -> None:
        for tf, mesh, ent in item_list:
            if is_shadow_pass and getattr(mesh, 'is_proxy', False):
                continue

            model_mat = tf.get_matrix()
            shader.set_mat4("model", model_mat)
            
            if not is_depth_pass and not is_unlit_pass:
                m3 = glm.mat3(model_mat)
                det = glm.determinant(m3)
                nm = glm.transpose(glm.inverse(m3)) if abs(det) > 1e-6 else m3
                
                l0 = glm.length(nm[0])
                l1 = glm.length(nm[1])
                l2 = glm.length(nm[2])
                if l0 > 1e-6: nm[0] /= l0
                if l1 > 1e-6: nm[1] /= l1
                if l2 > 1e-6: nm[2] /= l2
                
                shader.set_mat3("normalMatrix", nm)
            
            mat = mesh.material
            if mat:
                if not is_depth_pass or is_shadow_pass:
                    mat.apply(shader)
                
                r_state = mat.render_state
                
                if r_state.cull_face:
                    glEnable(GL_CULL_FACE)
                    glCullFace(int(r_state.cull_mode))
                else:
                    glDisable(GL_CULL_FACE)
                    
                if r_state.depth_test:
                    glEnable(GL_DEPTH_TEST)
                    glDepthFunc(int(r_state.depth_func))
                else:
                    glDisable(GL_DEPTH_TEST)
                    
                glDepthMask(GL_TRUE if force_depth_write and r_state.depth_write else GL_FALSE)
                
            geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
            if hasattr(geom_obj, 'draw'):
                geom_obj.draw()

    def _render_shadow_pass(self, light_space_matrix: glm.mat4, queue_items: List[Any]) -> None:
        if self.shadow_fbo == 0:
            self._setup_shadow_fbo()
            
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK) 
        
        # Keep polygon offset conservative; large values visibly detach the shadow from casters.
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(self.shadow_poly_offset_factor, self.shadow_poly_offset_units)
        
        self.pass_shadow_shader.use()
        self.pass_shadow_shader.set_mat4("lightSpaceMatrix", light_space_matrix)
        
        self._draw_geometry_list(queue_items, self.pass_shadow_shader, True, False, True, is_shadow_pass=True)
                
        # Restore state for the main color pass.
        glDisable(GL_POLYGON_OFFSET_FILL)
        
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])

    def _pass_depth(self, view: glm.mat4, proj: glm.mat4, near: float, far: float, queue_items: List[Any], is_ortho: bool = False) -> None:
        self.pass_depth_shader.use()
        self.pass_depth_shader.set_mat4("view", view)
        self.pass_depth_shader.set_mat4("projection", proj)
        self.pass_depth_shader.set_int("isOrthographic", 1 if is_ortho else 0)
        self.pass_depth_shader.set_float("near", near)
        self.pass_depth_shader.set_float("far", far)
        self._draw_geometry_list(queue_items, self.pass_depth_shader, True, False, True)

    def _linearize_depth_buffer(self, depth_buffer: np.ndarray, near: float, far: float, cam_mode: str) -> np.ndarray:
        depth = depth_buffer.astype(np.float32, copy=False)
        if cam_mode == "Orthographic":
            linear = near + depth * (far - near)
        else:
            ndc = depth * 2.0 - 1.0
            denom = far + near - ndc * (far - near)
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            linear = (2.0 * near * far) / denom
            
        linear = np.clip(linear, near, far)
        normalized = (linear - near) / (far - near)
        return (normalized * 255.0).astype(np.uint8)