import glm
import ctypes
import numpy as np
from OpenGL.GL import *
from typing import Any, Optional, Tuple, List, Union
from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.components import CameraComponent
from src.app.exceptions import RenderError
from src.app.config import MAX_LIGHTS
from src.engine.graphics.render_queue import RenderQueue

class Renderer:
    """Handles all core OpenGL rendering passes including shadows, scene geometry, and off-screen data capture."""
    
    def __init__(self) -> None:
        self.mat_standard_shader = ResourceManager.get_shader("mat_standard")
        self.mat_unlit_shader = ResourceManager.get_shader("mat_unlit")
        self.pass_depth_shader = ResourceManager.get_shader("pass_depth")
        self.pass_picking_shader = ResourceManager.get_shader("pass_picking")
        self.pass_shadow_shader = ResourceManager.get_shader("pass_shadow")
        self.editor_solid_shader = ResourceManager.get_shader("editor_solid")
        self.editor_proxy_shader = ResourceManager.get_shader("editor_proxy")
        
        self.queue = RenderQueue()
        
        self.wireframe: bool = False
        self.render_mode: int = 4 
        self.output_type: int = 0  
        self.comb_light: bool = True
        self.comb_tex: bool = True
        self.comb_vcolor: bool = True
        
        self.picking_fbo: Optional[int] = None
        self.picking_texture: Optional[int] = None
        self.picking_depth: Optional[int] = None
        
        self.msaa_fbo: Optional[int] = None
        self.msaa_color: Optional[int] = None
        self.msaa_depth: Optional[int] = None
        
        self.picking_width: int = 0
        self.picking_height: int = 0
        
        self.shadow_width: int = 4096
        self.shadow_height: int = 4096
        self.shadow_ortho_size: float = 50.0
        self.shadow_fbo: int = 0
        self.shadow_texture: int = 0
        
        while glGetError() != GL_NO_ERROR: pass
        self._setup_shadow_fbo()

    def toggle_wireframe(self) -> None:
        self.wireframe = not self.wireframe

    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        self.wireframe = wireframe
        self.render_mode = mode
        self.output_type = output
        self.comb_light = light
        self.comb_tex = tex
        self.comb_vcolor = vcolor

    def _setup_shadow_fbo(self) -> None:
        """Initializes the Framebuffer Object used for directional light shadow mapping."""
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        
        self.shadow_fbo = glGenFramebuffers(1)
        self.shadow_texture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
            self.shadow_width, self.shadow_height, 0, 
            GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0)
        )
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.shadow_texture, 0)
        
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        while glGetError() != GL_NO_ERROR: pass

    def _get_light_space_matrix(self, scene: Any) -> glm.mat4:
        """Computes the orthogonal projection-view matrix for the primary directional light."""
        for tf, light, ent in scene.cached_lights:
            if light.on and light.type == "Directional":
                light_dir = glm.normalize(glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0)))
                light_pos = -light_dir * 50.0 
                
                # Prevent Gimbal Lock when light is pointing directly downwards/upwards
                up_vec = glm.vec3(0.0, 1.0, 0.0)
                if abs(light_dir.y) > 0.999:
                    up_vec = glm.vec3(0.0, 0.0, 1.0)
                    
                view_matrix = glm.lookAt(light_pos, glm.vec3(0.0), up_vec)
                proj_matrix = glm.ortho(-self.shadow_ortho_size, self.shadow_ortho_size, 
                                        -self.shadow_ortho_size, self.shadow_ortho_size, 
                                        -100.0, 200.0)
                return proj_matrix * view_matrix
        return glm.mat4(1.0)

    def _render_shadow_pass(self, light_space_matrix: glm.mat4) -> None:
        """Executes the depth-only pre-pass to generate the shadow map."""
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK) 
        
        self.pass_shadow_shader.use()
        self.pass_shadow_shader.set_mat4("lightSpaceMatrix", light_space_matrix)
        
        self._draw_geometry_list(self.queue.opaque, self.pass_shadow_shader, True, False, True, is_shadow_pass=True)
        
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])

    def _get_active_camera_data(self, scene: Any) -> Tuple[Optional[CameraComponent], glm.mat4, glm.mat4, glm.vec3, int]:
        for tf, cam, ent in scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                return cam, cam.get_view_matrix(), cam.get_projection_matrix(), tf.global_position, scene.entities.index(ent)
        return None, glm.mat4(1.0), glm.mat4(1.0), glm.vec3(0), -1

    def _apply_lighting(self, scene: Any, shader: Any) -> None:
        """Injects lighting properties from the scene into the currently bound shader."""
        num_dir = num_point = num_spot = 0
        max_dir = MAX_LIGHTS.get("Directional", 8)
        max_point = MAX_LIGHTS.get("Point", 16)
        max_spot = MAX_LIGHTS.get("Spot", 8)
        
        for tf, light, ent in scene.cached_lights:
            if not light.on: continue
            
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
        """Iterates through a RenderQueue list and issues draw calls while respecting material states."""
        for tf, mesh, ent in item_list:
            model_mat = tf.get_matrix()
            shader.set_mat4("model", model_mat)
            
            if not is_depth_pass and not is_unlit_pass:
                m3 = glm.mat3(model_mat)
                det = glm.determinant(m3)
                shader.set_mat3("normalMatrix", glm.transpose(glm.inverse(m3)) if abs(det) > 1e-6 else m3)
            
            mat = mesh.material
            if mat:
                if not is_depth_pass:
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

    def _render_passes(self, scene: Any, active_camera: CameraComponent, cam_pos: glm.vec3, view_matrix: glm.mat4, projection_matrix: glm.mat4, light_space_matrix: glm.mat4, is_depth_pass: bool, is_unlit_pass: bool, use_light: bool, use_tex: bool, use_vcolor: bool) -> None:
        """Executes the main opaque and transparent rendering passes."""
        if is_depth_pass:
            active_shader = self.pass_depth_shader
        elif is_unlit_pass:
            active_shader = self.mat_unlit_shader
        else:
            active_shader = self.mat_standard_shader

        active_shader.use()
        active_shader.set_mat4("view", view_matrix)
        active_shader.set_mat4("projection", projection_matrix)
        
        if is_depth_pass:
            is_ortho = 1 if active_camera.mode == "Orthographic" else 0
            active_shader.set_int("isOrthographic", is_ortho)
            active_shader.set_float("near", active_camera.near)
            active_shader.set_float("far", active_camera.far)
        else:
            active_shader.set_vec3("viewPos", cam_pos)
            active_shader.set_int("combTex", int(use_tex))
            active_shader.set_int("combVColor", int(use_vcolor))
            
            if not is_unlit_pass:
                active_shader.set_int("combLight", int(use_light))
                
                if use_light:
                    glActiveTexture(GL_TEXTURE1) 
                    glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
                    active_shader.set_int("shadowMap", 1)
                    active_shader.set_mat4("lightSpaceMatrix", light_space_matrix)
                    
                self._apply_lighting(scene, active_shader)

        self._draw_geometry_list(self.queue.opaque, active_shader, is_depth_pass, is_unlit_pass, True, is_shadow_pass=False)
        
        force_transparent_depth = True if is_depth_pass else False
        self._draw_geometry_list(self.queue.transparent, active_shader, is_depth_pass, is_unlit_pass, force_transparent_depth, is_shadow_pass=False)
            
        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)

    def _render_proxies(self, view_matrix: glm.mat4, projection_matrix: glm.mat4, active_camera: CameraComponent) -> None:
        """Renders editor-only proxies (e.g., Light Icons, Camera Frustums)."""
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for tf, mesh, ent in self.queue.proxies:
            if active_camera and ent.get_component(type(active_camera)) == active_camera:
                continue 
                
            geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
            has_vcolor = getattr(geom_obj, 'has_vertex_color', False)
            
            if has_vcolor:
                self.editor_proxy_shader.use()
                self.editor_proxy_shader.set_mat4("view", view_matrix)
                self.editor_proxy_shader.set_mat4("projection", projection_matrix)
                self.editor_proxy_shader.set_mat4("model", tf.get_matrix())
            else:
                self.editor_solid_shader.use()
                self.editor_solid_shader.set_mat4("view", view_matrix)
                self.editor_solid_shader.set_mat4("projection", projection_matrix)
                self.editor_solid_shader.set_mat4("model", tf.get_matrix())
                
                base_c = mesh.material.base_color if hasattr(mesh, 'material') else glm.vec3(1.0)
                self.editor_solid_shader.set_vec3("solidColor", base_c)
                
            if hasattr(geom_obj, 'draw'):
                geom_obj.draw()

    def render_scene(self, scene: Any, window_w: int, window_h: int) -> None:
        """Primary entry point for rendering the scene to the active viewport."""
        if not scene: return
            
        cam, view_mat, proj_mat, cam_pos, _ = self._get_active_camera_data(scene)
        if not cam: return

        self.queue.build(scene, cam_pos)
        
        light_space_mat = self._get_light_space_matrix(scene)
        if self.comb_light:
            self._render_shadow_pass(light_space_mat)

        glViewport(0, 0, window_w, window_h)
        is_depth_pass = (self.output_type == 1)
        is_unlit_pass = (self.render_mode != 4)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        self._render_passes(scene, cam, cam_pos, view_mat, proj_mat, light_space_mat, is_depth_pass, is_unlit_pass, self.comb_light, self.comb_tex, self.comb_vcolor)
        self._render_proxies(view_mat, proj_mat, cam)

    def capture_fbo_frame(self, scene: Any, width: int, height: int, mode: str = "RGB", return_texture_id: bool = False) -> Union[bytes, int]:
        """
        Renders the scene off-screen and extracts the raw pixel data.
        Critical: Restores MSAA and FBO states post-capture to prevent global state leaks.
        """
        if width <= 0 or height <= 0:
            return 0 if return_texture_id else b""

        if self.picking_width != width or self.picking_height != height:
            self._setup_picking_fbo(width, height)

        cam, view_mat, proj_mat, cam_pos, _ = self._get_active_camera_data(scene)
        if not cam:
            return 0 if return_texture_id else b""

        self.queue.build(scene, cam_pos)
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)

        original_clear_color = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        if mode in ["RGB", "DEPTH"]:
            light_space_mat = self._get_light_space_matrix(scene)
            if self.comb_light and mode == "RGB":
                self._render_shadow_pass(light_space_mat)
                
            target_fbo = self.msaa_fbo if mode == "RGB" else self.picking_fbo
            glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
            glViewport(0, 0, width, height)
            
            if mode == "DEPTH":
                glClearColor(1.0, 1.0, 1.0, 1.0)
            else:
                glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            
            if mode == "RGB":
                glEnable(GL_MULTISAMPLE)
            else:
                # Disable MSAA to ensure pixel-perfect depth extraction
                glDisable(GL_MULTISAMPLE)

            is_depth_pass = (mode == "DEPTH")
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            self._render_passes(scene, cam, cam_pos, view_mat, proj_mat, light_space_mat, is_depth_pass, False, True, True, True)

            if mode == "RGB":
                glBindFramebuffer(GL_READ_FRAMEBUFFER, self.msaa_fbo)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.picking_fbo)
                glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
                
                if return_texture_id:
                    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
                    glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
                    return self.picking_texture

                glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)
                glReadBuffer(GL_COLOR_ATTACHMENT0)
                pixel_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            else:
                if return_texture_id:
                    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
                    glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
                    glEnable(GL_MULTISAMPLE) 
                    return self.picking_depth

                depth_raw = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
                depth_buffer = np.frombuffer(depth_raw, dtype=np.float32)
                depth_metric = self._linearize_depth_buffer(depth_buffer, float(cam.near), float(cam.far), str(cam.mode))
                pixel_data = depth_metric.tobytes()
                
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
            
            # Guarantee MSAA restoration to prevent state leaks impacting the primary UI view
            glEnable(GL_MULTISAMPLE) 
            return pixel_data
            
        elif mode in ["MASK", "MASK_SEMANTIC", "MASK_INSTANCE", "SEMANTIC", "INSTANCE"]:
            glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)
            glViewport(0, 0, width, height)
            
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_MULTISAMPLE)
            
            self.pass_picking_shader.use()
            self.pass_picking_shader.set_mat4("view", view_mat)
            self.pass_picking_shader.set_mat4("projection", proj_mat)

            mask_mode = "INSTANCE" if "INSTANCE" in mode else "SEMANTIC"
            self._draw_picking_masks(scene, mask_mode)
            
            glEnable(GL_BLEND)
            glEnable(GL_MULTISAMPLE)

            if return_texture_id:
                glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
                glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
                return self.picking_texture

            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixel_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
            return pixel_data

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glClearColor(original_clear_color[0], original_clear_color[1], original_clear_color[2], original_clear_color[3])
        return 0 if return_texture_id else b""

    def _linearize_depth_buffer(self, depth_buffer: np.ndarray, near: float, far: float, cam_mode: str) -> np.ndarray:
        depth = depth_buffer.astype(np.float32, copy=False)
        if cam_mode == "Orthographic":
            linear = near + depth * (far - near)
        else:
            ndc = depth * 2.0 - 1.0
            denom = far + near - ndc * (far - near)
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            linear = (2.0 * near * far) / denom
        linear = linear.astype(np.float32)
        linear[depth >= 1.0 - 1e-6] = np.inf
        return linear

    def _setup_picking_fbo(self, width: int, height: int) -> None:
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        
        if self.picking_fbo is not None:
            glDeleteFramebuffers(1, [self.picking_fbo])
            glDeleteTextures(2, [self.picking_texture, self.picking_depth])
            
        if getattr(self, 'msaa_fbo', None) is not None:
            glDeleteFramebuffers(1, [self.msaa_fbo])
            glDeleteRenderbuffers(2, [self.msaa_color, self.msaa_depth])

        self.picking_width = width
        self.picking_height = height

        self.msaa_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)
        
        self.msaa_color = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_color)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.msaa_color)
        
        self.msaa_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_depth)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.msaa_depth)

        self.picking_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)

        self.picking_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.picking_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.picking_texture, 0)

        self.picking_depth = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.picking_depth)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.picking_depth, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RenderError(f"Picking Framebuffer Object (FBO) generation failed. OpenGL Status Code: {status}")

    def _draw_picking_masks(self, scene: Any, mask_mode: str) -> None:
        try:
            from src.engine.scene.components.semantic_cmp import SemanticComponent
        except ImportError:
            return

        classes = {}
        if mask_mode == "SEMANTIC":
            try:
                from src.app import ctx
                classes = ctx.engine.get_semantic_classes()
            except Exception:
                pass

        for item_list in [self.queue.opaque, self.queue.transparent]:
            for tf, mesh, ent in item_list:
                semantic = ent.get_component(SemanticComponent)
                if not semantic or not mesh.visible: 
                    continue

                if mask_mode == "INSTANCE":
                    track_id = int(getattr(semantic, "track_id", -1))
                    if track_id <= 0:
                        track_id = scene.entities.index(ent) + 1
                    r = (track_id & 0x000000FF) / 255.0
                    g = ((track_id & 0x0000FF00) >> 8) / 255.0
                    b = ((track_id & 0x00FF0000) >> 16) / 255.0
                    color = glm.vec3(r, g, b)
                else:
                    c_id = semantic.class_id
                    c_info = classes.get(c_id, {})
                    c_color = c_info.get("color", [1.0, 1.0, 1.0]) if isinstance(c_info, dict) else [1.0, 1.0, 1.0]
                    if max(c_color) > 1.0:
                        c_color = [c / 255.0 for c in c_color]
                    color = glm.vec3(c_color[0], c_color[1], c_color[2])

                self.pass_picking_shader.set_vec3("u_ColorId", color)
                self.pass_picking_shader.set_mat4("model", tf.get_matrix())

                geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
                if hasattr(geom_obj, 'draw'): 
                    geom_obj.draw()

    def raycast_select(self, scene: Any, mx: float, my: float, width: int, height: int) -> int:
        if width <= 0 or height <= 0 or mx < 0 or my < 0 or mx >= width or my >= height:
            return -1

        if self.picking_width != width or self.picking_height != height:
            self._setup_picking_fbo(width, height)

        cam, view_mat, proj_mat, cam_pos, cam_idx = self._get_active_camera_data(scene)
        if not cam: 
            return -1

        self.queue.build(scene, cam_pos)
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)

        glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)
        glViewport(0, 0, width, height)
        
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)

        self.pass_picking_shader.use()
        self.pass_picking_shader.set_mat4("view", view_mat)
        self.pass_picking_shader.set_mat4("projection", proj_mat)

        for item_list in [self.queue.opaque, self.queue.transparent, self.queue.proxies]:
            for tf, mesh, ent in item_list:
                ent_idx = scene.entities.index(ent)
                if ent_idx == cam_idx:
                    continue
                
                r = (ent_idx & 0x000000FF) / 255.0
                g = ((ent_idx & 0x0000FF00) >> 8) / 255.0
                b = ((ent_idx & 0x00FF0000) >> 16) / 255.0

                self.pass_picking_shader.set_vec3("u_ColorId", glm.vec3(r, g, b))
                self.pass_picking_shader.set_mat4("model", tf.get_matrix())

                geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
                if hasattr(geom_obj, 'draw'):
                    geom_obj.draw()

        x = int(mx)
        y = int(height - my) 
        pixel_data = glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)

        if pixel_data:
            r, g, b, _ = pixel_data[0], pixel_data[1], pixel_data[2], pixel_data[3]
            if r == 255 and g == 255 and b == 255:
                return -1
            hit_idx = r + (g << 8) + (b << 16)
            if 0 <= hit_idx < len(scene.entities):
                return hit_idx

        return -1