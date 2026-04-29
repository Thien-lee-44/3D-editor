"""
Forward Rendering Pipeline.
Main execution pipeline for the Forward Rendering architecture.
Handles illumination passes and Off-screen GPU Color Picking via Framebuffer Objects (FBO).
"""

import glm
import ctypes
from OpenGL.GL import *
from typing import Any, Optional, List, Tuple

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.components import CameraComponent
from src.engine.scene.scene import Scene
from src.app.exceptions import RenderError
from src.app.config import MAX_LIGHTS
from src.engine.graphics.render_queue import RenderQueue


class Renderer:
    """
    Main execution pipeline for the Forward Rendering architecture.
    Handles PBR illumination passes and Off-screen GPU Color Picking via Framebuffer Objects (FBO).
    """
    
    def __init__(self) -> None:
        self.mat_standard_shader = ResourceManager.get_shader("mat_standard")
        self.mat_unlit_shader = ResourceManager.get_shader("mat_unlit")
        self.pass_depth_shader = ResourceManager.get_shader("pass_depth")
        self.pass_picking_shader = ResourceManager.get_shader("pass_picking")
        self.editor_solid_shader = ResourceManager.get_shader("editor_solid")
        self.editor_proxy_shader = ResourceManager.get_shader("editor_proxy")
        
        self.queue = RenderQueue()
        
        # Render State Flags
        self.wireframe: bool = False
        self.render_mode: int = 4 
        self.output_type: int = 0  
        self.comb_light: bool = True
        self.comb_tex: bool = True
        self.comb_vcolor: bool = True
        
        # GPU Color Picking FBO tracking variables
        self.picking_fbo: Optional[int] = None
        self.picking_texture: Optional[int] = None
        self.picking_depth: Optional[int] = None
        self.picking_width: int = 0
        self.picking_height: int = 0

    def toggle_wireframe(self) -> None:
        """Toggles the global wireframe rendering mode."""
        self.wireframe = not self.wireframe

    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        """Updates the global rendering pipeline configuration."""
        self.wireframe = wireframe
        self.render_mode = mode
        self.output_type = output
        self.comb_light = light
        self.comb_tex = tex
        self.comb_vcolor = vcolor

    def render_scene(self, scene: Any, window_w: int, window_h: int) -> None:
        """
        Executes the full rendering cycle: resolves the active camera, builds the 
        render queues, applies lighting uniforms, and dispatches OpenGL draw calls.
        """
        if not scene: 
            return
            
        glViewport(0, 0, window_w, window_h)
        
        active_camera: Optional[CameraComponent] = None
        cam_pos = glm.vec3(0)
        view_matrix = glm.mat4(1.0)
        projection_matrix = glm.mat4(1.0)
        
        # Resolve active camera constraints
        for tf, cam, ent in scene.cached_cameras:
            if cam.is_active:
                active_camera = cam
                cam_pos = tf.global_position
                view_matrix = cam.get_view_matrix()
                projection_matrix = cam.get_projection_matrix()
                break

        if not active_camera: 
            return

        # Sort renderables into opaque, transparent, and proxy queues
        self.queue.build(scene, cam_pos)

        is_depth_pass = (self.output_type == 1)
        is_unlit_pass = (self.render_mode != 4)
        
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
            active_shader.set_int("combTex", int(self.comb_tex))
            active_shader.set_int("combVColor", int(self.comb_vcolor))
            
            if not is_unlit_pass:
                active_shader.set_int("combLight", int(self.comb_light))
                
                num_dir = 0
                num_point = 0
                num_spot = 0
                max_dir = MAX_LIGHTS.get("Directional", 8)
                max_point = MAX_LIGHTS.get("Point", 16)
                max_spot = MAX_LIGHTS.get("Spot", 8)
                
                # Setup lighting uniforms
                for tf, light, ent in scene.cached_lights:
                    if not light.on: 
                        continue
                    
                    if light.type == "Directional" and num_dir < max_dir:
                        prefix = f"dirLights[{num_dir}]"
                        direction = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                        active_shader.set_vec3(f"{prefix}.direction", direction)
                        active_shader.set_vec3(f"{prefix}.ambient", light.ambient)
                        active_shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                        active_shader.set_vec3(f"{prefix}.specular", light.specular)
                        num_dir += 1
                        
                    elif light.type == "Point" and num_point < max_point:
                        prefix = f"pointLights[{num_point}]"
                        active_shader.set_vec3(f"{prefix}.position", tf.global_position)
                        active_shader.set_vec3(f"{prefix}.ambient", light.ambient)
                        active_shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                        active_shader.set_vec3(f"{prefix}.specular", light.specular)
                        active_shader.set_float(f"{prefix}.constant", light.constant)
                        active_shader.set_float(f"{prefix}.linear", light.linear)
                        active_shader.set_float(f"{prefix}.quadratic", light.quadratic)
                        num_point += 1
                        
                    elif light.type == "Spot" and num_spot < max_spot:
                        prefix = f"spotLights[{num_spot}]"
                        direction = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                        active_shader.set_vec3(f"{prefix}.position", tf.global_position)
                        active_shader.set_vec3(f"{prefix}.direction", direction)
                        active_shader.set_vec3(f"{prefix}.ambient", light.ambient)
                        active_shader.set_vec3(f"{prefix}.diffuse", light.diffuse)
                        active_shader.set_vec3(f"{prefix}.specular", light.specular)
                        active_shader.set_float(f"{prefix}.constant", light.constant)
                        active_shader.set_float(f"{prefix}.linear", light.linear)
                        active_shader.set_float(f"{prefix}.quadratic", light.quadratic)
                        active_shader.set_float(f"{prefix}.cutOff", light.cutOff)
                        active_shader.set_float(f"{prefix}.outerCutOff", light.outerCutOff)
                        num_spot += 1

                active_shader.set_int("numDirLights", num_dir)
                active_shader.set_int("numPointLights", num_point)
                active_shader.set_int("numSpotLights", num_spot)

        def draw_list(item_list: List[Any], force_depth_write: bool = True) -> None:
            """Executes rendering instructions for a categorized list of components."""
            current_mat_id = -1
            
            for tf, mesh, ent in item_list:
                model_mat = tf.get_matrix()
                active_shader.set_mat4("model", model_mat)
                
                if not is_depth_pass and not is_unlit_pass:
                    det = glm.determinant(glm.mat3(model_mat))
                    normal_mat = glm.transpose(glm.inverse(glm.mat3(model_mat))) if abs(det) > 1e-6 else glm.mat3(1.0)
                    active_shader.set_mat3("normalMatrix", normal_mat)
                
                if mesh.material:
                    if id(mesh.material) != current_mat_id and not is_depth_pass:
                        mesh.material.apply(active_shader)
                        current_mat_id = id(mesh.material)
                    
                    r_state = mesh.material.render_state
                    
                    if r_state.cull_face:
                        glEnable(GL_CULL_FACE)
                        glCullFace(r_state.cull_mode)
                    else:
                        glDisable(GL_CULL_FACE)
                        
                    if r_state.depth_test:
                        glEnable(GL_DEPTH_TEST)
                        glDepthFunc(r_state.depth_func)
                    else:
                        glDisable(GL_DEPTH_TEST)
                        
                    glDepthMask(GL_TRUE if force_depth_write and r_state.depth_write else GL_FALSE)
                    
                geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
                geom_obj.draw()

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        
        # 1. Draw Opaque (Write to Depth Buffer)
        draw_list(self.queue.opaque, force_depth_write=True)
        
        # 2. Draw Transparent (No Depth Write, Back-to-Front Painter's Algorithm)
        if not is_depth_pass:
            draw_list(self.queue.transparent, force_depth_write=False)
            
        # Restore default global GL states
        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
            
        self._render_proxies(view_matrix, projection_matrix, active_camera)

    def _render_proxies(self, view_matrix: glm.mat4, projection_matrix: glm.mat4, active_camera: Optional[CameraComponent]) -> None:
        """Renders editor-only wireframes and functional visualizers (Lights/Cameras)."""
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
                
            geom_obj.draw()

    # =========================================================================
    # GPU COLOR PICKING PIPELINE
    # =========================================================================

    def _setup_picking_fbo(self, width: int, height: int) -> None:
        """Allocates the VRAM for the off-screen Color Picking Framebuffer."""
        if self.picking_fbo is not None:
            glDeleteFramebuffers(1, [self.picking_fbo])
            glDeleteTextures(1, [self.picking_texture])
            glDeleteRenderbuffers(1, [self.picking_depth])

        self.picking_width = width
        self.picking_height = height

        self.picking_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)

        self.picking_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.picking_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.picking_texture, 0)

        self.picking_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.picking_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.picking_depth)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RenderError(f"Picking Framebuffer Object (FBO) generation failed. OpenGL Status Code: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def raycast_select(self, scene: Scene, mx: float, my: float, width: int, height: int) -> int:
        """
        Executes a discrete render pass encoding Entity IDs as RGB pixels to perform pixel-perfect 3D selection.
        """
        if width <= 0 or height <= 0 or mx < 0 or my < 0 or mx >= width or my >= height:
            return -1

        if self.picking_width != width or self.picking_height != height:
            self._setup_picking_fbo(width, height)

        glBindFramebuffer(GL_FRAMEBUFFER, self.picking_fbo)
        glViewport(0, 0, width, height)
        
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)

        self.pass_picking_shader.use()

        active_camera = None
        cam_idx = -1
        view_matrix = glm.mat4(1.0)
        projection_matrix = glm.mat4(1.0)

        for tf, cam, ent in scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                active_camera = cam
                view_matrix = cam.get_view_matrix()
                projection_matrix = cam.get_projection_matrix()
                cam_idx = scene.entities.index(ent)
                break

        if not active_camera:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return -1

        self.pass_picking_shader.set_mat4("view", view_matrix)
        self.pass_picking_shader.set_mat4("projection", projection_matrix)

        def draw_picking_list(item_list: List[Any]) -> None:
            for tf, mesh, ent in item_list:
                ent_idx = scene.entities.index(ent)
                if ent_idx == cam_idx:
                    continue
                
                # Bitwise conversion: Integer ID to distinct R, G, B float channels
                r = (ent_idx & 0x000000FF) / 255.0
                g = ((ent_idx & 0x0000FF00) >> 8) / 255.0
                b = ((ent_idx & 0x00FF0000) >> 16) / 255.0

                self.pass_picking_shader.set_vec3("u_ColorId", glm.vec3(r, g, b))
                self.pass_picking_shader.set_mat4("model", tf.get_matrix())

                geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
                geom_obj.draw()

        draw_picking_list(self.queue.opaque)
        draw_picking_list(self.queue.transparent)
        draw_picking_list(self.queue.proxies)

        # Translate Qt's top-left origin to OpenGL's bottom-left origin
        x = int(mx)
        y = int(height - my) 
        
        pixel_data = glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        
        # Restore normal rendering state machine
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)

        if pixel_data:
            r, g, b, a = pixel_data[0], pixel_data[1], pixel_data[2], pixel_data[3]
            
            if r == 255 and g == 255 and b == 255:
                return -1
                
            hit_idx = r + (g << 8) + (b << 16)
            if 0 <= hit_idx < len(scene.entities):
                return hit_idx

        return -1