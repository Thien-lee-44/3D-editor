"""
Editor Context Renderer.

Handles the visualization of Editor-specific overlays including Viewport Gizmos, 
Entity Bounding Boxes, Wireframes, and hardware-accelerated Mouse Picking.
"""

import math
import ctypes
import numpy as np
import glm
from OpenGL.GL import *
from typing import Any, Optional, Tuple, List, Union

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.geometry.primitives import PrimitivesManager
from src.engine.graphics.base_renderer import BaseRenderer

from src.app.exceptions import RenderError
from src.app.config import (
    GIZMO_RING_SEGMENTS, GIZMO_COLOR_X, GIZMO_COLOR_Y, GIZMO_COLOR_Z, 
    GIZMO_COLOR_HOVER, GIZMO_COLOR_CORE, HUD_COMPASS_SIZE, HUD_COMPASS_OFFSET,
    MAX_LIGHTS
)


class GizmoRenderer:
    """
    Renders 3D interactive handles (Translate, Rotate, Scale) in the viewport.
    Provides visual feedback during entity manipulation.
    """
    
    def __init__(self) -> None:
        self.solid_shader = ResourceManager.get_shader("editor_solid")
        self.gizmo_cube = PrimitivesManager.get_primitive("Cube")
        self.gizmo_cyl = PrimitivesManager.get_primitive("Cylinder")
        self.gizmo_cone = PrimitivesManager.get_primitive("Cone")
        self._setup_circle_vbo()

    def _setup_circle_vbo(self) -> None:
        """Constructs a procedural ring mesh for Rotation Gizmo Handles."""
        self.circle_segments = GIZMO_RING_SEGMENTS
        pts = []
        for i in range(self.circle_segments):
            angle = 2.0 * math.pi * i / self.circle_segments
            pts.extend([math.cos(angle), math.sin(angle), 0.0])
            
        pts_arr = np.array(pts, dtype=np.float32)
        
        self.circle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.circle_vbo)
        glBufferData(GL_ARRAY_BUFFER, pts_arr.nbytes, pts_arr, GL_STATIC_DRAW)
        
        self.main_vao = glGenVertexArrays(1)
        glBindVertexArray(self.main_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.circle_vbo)
        glEnableVertexAttribArray(0)
        # Stride is 12 bytes (3 floats * 4 bytes)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def render(self, scene: Any, active_camera: Optional[CameraComponent], camera_tf: Optional[TransformComponent], window_w: int, window_h: int, active_axis: str) -> None:
        """
        Calculates matrix transformations and renders the manipulation handles 
        (Arrows, Rings, Cubes) at the selected entity's position.
        """
        if not active_camera or not camera_tf: 
            return
            
        view_matrix = active_camera.get_view_matrix()
        proj_matrix = active_camera.get_projection_matrix()

        # Render Main Viewport Gizmo if an entity is selected
        if getattr(scene, 'selected_index', -1) != -1 and scene.entities:
            sel_entity = scene.entities[scene.selected_index]
            sel_tf = sel_entity.get_component(TransformComponent)
            sel_renderer = sel_entity.get_component(MeshRenderer)
            sel_light = sel_entity.get_component(LightComponent)
            sel_cam = sel_entity.get_component(CameraComponent)
            
            is_visible = sel_renderer.visible if sel_renderer else True
            
            # Hide gizmo if the selected entity is the active camera itself
            if sel_cam and sel_cam == active_camera:
                is_visible = False

            if is_visible and sel_tf: 
                is_hud_dir_light = (sel_light is not None and sel_light.type == "Directional")
                
                # Directional lights do not have physical positions in the scene to attach gizmos to
                if not is_hud_dir_light:
                    glClear(GL_DEPTH_BUFFER_BIT) 
                    
                    self.solid_shader.use()
                    glViewport(0, 0, window_w, window_h)
                    self.solid_shader.set_mat4("view", view_matrix)
                    self.solid_shader.set_mat4("projection", proj_matrix)
                    
                    g_pos = sel_tf.global_position
                    g_rot = sel_tf.global_quat_rot

                    # Maintain a constant visual scale for the gizmo regardless of camera distance
                    pixel_factor = 100.0 / max(window_h, 1.0) 
                    if active_camera.mode == "Perspective":
                        dist = glm.length(camera_tf.global_position - g_pos)
                        g_scale = dist * math.tan(math.radians(active_camera.fov / 2.0)) * pixel_factor
                    else:
                        g_scale = active_camera.ortho_size * pixel_factor
                        
                    base_model = glm.translate(glm.mat4(1.0), g_pos) * glm.mat4_cast(g_rot) * glm.scale(glm.mat4(1.0), glm.vec3(g_scale))
                    mode = getattr(scene, 'manipulation_mode', 'MOVE')
                    
                    # Pre-calculate orientation matrices for each axis
                    mat_x = glm.rotate(glm.mat4(base_model), glm.radians(-90.0), glm.vec3(0, 0, 1))
                    mat_y = glm.mat4(base_model)
                    mat_z = glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(1, 0, 0))

                    if mode == "MOVE":
                        # X-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'X' else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()

                        # Y-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Y' else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()

                        # Z-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Z' else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()
                        
                    elif mode == "SCALE":
                        # X-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['X', 'ALL'] else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        # Y-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['Y', 'ALL'] else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        # Z-Axis Handle
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['Z', 'ALL'] else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        # Center Scale Handle (Uniform Scale)
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'ALL' else glm.vec3(*GIZMO_COLOR_CORE))
                        self.solid_shader.set_mat4("model", glm.scale(base_model, glm.vec3(0.25)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                    elif mode == "ROTATE":
                        glBindVertexArray(self.main_vao)
                        
                        # X-Axis Ring
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'X' else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(0, 1, 0)), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'X' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        # Y-Axis Ring
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Y' else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(1, 0, 0)), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'Y' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        # Z-Axis Ring
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Z' else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.mat4(base_model), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'Z' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        glLineWidth(1.0)
                        glBindVertexArray(0)
                        
                    glEnable(GL_DEPTH_TEST)

        # Render Top-Right Orientation Compass (HUD)
        if getattr(scene, 'show_screen_axis', True):
            glViewport(window_w - HUD_COMPASS_OFFSET, window_h - HUD_COMPASS_OFFSET, HUD_COMPASS_SIZE, HUD_COMPASS_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT) 
            self.solid_shader.use()
            
            # Create a fixed orthographic-style view for the compass
            axis_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -3.0)) * glm.mat4(glm.mat3(view_matrix))
            axis_proj = glm.perspective(glm.radians(45.0), 1.0, 0.1, 10.0)
            
            def draw_corner_arrow(axis_char: str, color: glm.vec3) -> None:
                """Helper to draw compass arrows pointing in world-space directions."""
                self.solid_shader.set_vec3("solidColor", color)
                mat = glm.mat4(1.0)
                
                if axis_char == 'X': mat = glm.rotate(mat, glm.radians(-90.0), glm.vec3(0, 0, 1))
                elif axis_char == 'Z': mat = glm.rotate(mat, glm.radians(90.0), glm.vec3(1, 0, 0))
                elif axis_char == '-X': mat = glm.rotate(mat, glm.radians(90.0), glm.vec3(0, 0, 1))
                elif axis_char == '-Y': mat = glm.rotate(mat, glm.radians(180.0), glm.vec3(1, 0, 0))
                elif axis_char == '-Z': mat = glm.rotate(mat, glm.radians(-90.0), glm.vec3(1, 0, 0))
                
                self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                if self.gizmo_cyl: self.gizmo_cyl.draw()
                    
                self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat, glm.vec3(0, 1.0, 0)), glm.vec3(0.18, 0.3, 0.18)))
                if self.gizmo_cone: self.gizmo_cone.draw()

            self.solid_shader.set_mat4("view", axis_view)
            self.solid_shader.set_mat4("projection", axis_proj)

            # Center node of the compass
            self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_CORE))
            self.solid_shader.set_mat4("model", glm.scale(glm.mat4(1.0), glm.vec3(0.15)))
            if self.gizmo_cube: self.gizmo_cube.draw()

            draw_corner_arrow('X', glm.vec3(*GIZMO_COLOR_X))
            draw_corner_arrow('Y', glm.vec3(*GIZMO_COLOR_Y))
            draw_corner_arrow('Z', glm.vec3(*GIZMO_COLOR_Z))
            
            # Draw negative axes in gray
            draw_corner_arrow('-X', glm.vec3(0.5, 0.5, 0.5))
            draw_corner_arrow('-Y', glm.vec3(0.5, 0.5, 0.5))
            draw_corner_arrow('-Z', glm.vec3(0.5, 0.5, 0.5))


class HUDRenderer:
    """
    Renders 2D/3D overlays specifically tied to light orientations 
    or isolated viewport overlays independent of the main Scene Graph.
    """
    
    def __init__(self) -> None:
        self.solid_shader = ResourceManager.get_shader("editor_solid")
        self.proxy_shader = ResourceManager.get_shader("editor_proxy")
        self.sun_proxy = PrimitivesManager.get_proxy("proxy_dir.ply")
        
        self.circle_segments = GIZMO_RING_SEGMENTS
        pts = []
        for i in range(self.circle_segments):
            angle = 2.0 * math.pi * i / self.circle_segments
            pts.extend([math.cos(angle), math.sin(angle), 0.0])
            
        pts_arr = np.array(pts, dtype=np.float32)
        
        self.circle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.circle_vbo)
        glBufferData(GL_ARRAY_BUFFER, pts_arr.nbytes, pts_arr, GL_STATIC_DRAW)
        
        self.circle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.circle_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.circle_vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        self.sun_vao = None
        self.sun_index_count = 0
        self.sun_vertex_count = 0
        self.has_ebo = False
        
        # Pre-configure Sun Proxy VBO for Directional Light HUD
        if self.sun_proxy:
            self.sun_vao = glGenVertexArrays(1)
            glBindVertexArray(self.sun_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.sun_proxy.vbo)
            
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.sun_proxy.vertex_size * 4, ctypes.c_void_p(0))
            
            if self.sun_proxy.vertex_size >= 11:
                glEnableVertexAttribArray(3)
                glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, self.sun_proxy.vertex_size * 4, ctypes.c_void_p(8 * 4))

            if self.sun_proxy.ebo:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sun_proxy.ebo)
                self.has_ebo = True
                self.sun_index_count = len(self.sun_proxy.indices) if self.sun_proxy.indices is not None else 0
            else:
                self.sun_vertex_count = len(self.sun_proxy.vertices) // self.sun_proxy.vertex_size if hasattr(self.sun_proxy, 'vertices') else 0
                
            glBindVertexArray(0)

    def render(self, w: int, h: int, active_axis: str, is_hover: bool, target_tf: TransformComponent, view_matrix: glm.mat4) -> None:
        """Renders the directional light isolated rotation HUD in the Inspector panel."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        hud_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2.5)) * view_matrix
        proj = glm.perspective(glm.radians(45.0), w / max(h, 1), 0.1, 10.0)
        model = glm.mat4_cast(target_tf.global_quat_rot)

        if self.sun_vao:
            self.proxy_shader.use()
            self.proxy_shader.set_mat4("view", hud_view)
            self.proxy_shader.set_mat4("projection", proj)
            self.proxy_shader.set_mat4("model", glm.scale(model, glm.vec3(0.5)))
            
            glBindVertexArray(self.sun_vao)
            if self.has_ebo: 
                glDrawElements(GL_TRIANGLES, self.sun_index_count, GL_UNSIGNED_INT, None)
            else: 
                glDrawArrays(GL_TRIANGLES, 0, self.sun_vertex_count)
            glBindVertexArray(0)

        self.solid_shader.use()
        self.solid_shader.set_mat4("view", hud_view)
        self.solid_shader.set_mat4("projection", proj)
        
        glBindVertexArray(self.circle_vao)
        axes_data = [('X', glm.vec3(*GIZMO_COLOR_X)), ('Y', glm.vec3(*GIZMO_COLOR_Y)), ('Z', glm.vec3(*GIZMO_COLOR_Z))]
        
        for axis, default_color in axes_data:
            color = glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == axis and is_hover else default_color
            
            mat = glm.rotate(glm.mat4(model), glm.radians(90.0), glm.vec3(0, 1, 0) if axis == 'X' else (glm.vec3(1, 0, 0) if axis == 'Y' else glm.vec3(0, 0, 1)))
            mat = glm.scale(mat, glm.vec3(0.8))
            
            self.solid_shader.set_vec3("solidColor", color)
            self.solid_shader.set_mat4("model", mat)
            
            glLineWidth(3.0 if active_axis == axis else 1.0)
            glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
            
        glLineWidth(1.0)
        glBindVertexArray(0)


class Renderer(BaseRenderer):
    """
    Main Viewport Engine orchestrating the scene render loop alongside 
    UI Overlays, Bounding Boxes, and hardware-accelerated Mouse Picking.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.editor_solid_shader = ResourceManager.get_shader("editor_solid")
        self.editor_proxy_shader = ResourceManager.get_shader("editor_proxy")
        
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

    def toggle_wireframe(self) -> None:
        self.wireframe = not self.wireframe

    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        """Applies global viewport render settings (e.g. Unlit mode, Wireframe)."""
        self.wireframe = wireframe
        self.render_mode = mode
        self.output_type = output
        self.comb_light = light
        self.comb_tex = tex
        self.comb_vcolor = vcolor

    def _get_active_camera_data(self, scene: Any) -> Tuple[Optional[CameraComponent], glm.mat4, glm.mat4, glm.vec3, int]:
        for tf, cam, ent in scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                return cam, cam.get_view_matrix(), cam.get_projection_matrix(), tf.global_position, scene.entities.index(ent)
        return None, glm.mat4(1.0), glm.mat4(1.0), glm.vec3(0), -1

    def _render_passes(self, scene: Any, active_camera: CameraComponent, cam_pos: glm.vec3, view_matrix: glm.mat4, projection_matrix: glm.mat4, light_space_matrix: glm.mat4, is_depth_pass: bool, is_unlit_pass: bool, use_light: bool, use_tex: bool, use_vcolor: bool) -> None:
        """Executes drawing protocols for opaque and transparent scene layers."""
        if is_depth_pass: active_shader = self.pass_depth_shader
        elif is_unlit_pass: active_shader = self.mat_unlit_shader
        else: active_shader = self.mat_standard_shader

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

        # Opaque Pass
        self._draw_geometry_list(self.queue.opaque, active_shader, is_depth_pass, is_unlit_pass, True, is_shadow_pass=False)
        
        # Transparent Pass
        force_transparent_depth = True if is_depth_pass else False
        self._draw_geometry_list(self.queue.transparent, active_shader, is_depth_pass, is_unlit_pass, force_transparent_depth, is_shadow_pass=False)
            
        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)

    def _render_proxies(self, view_matrix: glm.mat4, projection_matrix: glm.mat4, active_camera: CameraComponent) -> None:
        """Draws Editor-only proxies like camera and light bounds."""
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for tf, mesh, ent in self.queue.proxies:
            if not self._is_entity_globally_visible(ent): 
                continue
                
            # Do not draw proxy for the camera currently being looked through
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
        """Main rendering loop entry point for the Qt viewport."""
        if not scene: return
        cam, view_mat, proj_mat, cam_pos, _ = self._get_active_camera_data(scene)
        if not cam: return

        self.queue.build(scene, cam_pos)
        
        # 1. Shadow Map Processing
        light_space_mat = glm.mat4(1.0)
        if self.comb_light and self.render_mode == 4:
            light_space_mat = self._get_light_space_matrix(scene)
            self._render_shadow_pass(light_space_mat, self.queue.opaque + self.queue.transparent)

        # 2. Main Color Output
        glViewport(0, 0, window_w, window_h)
        is_depth_pass = (self.output_type == 1)
        is_unlit_pass = (self.render_mode != 4)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        if is_depth_pass:
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._pass_depth(view_mat, proj_mat, float(cam.near), float(cam.far), self.queue.opaque + self.queue.transparent, cam.mode == "Orthographic")
        else:
            self._render_passes(scene, cam, cam_pos, view_mat, proj_mat, light_space_mat, is_depth_pass, is_unlit_pass, self.comb_light, self.comb_tex, self.comb_vcolor)
            self._render_proxies(view_mat, proj_mat, cam)

    def _setup_picking_fbo(self, width: int, height: int) -> None:
        """Initializes off-screen framebuffers for hardware color picking."""
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        if self.picking_fbo is not None:
            glDeleteFramebuffers(1, [self.picking_fbo])
            glDeleteTextures(2, [self.picking_texture, self.picking_depth])
        if getattr(self, 'msaa_fbo', None) is not None:
            glDeleteFramebuffers(1, [self.msaa_fbo])
            glDeleteRenderbuffers(2, [self.msaa_color, self.msaa_depth])

        self.picking_width = width
        self.picking_height = height

        # MSAA Buffer to reduce jagged edge selection errors
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

        # Picking Framebuffer Output
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

    def raycast_select(self, scene: Any, mx: float, my: float, width: int, height: int) -> int:
        """
        Translates an (X,Y) mouse click into a 3D Entity selection via Color-ID encoding.
        Bitwise shift algorithm embeds the entity index into RGB channels.
        """
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
                if not self._is_entity_globally_visible(ent): 
                    continue
                    
                ent_idx = scene.entities.index(ent)
                if ent_idx == cam_idx: 
                    continue
                
                # Bitwise shift to encode integer ID into 8-bit RGB channels
                r = (ent_idx & 0x000000FF) / 255.0
                g = ((ent_idx & 0x0000FF00) >> 8) / 255.0
                b = ((ent_idx & 0x00FF0000) >> 16) / 255.0
                
                self.pass_picking_shader.set_vec3("u_ColorId", glm.vec3(r, g, b))
                self.pass_picking_shader.set_mat4("model", tf.get_matrix())

                geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
                if hasattr(geom_obj, 'draw'): 
                    geom_obj.draw()

        x = int(mx)
        # Invert Y to match OpenGL Bottom-Left coordinate origin
        y = int(height - my) 
        pixel_data = glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)

        if pixel_data:
            r, g, b, _ = pixel_data[0], pixel_data[1], pixel_data[2], pixel_data[3]
            # White (255,255,255) represents the clear background
            if r == 255 and g == 255 and b == 255: 
                return -1
                
            hit_idx = r + (g << 8) + (b << 16)
            if 0 <= hit_idx < len(scene.entities): 
                return hit_idx
                
        return -1