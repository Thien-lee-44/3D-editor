"""
Editor-Specific Rendering Subsystem.
Handles the rendering of interactive 3D transformation manipulators (Gizmos), 
directional light proxies, and viewport orientation compasses.
"""

import glm
import math
import numpy as np
import ctypes
from OpenGL.GL import *
from typing import Any, Optional

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.geometry.primitives import PrimitivesManager

from src.app.config import (
    GIZMO_RING_SEGMENTS, GIZMO_COLOR_X, GIZMO_COLOR_Y, GIZMO_COLOR_Z, 
    GIZMO_COLOR_HOVER, GIZMO_COLOR_CORE, HUD_COMPASS_SIZE, HUD_COMPASS_OFFSET
)


class GizmoRenderer:
    """
    Renders 3D transformation manipulators (Translate, Rotate, Scale) 
    and the screen-space orientation compass for the active viewport.
    """
    
    def __init__(self) -> None:
        self.solid_shader = ResourceManager.get_shader("editor_solid")
        self.gizmo_cube = PrimitivesManager.get_primitive("Cube")
        self.gizmo_cyl = PrimitivesManager.get_primitive("Cylinder")
        self.gizmo_cone = PrimitivesManager.get_primitive("Cone")
        self._setup_circle_vbo()

    def _setup_circle_vbo(self) -> None:
        """Generates a hardware buffer for a 2D unit circle used by rotation gizmos."""
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def render(self, scene: Any, active_camera: Optional[CameraComponent], camera_tf: Optional[TransformComponent], window_w: int, window_h: int, active_axis: str) -> None:
        """
        Draws the interactive manipulation gizmo at the target transform's world position.
        Adjusts scale dynamically to maintain a constant screen-space size.
        """
        if not active_camera or not camera_tf: 
            return
            
        view_matrix = active_camera.get_view_matrix()
        proj_matrix = active_camera.get_projection_matrix()

        if getattr(scene, 'selected_index', -1) != -1 and scene.entities:
            sel_entity = scene.entities[scene.selected_index]
            sel_tf = sel_entity.get_component(TransformComponent)
            sel_renderer = sel_entity.get_component(MeshRenderer)
            sel_light = sel_entity.get_component(LightComponent)
            sel_cam = sel_entity.get_component(CameraComponent)
            
            is_visible = sel_renderer.visible if sel_renderer else True

            # Prevent rendering the gizmo if the selected entity is the active camera
            if sel_cam and sel_cam == active_camera:
                is_visible = False

            if is_visible and sel_tf: 
                is_hud_dir_light = (sel_light is not None and sel_light.type == "Directional")
                
                if not is_hud_dir_light:
                    glClear(GL_DEPTH_BUFFER_BIT) 
                    
                    self.solid_shader.use()
                    glViewport(0, 0, window_w, window_h)
                    self.solid_shader.set_mat4("view", view_matrix)
                    self.solid_shader.set_mat4("projection", proj_matrix)
                    
                    g_pos = sel_tf.global_position
                    g_rot = sel_tf.global_quat_rot

                    # Consistent pixel-size scaling logic
                    pixel_factor = 100.0 / max(window_h, 1.0) 
                    if active_camera.mode == "Perspective":
                        dist = glm.length(camera_tf.global_position - g_pos)
                        g_scale = dist * math.tan(math.radians(active_camera.fov / 2.0)) * pixel_factor
                    else:
                        g_scale = active_camera.ortho_size * pixel_factor
                        
                    base_model = glm.translate(glm.mat4(1.0), g_pos) * glm.mat4_cast(g_rot) * glm.scale(glm.mat4(1.0), glm.vec3(g_scale))
                    mode = getattr(scene, 'manipulation_mode', 'MOVE')
                    
                    mat_x = glm.rotate(glm.mat4(base_model), glm.radians(-90.0), glm.vec3(0, 0, 1))
                    mat_y = glm.mat4(base_model)
                    mat_z = glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(1, 0, 0))

                    if mode == "MOVE":
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'X' else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                            
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()

                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Y' else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                            
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()

                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Z' else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                            
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 1.1, 0)), glm.vec3(0.18, 0.3, 0.18)))
                        if self.gizmo_cone: self.gizmo_cone.draw()
                        
                    elif mode == "SCALE":
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['X', 'ALL'] else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_x, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['Y', 'ALL'] else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_y, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis in ['Z', 'ALL'] else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 0.45, 0)), glm.vec3(0.06, 0.9, 0.06)))
                        if self.gizmo_cyl: self.gizmo_cyl.draw()
                        self.solid_shader.set_mat4("model", glm.scale(glm.translate(mat_z, glm.vec3(0, 1.05, 0)), glm.vec3(0.15)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'ALL' else glm.vec3(*GIZMO_COLOR_CORE))
                        self.solid_shader.set_mat4("model", glm.scale(base_model, glm.vec3(0.25)))
                        if self.gizmo_cube: self.gizmo_cube.draw()

                    elif mode == "ROTATE":
                        glBindVertexArray(self.main_vao)
                        
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'X' else glm.vec3(*GIZMO_COLOR_X))
                        self.solid_shader.set_mat4("model", glm.scale(glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(0, 1, 0)), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'X' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Y' else glm.vec3(*GIZMO_COLOR_Y))
                        self.solid_shader.set_mat4("model", glm.scale(glm.rotate(glm.mat4(base_model), glm.radians(90.0), glm.vec3(1, 0, 0)), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'Y' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_HOVER) if active_axis == 'Z' else glm.vec3(*GIZMO_COLOR_Z))
                        self.solid_shader.set_mat4("model", glm.scale(glm.mat4(base_model), glm.vec3(1.5)))
                        glLineWidth(3.0 if active_axis == 'Z' else 1.0)
                        glDrawArrays(GL_LINE_LOOP, 0, self.circle_segments)
                        
                        glLineWidth(1.0)
                        glBindVertexArray(0)
                        
                    glEnable(GL_DEPTH_TEST)

        if getattr(scene, 'show_screen_axis', True):
            glViewport(window_w - HUD_COMPASS_OFFSET, window_h - HUD_COMPASS_OFFSET, HUD_COMPASS_SIZE, HUD_COMPASS_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT) 
            self.solid_shader.use()
            
            axis_view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -3.0)) * glm.mat4(glm.mat3(view_matrix))
            axis_proj = glm.perspective(glm.radians(45.0), 1.0, 0.1, 10.0)
            
            def draw_corner_arrow(axis_char: str, color: glm.vec3) -> None:
                """Helper to render individual axes on the orientation compass."""
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

            self.solid_shader.set_vec3("solidColor", glm.vec3(*GIZMO_COLOR_CORE))
            self.solid_shader.set_mat4("model", glm.scale(glm.mat4(1.0), glm.vec3(0.15)))
            if self.gizmo_cube: self.gizmo_cube.draw()

            draw_corner_arrow('X', glm.vec3(*GIZMO_COLOR_X))
            draw_corner_arrow('Y', glm.vec3(*GIZMO_COLOR_Y))
            draw_corner_arrow('Z', glm.vec3(*GIZMO_COLOR_Z))
            draw_corner_arrow('-X', glm.vec3(0.5, 0.5, 0.5))
            draw_corner_arrow('-Y', glm.vec3(0.5, 0.5, 0.5))
            draw_corner_arrow('-Z', glm.vec3(0.5, 0.5, 0.5))


class HUDRenderer:
    """
    Handles rendering for the directional light (Sun) manipulator HUD, 
    drawn in an isolated off-screen context.
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
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)

        self.sun_vao = None
        self.sun_index_count = 0
        self.sun_vertex_count = 0
        self.has_ebo = False
        
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
        """
        Executes the render pass for the Sun HUD, overlaying the rotation rings 
        and proxy mesh onto the isolated viewport.
        """
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