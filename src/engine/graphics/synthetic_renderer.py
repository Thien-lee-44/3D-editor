import glm
import numpy as np
import colorsys
import ctypes
from OpenGL.GL import *
from typing import Any, Dict, List, Tuple, Union, Optional

from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.graphics.base_renderer import BaseRenderer
from src.engine.synthetic.label_utils import LabelUtils

class SyntheticRenderer(BaseRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.fbo: int = 0
        self.tex_rgb: int = 0
        self.tex_mask: int = 0
        self.rbo_depth: int = 0
        
        self.msaa_fbo: int = 0
        self.msaa_color: int = 0
        self.msaa_depth: int = 0
        self.msaa_samples: int = 8
        
        self.current_res: Tuple[int, int] = (0, 0)
        
        self._pixel_buffer: Optional[np.ndarray] = None
        self._buffer_res: Tuple[int, int] = (0, 0)

    def _get_active_camera_data(self, scene: Any) -> Tuple[Optional[Any], glm.mat4, glm.mat4, glm.vec3, int]:
        for tf, cam, ent in scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                return cam, cam.get_view_matrix(), cam.get_projection_matrix(), tf.global_position, scene.entities.index(ent)
        return None, glm.mat4(1.0), glm.mat4(1.0), glm.vec3(0), -1

    def _ensure_fbo(self, width: int, height: int) -> None:
        if (width, height) == self.current_res:
            return

        if self.fbo != 0:
            glDeleteFramebuffers(2, [self.fbo, self.msaa_fbo])
            glDeleteTextures(2, [self.tex_rgb, self.tex_mask])
            glDeleteRenderbuffers(3, [self.rbo_depth, self.msaa_color, self.msaa_depth])

        self.msaa_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)
        
        self.msaa_color = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_color)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.msaa_samples, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.msaa_color)
        
        self.msaa_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_depth)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.msaa_samples, GL_DEPTH_COMPONENT32F, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.msaa_depth)

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        self.tex_rgb = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_rgb)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex_rgb, 0)

        self.tex_mask = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_mask)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.tex_mask, 0)

        self.rbo_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo_depth)

        self.current_res = (width, height)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _pass_standard(self, scene: Any, view: glm.mat4, proj: glm.mat4, cam_pos: glm.vec3, light_space_mat: glm.mat4) -> None:
        self.mat_standard_shader.use()
        self.mat_standard_shader.set_mat4("view", view)
        self.mat_standard_shader.set_mat4("projection", proj)
        self.mat_standard_shader.set_vec3("viewPos", cam_pos)
        
        self.mat_standard_shader.set_int("combLight", 1)
        self.mat_standard_shader.set_int("combTex", 1)
        self.mat_standard_shader.set_int("combVColor", 1)
        
        glActiveTexture(GL_TEXTURE1) 
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        self.mat_standard_shader.set_int("shadowMap", 1)
        self.mat_standard_shader.set_mat4("lightSpaceMatrix", light_space_mat)
        
        self._apply_lighting(scene, self.mat_standard_shader)
        self._draw_geometry_list(self.queue.opaque + self.queue.transparent, self.mat_standard_shader, False, False, True)

    def _pass_mask(self, scene: Any, view: glm.mat4, proj: glm.mat4, mode: str, is_preview: bool = False) -> None:
        self.pass_picking_shader.use()
        self.pass_picking_shader.set_mat4("view", view)
        self.pass_picking_shader.set_mat4("projection", proj)

        classes = {}
        try:
            from src.app import ctx
            classes = ctx.engine.get_semantic_classes()
        except Exception: pass

        target_map = {}
        if mode == "INSTANCE":
            from src.engine.scene.components import MeshRenderer
            dense_id = 1
            for e in scene.entities:
                e_sem = e.get_component(SemanticComponent)
                e_mesh = e.get_component(MeshRenderer)
                if e_sem and e_mesh and getattr(e_mesh, 'visible', True) and not getattr(e_mesh, 'is_proxy', False):
                    t = e
                    p = e.parent
                    while p:
                        p_sem = p.get_component(SemanticComponent)
                        if p_sem and getattr(p_sem, 'is_merged_instance', False):
                            t = p
                        p = p.parent
                    if t not in target_map:
                        target_map[t] = dense_id
                        dense_id += 1

        golden_ratio_conjugate = 0.618033988749895

        for tf, mesh, ent in self.queue.opaque + self.queue.transparent:
            if not self._is_entity_globally_visible(ent) or getattr(mesh, 'is_proxy', False):
                continue
                
            semantic = ent.get_component(SemanticComponent)

            if mode == "INSTANCE":
                track_id = -1
                if semantic and getattr(semantic, 'track_id', -1) > 0:
                    track_id = semantic.track_id
                else:
                    target = ent
                    p = ent.parent
                    while p:
                        p_sem = p.get_component(SemanticComponent)
                        if p_sem and getattr(p_sem, 'is_merged_instance', False):
                            target = p
                        p = p.parent
                    track_id = target_map.get(target, -1)

                if track_id <= 0: track_id = scene.entities.index(ent) + 1
                
                if is_preview:
                    hue = (track_id * golden_ratio_conjugate) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
                    color = glm.vec3(r, g, b)
                else:
                    color = glm.vec3((track_id & 0xFF)/255.0, ((track_id & 0xFF00)>>8)/255.0, ((track_id & 0xFF0000)>>16)/255.0)
            else:
                c_id = semantic.class_id if semantic else 0
                c_color = classes.get(c_id, {}).get("color", [1.0, 1.0, 1.0])
                if max(c_color) > 1.0: c_color = [c / 255.0 for c in c_color]
                color = glm.vec3(*c_color[:3])

            self.pass_picking_shader.set_vec3("u_ColorId", color)
            self.pass_picking_shader.set_mat4("model", tf.get_matrix())
            geom_obj = getattr(mesh.geometry, 'mesh', mesh.geometry)
            if hasattr(geom_obj, 'draw'): geom_obj.draw()

    def capture_fbo_frames(self, scene: Any, width: int, height: int, modes: List[str], download: bool = True, bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, bytes]:
        if width <= 0 or height <= 0:
            return {}

        cam, view_mat, proj_mat, cam_pos, _ = self._get_active_camera_data(scene)
        if not cam:
            return {}

        requested: List[str] = []
        for mode in modes:
            if mode in ["RGB", "DEPTH", "SEMANTIC", "INSTANCE", "INSTANCE_PREVIEW"] and mode not in requested:
                requested.append(mode)

        if not requested:
            return {}

        self.queue.build(scene, cam_pos)
        self._ensure_fbo(width, height)

        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)

        light_space_mat = glm.mat4(1.0)
        if "RGB" in requested:
            light_space_mat = self._get_light_space_matrix(scene)
            self._render_shadow_pass(light_space_mat, self.queue.opaque + self.queue.transparent)

        outputs: Dict[str, bytes] = {}

        if "RGB" in requested:
            glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)
            glViewport(0, 0, width, height)
            glDrawBuffer(GL_COLOR_ATTACHMENT0)
            
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_DEPTH_TEST)
            
            glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            self._pass_standard(scene, view_mat, proj_mat, cam_pos, light_space_mat)
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.msaa_fbo)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.fbo)
            glDrawBuffer(GL_COLOR_ATTACHMENT0)
            
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
            
            if download:
                glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
                glReadBuffer(GL_COLOR_ATTACHMENT0)
                outputs["RGB"] = self._download_pixels("RGB", width, height, cam)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, width, height)
        
        glDisable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)

        for mode in requested:
            if mode == "RGB": continue 
            
            if mode == "DEPTH":
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                glClear(GL_DEPTH_BUFFER_BIT)
                self._pass_depth(view_mat, proj_mat, float(cam.near), float(cam.far), self.queue.opaque + self.queue.transparent, cam.mode == "Orthographic")
                if download:
                    outputs["DEPTH"] = self._download_pixels("DEPTH", width, height, cam)
            elif mode == "INSTANCE_PREVIEW":
                glDrawBuffer(GL_COLOR_ATTACHMENT1)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self._pass_mask(scene, view_mat, proj_mat, "INSTANCE", is_preview=True)
                if download:
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
                    glReadBuffer(GL_COLOR_ATTACHMENT1)
                    outputs["INSTANCE_PREVIEW"] = self._download_pixels("MASK", width, height, cam)
            elif mode in ["SEMANTIC", "INSTANCE"]:
                glDrawBuffer(GL_COLOR_ATTACHMENT1)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self._pass_mask(scene, view_mat, proj_mat, mode, is_preview=False)
                if download:
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
                    glReadBuffer(GL_COLOR_ATTACHMENT1)
                    outputs[mode] = self._download_pixels("MASK", width, height, cam)

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])

        glEnable(GL_MULTISAMPLE)

        return outputs

    def capture_fbo_frame(self, scene: Any, width: int, height: int, mode: str = "RGB", return_texture_id: bool = False, is_preview: bool = False, bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Union[bytes, int]:
        target_mode = "INSTANCE_PREVIEW" if mode == "INSTANCE" and is_preview else mode
        
        if width <= 0 or height <= 0: return 0 if return_texture_id else b""
        
        outputs = self.capture_fbo_frames(scene, width, height, [target_mode], download=not return_texture_id, bg_color=bg_color)
        
        if return_texture_id:
            if target_mode == "RGB": return self.tex_rgb
            elif target_mode == "DEPTH": return self.rbo_depth
            else: return self.tex_mask
            
        return outputs.get(target_mode, b"")

    def _download_pixels(self, attachment_type: str, width: int, height: int, cam: Any) -> bytes:
        if attachment_type == "DEPTH":
            depth_raw = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
            depth_buffer = np.frombuffer(depth_raw, dtype=np.float32)
            near = float(getattr(cam, 'near', 0.1))
            far = float(getattr(cam, 'far', 100.0))
            is_ortho = getattr(cam, 'mode', 'Perspective') == 'Orthographic'
            depth_8bit = self._linearize_depth_buffer(depth_buffer, near, far, "Orthographic" if is_ortho else "Perspective")
            return depth_8bit.tobytes()
        else:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
            glReadBuffer(GL_COLOR_ATTACHMENT0 if attachment_type == "RGB" else GL_COLOR_ATTACHMENT1)
            
            if self._buffer_res != (width, height) or self._pixel_buffer is None:
                self._pixel_buffer = np.empty((height, width, 4), dtype=np.uint8)
                self._buffer_res = (width, height)
                
            glReadPixels(
                0, 0, width, height, 
                GL_RGBA, 
                GL_UNSIGNED_BYTE, 
                ctypes.c_void_p(self._pixel_buffer.ctypes.data)
            )
            
            return np.ascontiguousarray(self._pixel_buffer[:, :, :3]).tobytes()