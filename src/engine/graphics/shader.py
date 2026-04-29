"""
GLSL Shader Program Manager.

Handles the complete lifecycle of OpenGL shaders: disk reading, compilation, 
hardware linking, and runtime uniform variable injection.
"""

import os
import glm
from OpenGL.GL import *
from typing import Any

from src.app.exceptions import ShaderError, ResourceError


class Shader:
    """Manages an executable GPU program pipeline."""
    
    def __init__(self, vertex_path: str, fragment_path: str) -> None:
        v_src = self._read_file(vertex_path)
        f_src = self._read_file(fragment_path)
        self.program = self._compile_shaders(v_src, f_src)

    def _read_file(self, filepath: str) -> str:
        """Reads raw GLSL source code from disk."""
        if not os.path.exists(filepath):
            raise ResourceError(f"Shader source file missing: '{filepath}'")
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _compile_shaders(self, v_src: str, f_src: str) -> int:
        """Compiles individual shader stages and links them. Halts strictly upon syntax errors."""
        # Vertex Shader
        v_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(v_shader, v_src)
        glCompileShader(v_shader)
        
        if not glGetShaderiv(v_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(v_shader)
            raise ShaderError(f"VERTEX SHADER COMPILATION FAILED:\n{info_log.decode('utf-8')}")
        
        # Fragment Shader
        f_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(f_shader, f_src)
        glCompileShader(f_shader)

        if not glGetShaderiv(f_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(f_shader)
            raise ShaderError(f"FRAGMENT SHADER COMPILATION FAILED:\n{info_log.decode('utf-8')}")

        # Linking
        prog = glCreateProgram()
        glAttachShader(prog, v_shader)
        glAttachShader(prog, f_shader)
        glLinkProgram(prog)
        
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            info_log = glGetProgramInfoLog(prog)
            raise ShaderError(f"SHADER PROGRAM LINKING FAILED:\n{info_log.decode('utf-8')}")
        
        glDeleteShader(v_shader)
        glDeleteShader(f_shader)
        
        return prog

    def use(self) -> None:
        """Activates the shader program in the OpenGL state machine."""
        glUseProgram(self.program)

    # =========================================================================
    # UNIFORM INJECTION API (CPU to GPU transmission)
    # =========================================================================
    
    def set_mat4(self, name: str, mat: Any) -> None:
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniformMatrix4fv(loc, 1, GL_FALSE, glm.value_ptr(mat))

    def set_mat3(self, name: str, mat: Any) -> None:
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniformMatrix3fv(loc, 1, GL_FALSE, glm.value_ptr(mat))

    def set_vec3(self, name: str, vec: Any) -> None:
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform3fv(loc, 1, glm.value_ptr(vec))

    def set_float(self, name: str, value: float) -> None:
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform1f(loc, value)

    def set_int(self, name: str, value: int) -> None:
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform1i(loc, value)