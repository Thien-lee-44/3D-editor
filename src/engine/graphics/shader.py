import os
import glm
from OpenGL.GL import *
from typing import Any

from src.app.exceptions import ShaderError, ResourceError

class Shader:
    """
    Handles the complete lifecycle of GLSL Shader Programs: from disk reading, 
    compilation, hardware linking, to runtime uniform injection.
    """
    
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
        """
        Compiles individual shader stages and links them into an executable GPU program.
        Raises explicit ShaderError exceptions to halt execution immediately upon syntax errors.
        """
        # Vertex Shader Compilation Pipeline
        v_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(v_shader, v_src)
        glCompileShader(v_shader)
        
        if not glGetShaderiv(v_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(v_shader)
            raise ShaderError(f"VERTEX SHADER COMPILATION FAILED:\n{info_log.decode('utf-8')}")
        
        # Fragment Shader Compilation Pipeline
        f_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(f_shader, f_src)
        glCompileShader(f_shader)

        if not glGetShaderiv(f_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(f_shader)
            raise ShaderError(f"FRAGMENT SHADER COMPILATION FAILED:\n{info_log.decode('utf-8')}")

        # Linking Stage
        prog = glCreateProgram()
        glAttachShader(prog, v_shader)
        glAttachShader(prog, f_shader)
        glLinkProgram(prog)
        
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            info_log = glGetProgramInfoLog(prog)
            raise ShaderError(f"SHADER PROGRAM LINKING FAILED:\n{info_log.decode('utf-8')}")
        
        # Cleanup: Intermediate shader objects are no longer required after successful linking
        glDeleteShader(v_shader)
        glDeleteShader(f_shader)
        
        return prog

    def use(self) -> None:
        """Activates the shader program in the current OpenGL context state machine."""
        glUseProgram(self.program)

    # =========================================================================
    # UNIFORM INJECTION API
    # Handles data transmission across the CPU-GPU boundary.
    # =========================================================================
    
    def set_mat4(self, name: str, mat: Any) -> None:
        """Transmits a 4x4 matrix (e.g., Model, View, Projection matrices)."""
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniformMatrix4fv(loc, 1, GL_FALSE, glm.value_ptr(mat))

    def set_mat3(self, name: str, mat: Any) -> None:
        """Transmits a 3x3 matrix (typically used for Normal matrix transformations)."""
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniformMatrix3fv(loc, 1, GL_FALSE, glm.value_ptr(mat))

    def set_vec3(self, name: str, vec: Any) -> None:
        """Transmits a 3-component vector (e.g., RGB Colors, 3D Positions, Directions)."""
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform3fv(loc, 1, glm.value_ptr(vec))

    def set_float(self, name: str, value: float) -> None:
        """Transmits a scalar float value."""
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform1f(loc, value)

    def set_int(self, name: str, value: int) -> None:
        """Transmits an integer (often used for texture unit binding or boolean flags)."""
        loc = glGetUniformLocation(self.program, name)
        if loc != -1: 
            glUniform1i(loc, value)