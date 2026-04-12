import math
import numpy as np
from typing import Tuple
from src.engine.graphics.buffer_objects import BufferObject
from src.app.exceptions import SimulationError
from src.app.config import DEFAULT_MATH_FORMULA, DEFAULT_MATH_RANGE, DEFAULT_MATH_RESOLUTION

class MathSurface(BufferObject):
    """
    Procedurally generates a 3D surface mesh from a string-based mathematical equation f(x, y).
    Algorithm: Implements Adaptive Tessellation via Non-linear Space Warping.
    Instead of uniformly sampling the grid, it computes the first-order derivatives (gradients) 
    of the surface and allocates higher vertex density to regions with steep slopes.
    """
    
    def __init__(self, formula_str: str = DEFAULT_MATH_FORMULA, 
                 x_range: Tuple[float, float] = DEFAULT_MATH_RANGE, 
                 y_range: Tuple[float, float] = DEFAULT_MATH_RANGE, 
                 resolution: int = DEFAULT_MATH_RESOLUTION) -> None:
                     
        if callable(formula_str):
            formula_str = DEFAULT_MATH_FORMULA
            
        self.formula_str = formula_str
        vertices = []
        indices = []
        
        # Sandbox the evaluation environment to prevent arbitrary code execution vulnerabilities
        allowed_funcs = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs
        }
        
        # ENFORCING ARCHITECTURE: Throw Domain Exception on syntax failure instead of logging
        try:
            code = compile(formula_str, "<string>", "eval")
        except Exception as e:
            raise SimulationError(f"Mathematical syntax error in formula '{formula_str}'.\nDetails: {e}")

        def evaluate_z(x_val: float, y_val: float) -> float:
            """Safely evaluates the pre-compiled AST node for a given (x,y) coordinate."""
            try:
                val = eval(code, {"__builtins__": {}}, {**allowed_funcs, "x": x_val, "y": y_val})
                if isinstance(val, complex) or math.isnan(val) or math.isinf(val):
                    return np.nan
                return float(val)
            except Exception:
                return np.nan 

        # ========================================================
        # PHASE 1: ADAPTIVE SAMPLING VIA GRADIENT ANALYSIS
        # ========================================================
        coarse_res = min(resolution, 60) 
        cx = np.linspace(x_range[0], x_range[1], coarse_res)
        cy = np.linspace(y_range[0], y_range[1], coarse_res)
        
        CZ = np.zeros((coarse_res, coarse_res))
        for i in range(coarse_res):
            for j in range(coarse_res):
                val = evaluate_z(cx[i], cy[j])
                CZ[i, j] = 0.0 if np.isnan(val) else val 

        diff_x = np.abs(np.diff(CZ, axis=0))
        diff_x = np.vstack((diff_x, diff_x[-1:, :])) 
        
        diff_y = np.abs(np.diff(CZ, axis=1))
        diff_y = np.hstack((diff_y, diff_y[:, -1:])) 
        
        max_x = np.max(diff_x)
        max_y = np.max(diff_y)
        weight_x = 1.0 + 15.0 * (diff_x / (max_x if max_x > 0 else 1.0))
        weight_y = 1.0 + 15.0 * (diff_y / (max_y if max_y > 0 else 1.0))
        
        cdf_x = np.cumsum(np.max(weight_x, axis=1))
        cdf_y = np.cumsum(np.max(weight_y, axis=0))
        
        cdf_x = (cdf_x - cdf_x[0]) / (cdf_x[-1] - cdf_x[0])
        cdf_y = (cdf_y - cdf_y[0]) / (cdf_y[-1] - cdf_y[0])
        
        uniform_t = np.linspace(0, 1, resolution)
        x_steps = np.interp(uniform_t, cdf_x, cx)
        y_steps = np.interp(uniform_t, cdf_y, cy)

        # ========================================================
        # PHASE 2: HIGH-RESOLUTION HEIGHTMAP EVALUATION
        # ========================================================
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = evaluate_z(x_steps[i], y_steps[j])
                
        # ========================================================
        # PHASE 3: MESH GENERATION & NORMAL ESTIMATION
        # ========================================================
        for i in range(resolution):
            for j in range(resolution):
                x = x_steps[i]
                y = y_steps[j]
                z = Z[i, j]
                
                if np.isnan(z):
                    vertices.extend([x, 0.0, y, 0.0, 1.0, 0.0, 0.0, 0.0])
                    continue

                x0 = max(0, i - 1)
                x1 = min(resolution - 1, i + 1)
                y0 = max(0, j - 1)
                y1 = min(resolution - 1, j + 1)
                
                dx_forward = x_steps[x1] - x_steps[i] if x1 != i else 1e-6
                dx_backward = x_steps[i] - x_steps[x0] if x0 != i else 1e-6
                dx_center = x_steps[x1] - x_steps[x0] if x1 != x0 else 1e-6
                
                dy_forward = y_steps[y1] - y_steps[j] if y1 != j else 1e-6
                dy_backward = y_steps[j] - y_steps[y0] if y0 != j else 1e-6
                dy_center = y_steps[y1] - y_steps[y0] if y1 != y0 else 1e-6
                
                if x1 != i and not np.isnan(Z[x1, j]) and x0 != i and not np.isnan(Z[x0, j]):
                    dz_dx = (Z[x1, j] - Z[x0, j]) / dx_center
                elif x1 != i and not np.isnan(Z[x1, j]):
                    dz_dx = (Z[x1, j] - z) / dx_forward
                elif x0 != i and not np.isnan(Z[x0, j]):
                    dz_dx = (z - Z[x0, j]) / dx_backward
                else:
                    dz_dx = 0.0
                
                if y1 != j and not np.isnan(Z[i, y1]) and y0 != j and not np.isnan(Z[i, y0]):
                    dz_dy = (Z[i, y1] - Z[i, y0]) / dy_center
                elif y1 != j and not np.isnan(Z[i, y1]):
                    dz_dy = (Z[i, y1] - z) / dy_forward
                elif y0 != j and not np.isnan(Z[i, y0]):
                    dz_dy = (z - Z[i, y0]) / dy_backward
                else:
                    dz_dy = 0.0
                
                norm = np.array([-dz_dx, 1.0, -dz_dy])
                norm_length = np.linalg.norm(norm)
                if norm_length > 0: 
                    norm = norm / norm_length
                
                u = i / (resolution - 1)
                v = j / (resolution - 1)
                vertices.extend([x, z, y, norm[0], norm[1], norm[2], u, v])

        # ========================================================
        # PHASE 4: TOPOLOGICAL TRIANGULATION (NaN Filtering)
        # ========================================================
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                p0 = i * resolution + j
                p1 = p0 + 1
                p2 = p0 + resolution
                p3 = p2 + 1
                
                v0_valid = not np.isnan(Z[i, j])
                v1_valid = not np.isnan(Z[i, j+1])
                v2_valid = not np.isnan(Z[i+1, j])
                v3_valid = not np.isnan(Z[i+1, j+1])
                
                if v0_valid and v1_valid and v2_valid:
                    indices.extend([p0, p1, p2])
                
                if v1_valid and v3_valid and v2_valid:
                    indices.extend([p1, p3, p2])

        super().__init__(vertices, indices)
        self.name = f"f(x,y): {formula_str}"