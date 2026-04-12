"""
Centralized Configuration and System Constants.
Acts as the Single Source of Truth (SSOT) to eliminate magic numbers, 
hardcoded strings, and disjointed limits across the Engine and UI modules.
"""

from pathlib import Path
from typing import Dict, Tuple

# =========================================================================
# 1. FILE SYSTEM PATHS
# Uses pathlib to resolve absolute paths dynamically based on the file location.
# =========================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

ASSETS_DIR: Path = BASE_DIR / "assets"
MODELS_DIR: Path = ASSETS_DIR / "models"
SHADERS_DIR: Path = ASSETS_DIR / "shaders"
TEXTURES_DIR: Path = ASSETS_DIR / "textures"
PROJECTS_DIR: Path = BASE_DIR / "projects"

# =========================================================================
# 2. APP & WORKSPACE SETTINGS
# =========================================================================
APP_TITLE: str = "3D Scene Editor & Synthetic Data Engine"
DEFAULT_WINDOW_SIZE: Tuple[int, int] = (1280, 720)

PROJECT_MANAGER_TITLE: str = "Project Manager"
PROJECT_MANAGER_MIN_SIZE: Tuple[int, int] = (400, 300)
MAX_UNDO_HISTORY: int = 30
DEFAULT_EXPORT_FOLDER: str = "Exported_Scene"

# =========================================================================
# 3. CORE RENDER & DISPLAY DEFAULTS
# =========================================================================
TARGET_FPS: int = 60
MSAA_SAMPLES: int = 8
DEFAULT_BG_COLOR: Tuple[float, float, float] = (0.15, 0.15, 0.15)
DEFAULT_ASPECT_RATIO: float = 16.0 / 9.0

RENDER_MODE_FLAT: int = 0
RENDER_MODE_COMBINED: int = 4

# =========================================================================
# 4. SCENE & CAMERA DEFAULTS
# =========================================================================
DEFAULT_CAMERA_NAME: str = "Main Camera"
DEFAULT_CAMERA_FOV: float = 45.0
DEFAULT_CAMERA_NEAR: float = 0.1
DEFAULT_CAMERA_FAR: float = 1000.0
DEFAULT_SCENE_CAM_POS: Tuple[float, float, float] = (0.0, 0.0, 5.0)

CAMERA_MOVE_SPEED: float = 5.0
CAMERA_ROTATION_SPEED: float = 80.0

# =========================================================================
# 5. LIGHTING & HARDWARE LIMITS
# =========================================================================
MAX_LIGHTS: Dict[str, int] = {
    "Directional": 8,
    "Point": 16,
    "Spot": 8
}

DEFAULT_LIGHT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_LIGHT_INTENSITY: float = 1.0
DEFAULT_LIGHT_AMBIENT: float = 0.2
DEFAULT_LIGHT_DIFFUSE: float = 1.0
DEFAULT_LIGHT_SPECULAR: float = 1.0

# Attenuation (Standard for ~50m range)
DEFAULT_LIGHT_CONSTANT: float = 1.0
DEFAULT_LIGHT_LINEAR: float = 0.09
DEFAULT_LIGHT_QUADRATIC: float = 0.032

# Spotlight Angles (Degrees)
DEFAULT_SPOT_INNER_ANGLE: float = 12.5
DEFAULT_SPOT_OUTER_ANGLE: float = 15.0
DEFAULT_SCENE_LIGHT_ROT: Tuple[float, float, float] = (-45.0, -45.0, 0.0)

# =========================================================================
# 6. ENTITY & TRANSFORM DEFAULTS
# =========================================================================
DEFAULT_ENTITY_NAME: str = "New Entity"
DEFAULT_GROUP_NAME: str = "New Group"

DEFAULT_SPAWN_POSITION: Tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_SPAWN_ROTATION: Tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_SPAWN_SCALE: Tuple[float, float, float] = (1.0, 1.0, 1.0)
PASTE_OFFSET: Tuple[float, float, float] = (0.5, 0.0, 0.5)

# =========================================================================
# 7. MATERIAL & MESH DEFAULTS
# =========================================================================
DEFAULT_MAT_AMBIENT: Tuple[float, float, float] = (0.2, 0.2, 0.2)
DEFAULT_MAT_DIFFUSE: Tuple[float, float, float] = (0.8, 0.8, 0.8)
DEFAULT_MAT_SPECULAR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_MAT_EMISSION: Tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_MAT_BASE_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)

DEFAULT_MAT_SHININESS: float = 32.0
DEFAULT_MAT_OPACITY: float = 1.0

DEFAULT_MAT_AMB_STRENGTH: float = 0.5
DEFAULT_MAT_DIFF_STRENGTH: float = 1.0
DEFAULT_MAT_SPEC_STRENGTH: float = 1.0

# =========================================================================
# 8. EDITOR GIZMO & HUD VISUALS
# =========================================================================
DEFAULT_MANIPULATION_MODE: str = "MOVE"
DEFAULT_PROXY_SCALE: float = 0.2
DEFAULT_POINT_SIZE: float = 0.2
GIZMO_RING_SEGMENTS: int = 64

# Axis Colors
GIZMO_COLOR_X: Tuple[float, float, float] = (1.0, 0.2, 0.2)     # Red
GIZMO_COLOR_Y: Tuple[float, float, float] = (0.2, 1.0, 0.2)     # Green
GIZMO_COLOR_Z: Tuple[float, float, float] = (0.2, 0.5, 1.0)     # Blue
GIZMO_COLOR_HOVER: Tuple[float, float, float] = (1.0, 1.0, 0.0) # Yellow
GIZMO_COLOR_CORE: Tuple[float, float, float] = (0.6, 0.6, 0.6)  # Grey

# HUD Compass Viewport
HUD_COMPASS_SIZE: int = 100
HUD_COMPASS_OFFSET: int = 130
HUD_AXIS_PADDING: int = 80
HUD_AXIS_SCALE: int = 50
SUN_HUD_MIN_HEIGHT: int = 200

# =========================================================================
# 9. PROCEDURAL MATH SURFACE DEFAULTS
# =========================================================================
DEFAULT_MATH_FORMULA: str = "sin(x) * cos(y)"
DEFAULT_MATH_RANGE: Tuple[float, float] = (-5.0, 5.0)
DEFAULT_MATH_RESOLUTION: int = 50

MATH_LIMIT_MIN: float = -100.0
MATH_LIMIT_MAX: float = 100.0
MATH_RES_MIN: int = 10
MATH_RES_MAX: int = 500

# =========================================================================
# 10. DOMAIN MAPPINGS (UI <-> ENGINE <-> EXPORT)
# =========================================================================
TEXTURE_CHANNELS: Dict[str, str] = {
    "Diffuse Map": "map_diffuse",
    "Specular Map": "map_specular",
    "Emission Map": "map_emission",
    "Ambient (AO) Map": "map_ambient",
    "Shininess Map": "map_shininess",
    "Opacity (Alpha) Map": "map_opacity",
    "Bump / Normal Map": "map_bump",
    "Reflection Map": "map_reflection"
}

MTL_TOKENS: Dict[str, str] = {
    "map_diffuse": "map_Kd",
    "map_specular": "map_Ks",
    "map_ambient": "map_Ka",
    "map_emission": "map_Ke",
    "map_bump": "map_Bump",
    "map_opacity": "map_d",
    "map_shininess": "map_Ns",
    "map_reflection": "map_refl"
}

# =========================================================================
# 11. UI WIDGET LIMITS & STYLES
# =========================================================================
DEFAULT_UI_MARGIN: int = 5
DEFAULT_UI_SPACING: int = 5

PANEL_TITLE_UNKNOWN: str = "Unknown Panel"
PANEL_TITLE_ASSET: str = "Asset Browser"
PANEL_TITLE_HIERARCHY: str = "Hierarchy"
PANEL_TITLE_INSPECTOR: str = "Inspector"
PANEL_TITLE_MATH_GEN: str = "Math Generator"

PANEL_MIN_WIDTH_DEFAULT: int = 250
PANEL_MIN_WIDTH_INSPECTOR: int = 380
PANEL_CONTENT_MIN_WIDTH: int = 320

ASSET_ICON_SIZE: int = 48
ASSET_LIST_SPACING: int = 2
TEX_THUMB_SIZE: int = 32

GLOBAL_NUMERIC_MIN: float = -10000.0
GLOBAL_NUMERIC_MAX: float = 10000.0

LIGHT_INTENSITY_RANGE: Tuple[float, float] = (0.0, 50.0)
LIGHT_FACTOR_RANGE: Tuple[float, float] = (0.0, 1.0)
LIGHT_YAW_RANGE: Tuple[float, float] = (0.0, 360.0)
LIGHT_PITCH_RANGE: Tuple[float, float] = (-90.0, 90.0)
LIGHT_CUTOFF_RANGE: Tuple[float, float] = (0.0, 90.0)

LIGHT_ATTEN_CONST_RANGE: Tuple[float, float] = (0.0, 10.0)
LIGHT_ATTEN_LIN_RANGE: Tuple[float, float] = (0.0, 1.0)
LIGHT_ATTEN_QUAD_RANGE: Tuple[float, float] = (0.0, 1.0)

MAT_FACTOR_RANGE: Tuple[float, float] = (0.0, 1.0)
MAT_SHININESS_RANGE: Tuple[float, float] = (0.1, 1024.0)
MAT_OPACITY_RANGE: Tuple[float, float] = (0.0, 1.0)

TRANSFORM_POS_STEP: float = 0.1
TRANSFORM_ROT_RANGE: Tuple[float, float] = (-360.0, 360.0)
TRANSFORM_ROT_STEP: float = 1.0
TRANSFORM_SCL_MIN: float = 0.01
TRANSFORM_SCL_STEP: float = 0.01

CAMERA_FOV_RANGE: Tuple[float, float] = (1.0, 179.0)
CAMERA_FOV_STEP: float = 1.0
CAMERA_ORTHO_RANGE: Tuple[float, float] = (0.1, 1000.0)
CAMERA_ORTHO_STEP: float = 0.1
CAMERA_NEAR_RANGE: Tuple[float, float] = (0.001, 1000.0)
CAMERA_NEAR_STEP: float = 0.1
CAMERA_FAR_RANGE: Tuple[float, float] = (0.1, 10000.0)
CAMERA_FAR_STEP: float = 1.0

COLOR_VEC_RANGE: Tuple[float, float] = (0.0, 100.0)
COLOR_VEC_STEP: float = 0.01

# --- Stylesheets ---
VIEWPORT_HUD_STYLE: str = "color: white; font-weight: bold; background: transparent;"
CONTEXT_MENU_STYLE: str = "QMenu { font-size: 13px; padding: 5px; } QMenu::item { padding: 5px 20px; }"
STYLE_TEX_THUMB: str = "background-color: #222; border: 1px solid #555;"
STYLE_TEX_EMPTY: str = "color: gray; font-style: italic;"
STYLE_TEX_LOADED: str = "color: #4CAF50; font-weight: bold;"

STYLE_BTN_RESET: str = "background-color: #f0ad4e; color: black; font-weight: bold; padding: 4px;"
STYLE_COLOR_BTN_DARK_TEXT: str = "color: black; font-weight: bold; border-radius: 4px; border: 1px solid #666;"
STYLE_COLOR_BTN_LIGHT_TEXT: str = "color: white; font-weight: bold; border-radius: 4px; border: 1px solid #666;"
STYLE_BTN_ACTIVE_CAM: str = "background-color: #28a745; color: white; font-weight: bold;"
STYLE_BTN_INACTIVE_CAM: str = "background-color: #007bff; color: white; font-weight: bold;"