"""
Main Controller.
Root Orchestrator initializing all sub-modules and mapping top-level UI logic.
"""

from PySide6.QtCore import QTimer
from src.app import ctx, AppEvent, config
from src.ui.error_handler import safe_execute

from src.ui.main_window import EditorMainWindow
from src.ui.controllers.project_ctrl import ProjectController
from src.ui.controllers.viewport_ctrl import ViewportController
from src.ui.controllers.hierarchy_ctrl import HierarchyController
from src.ui.controllers.inspector_ctrl import InspectorController
from src.ui.controllers.asset_ctrl import AssetController
from src.ui.controllers.math_gen_ctrl import MathGenController


class MainController:
    """
    Root Orchestrator.
    Initializes all sub-modules and manages Top Menu / Toolbar logic.
    """
    def __init__(self) -> None:
        self.project_ctrl = ProjectController()
        self.viewport_ctrl = ViewportController()
        self.hierarchy_ctrl = HierarchyController()
        self.inspector_ctrl = InspectorController()
        self.asset_ctrl = AssetController()
        self.math_gen_ctrl = MathGenController()

        self.main_window = EditorMainWindow(controller=self)
        ctx.main_window = self.main_window 

        self.main_window.register_dock(self.hierarchy_ctrl.view)
        self.main_window.register_dock(self.inspector_ctrl.view)
        self.main_window.register_dock(self.asset_ctrl.view)
        self.main_window.register_dock(self.math_gen_ctrl.view)
        
        self.main_window.set_central_viewport(self.viewport_ctrl.view)

        # Setup continuous input polling based on TARGET_FPS
        self.input_timer = QTimer()
        poll_interval = int(1000 / config.TARGET_FPS) 
        self.input_timer.timeout.connect(self._poll_continuous_input)
        self.input_timer.start(poll_interval) 

        ctx.events.subscribe(AppEvent.SCENE_CHANGED, self.main_window.gl_widget.update)

    @safe_execute(context="Camera Input Update")
    def _poll_continuous_input(self) -> None:
        """Polls the viewport controller for continuous input (e.g., WASD camera movement)."""
        if self.viewport_ctrl.process_continuous_input():
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Empty Group")
    def add_empty_group(self) -> None:
        """Instantiates an empty transformation node."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.add_empty_group()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Group Entities")
    def group_selected(self) -> None:
        """Wraps the actively selected multi-item collection into a new parent group."""
        ids = self.hierarchy_ctrl.selected_multi_ids
        if len(ids) > 1:
            ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
            ctx.engine.group_selected_entities(ids)
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Ungroup Entity")
    def ungroup_selected(self) -> None:
        """Dissolves the currently selected group, reparenting its children."""
        idx = ctx.engine.get_selected_entity_id()
        if idx >= 0:
            ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
            ctx.engine.ungroup_selected_entity()
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Spawn Primitive")
    def spawn_primitive(self, name: str, is_2d: bool) -> None:
        """Instantiates an entity equipped with a standard geometric mesh."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.spawn_primitive(name, is_2d)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.spawn_primitive(name, is_2d)
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Light")
    def add_light(self, light_type: str, proxy_enabled: bool, global_light_on: bool) -> None:
        """Instantiates a lighting entity (Directional, Point, Spot)."""
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.add_light(light_type, proxy_enabled, global_light_on)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.add_light(light_type, proxy_enabled, global_light_on)
            
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Camera")
    def add_camera(self, proxy_enabled: bool) -> None:
        """Instantiates an auxiliary viewpoint camera entity."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.add_camera(proxy_enabled)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.add_camera(proxy_enabled)
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Copy Entity")
    def copy_selected(self) -> None: 
        """Copies the selected entity to the clipboard."""
        ctx.engine.copy_selected()
        
    @safe_execute(context="Cut Entity")
    def cut_selected(self) -> None: 
        """Cuts the selected entity to the clipboard, triggering a state mutation."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.cut_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        
    @safe_execute(context="Paste Entity")
    def paste_copied(self) -> None: 
        """Pastes the clipboard contents into the active scene graph."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.paste_copied()
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.paste_copied()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)
            
    @safe_execute(context="Delete Entity")
    def delete_selected(self) -> None: 
        """Deletes the selected entity and its descendants from the scene graph."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.delete_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle Visibility")
    def toggle_visibility_selected(self) -> None:
        """Toggles the rendering state of the currently selected entity."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.toggle_visibility_selected()
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Update Render Settings")
    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        """Updates the global Forward Renderer pipeline flags."""
        ctx.engine.set_render_settings(wireframe, mode, output, light, tex, vcolor)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Change Manipulation Mode")
    def set_manipulation_mode(self, mode: str) -> None:
        """Switches the active Gizmo tool (Translate, Rotate, Scale)."""
        ctx.engine.set_manipulation_mode(mode)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle All Lights")
    def toggle_all_lights(self, state: bool) -> None:
        """Globally overrides the emission state of all lighting entities."""
        ctx.engine.toggle_all_lights(state)
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED) 
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle All Proxies")
    def toggle_all_proxies(self, state: bool) -> None:
        """Globally toggles the rendering of unlit Editor Proxy meshes."""
        ctx.engine.toggle_all_proxies(state)
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)