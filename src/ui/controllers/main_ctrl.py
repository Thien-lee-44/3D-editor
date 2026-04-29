"""
Main Application Controller.

Serves as the Root Orchestrator for the Editor framework.
Initializes all sub-modules, manages Top Menu / Toolbar interactions, 
and acts as the central hub connecting UI events to Engine execution.
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
from src.ui.controllers.generator_ctrl import GeneratorController 
from src.ui.controllers.timeline_ctrl import TimelineController


class MainController:
    """The foundational entry point for mapping application controllers to the UI."""

    def __init__(self) -> None:
        self.project_ctrl = ProjectController()
        self.viewport_ctrl = ViewportController()
        self.hierarchy_ctrl = HierarchyController()
        self.inspector_ctrl = InspectorController()
        self.asset_ctrl = AssetController()
        self.math_gen_ctrl = MathGenController()
        self.generator_ctrl = GeneratorController()
        self.timeline_ctrl = TimelineController()
        
        self.main_window = EditorMainWindow(controller=self)
        ctx.main_window = self.main_window 

        self.main_window.register_dock(self.hierarchy_ctrl.view)
        self.main_window.register_dock(self.inspector_ctrl.view)
        self.main_window.register_dock(self.asset_ctrl.view)
        self.main_window.register_dock(self.math_gen_ctrl.view)
        self.main_window.register_dock(self.timeline_ctrl.view) 
        
        self.main_window.set_central_viewport(self.viewport_ctrl.view)

        # Global rendering heartbeat timer
        self.input_timer = QTimer()
        poll_interval = int(1000 / config.TARGET_FPS) 
        self.input_timer.timeout.connect(self._poll_continuous_input)
        self.input_timer.start(poll_interval) 

        # Wire hardware re-render to the global scene mutation event
        ctx.events.subscribe(AppEvent.SCENE_CHANGED, self.main_window.gl_widget.update)

    @safe_execute(context="Tick Update")
    def _poll_continuous_input(self) -> None:
        """Checks for active inputs and steps the animation timeline forward."""
        dt = 1.0 / config.TARGET_FPS
        self.timeline_ctrl.advance_time(dt)
        
        if self.viewport_ctrl.process_continuous_input():
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    # =========================================================================
    # ENTITY CREATION & MUTATION API
    # =========================================================================

    @safe_execute(context="Add Empty Group")
    def add_empty_group(self) -> None:
        """Commands the Engine to instantiate a logical grouping entity."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.add_empty_group()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Group Entities")
    def group_selected(self) -> None:
        """Triggers the grouping math logic if multiple objects are selected."""
        ids = self.hierarchy_ctrl.selected_multi_ids
        if len(ids) > 1:
            ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
            ctx.engine.group_selected_entities(ids)
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Ungroup Entity")
    def ungroup_selected(self) -> None:
        """Dissolves the selected group and elevates its children up the hierarchy."""
        idx = ctx.engine.get_selected_entity_id()
        if idx >= 0:
            ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
            ctx.engine.ungroup_selected_entity()
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Spawn Primitive")
    def spawn_primitive(self, name: str, is_2d: bool) -> None:
        """Instantiates geometric primitives (e.g. Cube, Sphere) directly into the Scene."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        # Centralized Context Binding implementation
        has_gl = hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget')
        if has_gl: 
            ctx.main_window.gl_widget.makeCurrent()
            
        try:
            ctx.engine.spawn_primitive(name, is_2d)
        finally:
            if has_gl: 
                ctx.main_window.gl_widget.doneCurrent()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Light")
    def add_light(self, light_type: str, proxy_enabled: bool, global_light_on: bool) -> None:
        """Spawns an illumination source while enforcing maximum hardware limits."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)

        has_gl = hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget')
        if has_gl: 
            ctx.main_window.gl_widget.makeCurrent()
            
        try:
            ctx.engine.add_light(light_type, proxy_enabled, global_light_on)
        finally:
            if has_gl: 
                ctx.main_window.gl_widget.doneCurrent()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Camera")
    def add_camera(self, proxy_enabled: bool) -> None:
        """Places a new viewpoint entity into the workspace."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)

        has_gl = hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget')
        if has_gl: 
            ctx.main_window.gl_widget.makeCurrent()
            
        try:
            ctx.engine.add_camera(proxy_enabled)
        finally:
            if has_gl: 
                ctx.main_window.gl_widget.doneCurrent()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Copy Entity")
    def copy_selected(self) -> None: 
        ctx.engine.copy_selected()
        
    @safe_execute(context="Cut Entity")
    def cut_selected(self) -> None: 
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.cut_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        
    @safe_execute(context="Paste Entity")
    def paste_copied(self) -> None: 
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        has_gl = hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget')
        if has_gl: 
            ctx.main_window.gl_widget.makeCurrent()
            
        try:
            ctx.engine.paste_copied()
        finally:
            if has_gl: 
                ctx.main_window.gl_widget.doneCurrent()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)
            
    @safe_execute(context="Delete Entity")
    def delete_selected(self) -> None: 
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.delete_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle Visibility")
    def toggle_visibility_selected(self) -> None:
        """Inverts the render state (hide/unhide) of the active selection."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.toggle_visibility_selected()
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    # =========================================================================
    # RENDER & MANIPULATION SETTINGS
    # =========================================================================

    @safe_execute(context="Update Render Settings")
    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        """Passes environment rendering configurations down to the OpenGL Pipeline."""
        ctx.engine.set_render_settings(wireframe, mode, output, light, tex, vcolor)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Change Manipulation Mode")
    def set_manipulation_mode(self, mode: str) -> None:
        """Toggles the state of the active viewport Gizmo (Translate, Rotate, Scale)."""
        ctx.engine.set_manipulation_mode(mode)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle All Lights")
    def toggle_all_lights(self, state: bool) -> None:
        """Globally powers on or shuts down all illumination sources."""
        ctx.engine.toggle_all_lights(state)
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED) 
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Toggle All Proxies")
    def toggle_all_proxies(self, state: bool) -> None:
        """Hides or reveals the unlit icons representing Light/Camera entities."""
        ctx.engine.toggle_all_proxies(state)
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)