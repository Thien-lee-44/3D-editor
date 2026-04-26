from PySide6.QtWidgets import (QMainWindow, QDockWidget, QWidget, QHBoxLayout, QVBoxLayout,
                               QRadioButton, QToolBar, QComboBox, QCheckBox, 
                               QLabel, QFrame, QSpacerItem, QSizePolicy, 
                               QColorDialog, QMessageBox, QMenu, QStackedWidget)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QAction
from typing import Any, Optional, List, Dict

from src.app import ctx, AppEvent

from src.app.config import (
    APP_TITLE, 
    DEFAULT_WINDOW_SIZE, 
    DEFAULT_BG_COLOR, 
    RENDER_MODE_COMBINED, 
    RENDER_MODE_FLAT
)

class EditorMainWindow(QMainWindow):
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self.setWindowTitle(APP_TITLE)
        self.resize(*DEFAULT_WINDOW_SIZE)
        
        self.dock_math: Optional[QDockWidget] = None
        self.hierarchy_view: Optional[QWidget] = None
        
        self._all_docks: List[QDockWidget] = []
        self._dock_states: Dict[QDockWidget, bool] = {}

        self.create_menu_bar()
        self.create_toolbar()
        self._setup_shortcuts()
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)

    def _setup_shortcuts(self) -> None:
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self._controller.project_ctrl.undo)
        
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self._controller.project_ctrl.redo)

    def set_central_viewport(self, viewport_widget: QWidget) -> None:
        self.gl_widget = viewport_widget
        
        self.central_container = QFrame()
        self.central_layout = QVBoxLayout(self.central_container)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        
        self.stacked_view = QStackedWidget()
        self.stacked_view.addWidget(self.gl_widget)
        
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.setSpacing(0)
        
        try:
            from src.ui.views.viewport.preview_viewport import PreviewViewportWindow
            self.preview_viewport = PreviewViewportWindow(controller=self._controller)
            self.preview_layout.addWidget(self.preview_viewport, stretch=1)
        except ImportError:
            placeholder = QFrame()
            placeholder.setStyleSheet("background-color: #1a1a1a;")
            self.preview_viewport = placeholder
            self.preview_layout.addWidget(placeholder, stretch=1)
            
        self.stacked_view.addWidget(self.preview_container)
        self.central_layout.addWidget(self.stacked_view)
        self.setCentralWidget(self.central_container)

    def register_dock(self, panel_view: QWidget) -> None:
        title = getattr(panel_view, "PANEL_TITLE", "Panel")
        area = getattr(panel_view, "PANEL_DOCK_AREA", Qt.RightDockWidgetArea)
        
        dock = QDockWidget(title, self)
        dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        dock.setWidget(panel_view)
        self.addDockWidget(area, dock)
        
        self._all_docks.append(dock)
        
        if title == "Hierarchy":
            self.hierarchy_view = panel_view
            self.hierarchy_view.setContextMenuPolicy(Qt.CustomContextMenu)
            self.hierarchy_view.customContextMenuRequested.connect(self.show_context_menu)
            
        elif title == "Math Generator":
            dock.setFeatures(dock.features() | QDockWidget.DockWidgetClosable)
            self.dock_math = dock
            self.dock_math.hide() 

    def _on_mode_changed(self, index: int) -> None:
        self.stacked_view.setCurrentIndex(index)
        is_preview = (index == 1)
        
        if is_preview:
            self._dock_states = {dock: dock.isVisible() for dock in self._all_docks}
            for dock in self._all_docks:
                dock.hide()
                
            self.toolbar_tools.setVisible(False)
            self.toolbar_view.setVisible(False)
            self.toolbar_preview.setVisible(True)
            
            if not getattr(self, '_generator_embedded', False):
                gen_ctrl = getattr(self._controller, 'generator_ctrl', None)
                if not gen_ctrl:
                    from src.ui.controllers.generator_ctrl import GeneratorController
                    self._controller.generator_ctrl = GeneratorController()
                    gen_ctrl = self._controller.generator_ctrl
                
                if gen_ctrl and hasattr(gen_ctrl, 'view'):
                    gen_ctrl.view.setStyleSheet("QWidget { background-color: #2b2b2b; } QGroupBox { border: 1px solid #444; margin-top: 1ex; padding: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
                    self.preview_layout.addWidget(gen_ctrl.view)
                    self._generator_embedded = True

            gen_ctrl = getattr(self._controller, 'generator_ctrl', None)
            if gen_ctrl and hasattr(gen_ctrl, 'refresh_preview_display'):
                gen_ctrl.refresh_preview_display()
        else:
            for dock in self._all_docks:
                if self._dock_states.get(dock, True):
                    dock.show()
                    
            self.toolbar_tools.setVisible(True)
            self.toolbar_view.setVisible(True)
            self.toolbar_preview.setVisible(False)
                
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    def _on_entity_selected(self, entity_id: int) -> None:
        if entity_id < 0:
            return
            
        data = ctx.engine.get_selected_entity_data()
        if not data: return
        
        self.rad_mov.setEnabled(True)
        self.rad_rot.setEnabled(True)
        self.rad_scl.setEnabled(True)
        
        is_light = bool(data.get("light"))
        is_cam = bool(data.get("cam"))
        
        if is_light:
            light_type = data["light"]["type"]
            self.rad_scl.setEnabled(False) 
            
            if light_type == "Directional": 
                self.rad_mov.setEnabled(False) 
                if self.rad_mov.isChecked(): self.rad_rot.setChecked(True)
            elif light_type == "Point": 
                self.rad_rot.setEnabled(False) 
                if self.rad_rot.isChecked(): self.rad_mov.setChecked(True)
                    
        elif is_cam:
            self.rad_scl.setEnabled(False) 

    def create_menu_bar(self) -> None:
        menubar = self.menuBar()
        
        menu_file = menubar.addMenu("File")
        menu_file.addAction("New Project", self._controller.project_ctrl.new_project)
        menu_file.addAction("Open Project...", self._controller.project_ctrl.load_project)
        menu_file.addAction("Save Project...", self._controller.project_ctrl.save_project)
        menu_file.addSeparator()
        menu_file.addAction("Export to .obj", self._controller.project_ctrl.export_obj)
        
        menu_add = menubar.addMenu("Add")
        menu_add.addAction("Empty Group").triggered.connect(self._controller.add_empty_group)
        menu_add.addSeparator()
        
        menu_3d = menu_add.addMenu("3D Primitives")
        for n in ctx.engine.get_3d_primitive_names():
            act = QAction(n, self)
            act.triggered.connect(lambda checked=False, name=n: self._controller.spawn_primitive(name, False))
            menu_3d.addAction(act)
        
        menu_2d = menu_add.addMenu("2D Primitives")
        for n in ctx.engine.get_2d_primitive_names():
            act = QAction(n, self)
            act.setData(n)
            act.triggered.connect(lambda checked=False, name=n: self._controller.spawn_primitive(name, True))
            menu_2d.addAction(act)
        
        menu_add.addSeparator()
        menu_add.addAction("3D Math Surface").triggered.connect(
            lambda: (self.dock_math.show(), self.dock_math.raise_()) if self.dock_math else None
        )
        menu_add.addSeparator()
        
        menu_light = menu_add.addMenu("Lights")
        for label, ltype in [("Point Light", "Point"), ("Spot Light", "Spot"), ("Directional Light (Sun)", "Directional")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, lt=ltype: self._controller.add_light(
                lt, self.chk_global_proxy.isChecked(), self.chk_global_light.isChecked()
            ))
            menu_light.addAction(act)
            
        menu_add.addSeparator()
        menu_add.addAction("Camera").triggered.connect(lambda: self._controller.add_camera(self.chk_global_proxy.isChecked()))

        menu_settings = menubar.addMenu("Settings")
        menu_settings.addAction("Background Color", self.action_change_bg_color)

    def create_toolbar(self) -> None:
        self.toolbar_workspace = QToolBar("Workspace")
        self.toolbar_workspace.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_workspace)
        
        self.toolbar_workspace.addWidget(QLabel(" Workspace: "))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["3D Edit Mode", "Synthetic Data Preview"])
        self.mode_selector.setFixedWidth(160)
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        self.toolbar_workspace.addWidget(self.mode_selector)

        self.toolbar_tools = QToolBar("Transform Tools")
        self.toolbar_tools.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_tools)

        self.container_transform_tools = QWidget()
        lay_tools = QHBoxLayout(self.container_transform_tools)
        lay_tools.setContentsMargins(10, 2, 20, 2)
        lay_tools.addWidget(QLabel("Tools:"))
        
        self.rad_mov = QRadioButton("Move")
        self.rad_mov.setChecked(True)
        self.rad_rot = QRadioButton("Rotate")
        self.rad_scl = QRadioButton("Scale")
        
        for r in [self.rad_mov, self.rad_rot, self.rad_scl]: 
            r.toggled.connect(self._on_tool_changed)
            lay_tools.addWidget(r)
            
        self.toolbar_tools.addWidget(self.container_transform_tools)

        self.toolbar_view = QToolBar("View Settings")
        self.toolbar_view.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_view)
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.chk_wire = QCheckBox("Wireframe")
        self.chk_wire.stateChanged.connect(self._on_render_settings_changed)
        
        self.chk_global_proxy = QCheckBox("Proxies")
        self.chk_global_proxy.setChecked(True)
        self.chk_global_proxy.stateChanged.connect(lambda s: self._controller.toggle_all_proxies(s != 0))
        
        self.chk_global_light = QCheckBox("Lights")
        self.chk_global_light.setChecked(True)
        self.chk_global_light.stateChanged.connect(lambda s: self._controller.toggle_all_lights(s != 0))
        
        layout.addWidget(QLabel("Display: "))
        layout.addWidget(self.chk_wire)
        layout.addWidget(self.chk_global_proxy)
        layout.addWidget(self.chk_global_light)
        
        self.cmb_render = QComboBox()
        self.cmb_render.addItems(["Flat Color", "Combined"])
        self.cmb_render.setCurrentIndex(1)
        self.cmb_render.currentIndexChanged.connect(self._on_render_settings_changed)
        layout.addWidget(QLabel(" | Render Mode: "))
        layout.addWidget(self.cmb_render)
        
        self.w_comb_opts = QWidget()
        lay_comb = QHBoxLayout(self.w_comb_opts)
        lay_comb.setContentsMargins(0, 0, 0, 0)
        self.chk_comb_light = QCheckBox("+Lighting")
        self.chk_comb_light.setChecked(True)
        self.chk_comb_tex = QCheckBox("+Textures")
        self.chk_comb_tex.setChecked(True)
        self.chk_comb_vcolor = QCheckBox("+Vertex Colors")
        self.chk_comb_vcolor.setChecked(True)
        
        for c in [self.chk_comb_light, self.chk_comb_tex, self.chk_comb_vcolor]:
            c.stateChanged.connect(self._on_render_settings_changed)
            lay_comb.addWidget(c)
        layout.addWidget(self.w_comb_opts)
        
        self.cmb_output = QComboBox()
        self.cmb_output.addItems(["Color (RGB)", "Depth Map"])
        self.cmb_output.currentIndexChanged.connect(self._on_render_settings_changed)
        layout.addWidget(QLabel(" | Output: "))
        layout.addWidget(self.cmb_output)

        self.toolbar_view.addWidget(container)

        # =========================================================
        # PREVIEW TOOLBAR (Native Main Window Integration)
        # =========================================================
        self.toolbar_preview = QToolBar("Preview Tools")
        self.toolbar_preview.setMovable(False)
        self.toolbar_preview.setStyleSheet("QToolBar { background-color: #252525; border-bottom: 1px solid #111; }")
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_preview)
        
        self.lbl_preview_time = QLabel("Time: 0.00s")
        self.lbl_preview_time.setStyleSheet("color: #4CA8FE; font-family: Consolas; font-weight: bold; margin-left: 10px; margin-right: 15px;")
        self.toolbar_preview.addWidget(self.lbl_preview_time)

        self.lbl_preview_status = QLabel("Status: Idle")
        self.lbl_preview_status.setStyleSheet("color: #AAAAAA; font-family: Consolas;")
        self.toolbar_preview.addWidget(self.lbl_preview_status)

        spacer1 = QWidget()
        spacer1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar_preview.addWidget(spacer1)
        
        self.toolbar_preview.addWidget(QLabel("Mode: "))
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["RGB", "SEMANTIC", "INSTANCE", "DEPTH"])
        self.cmb_preview_mode.currentTextChanged.connect(self._on_preview_mode_changed)
        self.toolbar_preview.addWidget(self.cmb_preview_mode)
        
        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar_preview.addWidget(spacer2)

        self.chk_preview_bbox = QCheckBox("BBox")
        self.chk_preview_bbox.setChecked(True)
        self.chk_preview_bbox.stateChanged.connect(self._on_preview_overlay_toggled)
        self.toolbar_preview.addWidget(self.chk_preview_bbox)
        
        self.lbl_preview_stats = QLabel("Obj: 0 | Occ: 0.0")
        self.lbl_preview_stats.setStyleSheet("color: #aaa; font-family: Consolas; margin-left: 15px; margin-right: 10px;")
        self.toolbar_preview.addWidget(self.lbl_preview_stats)
        
        self.toolbar_preview.setVisible(False)

    # =========================================================================
    # PREVIEW DELEGATES
    # =========================================================================
    
    def _on_preview_mode_changed(self, mode: str) -> None:
        gen_ctrl = getattr(self._controller, 'generator_ctrl', None)
        if gen_ctrl and hasattr(gen_ctrl, 'refresh_preview_display'):
            gen_ctrl.refresh_preview_display()

    def _on_preview_overlay_toggled(self, state: int) -> None:
        pv = getattr(self, 'preview_viewport', None)
        if pv and pv.display_frame.payload:
            payload_copy = pv.display_frame.payload.copy()
            pv.update_frame(payload_copy)

    # =========================================================================
    # EDITOR DELEGATES 
    # =========================================================================

    def _on_tool_changed(self) -> None:
        if self.rad_rot.isChecked(): self._controller.set_manipulation_mode("ROTATE")
        elif self.rad_mov.isChecked(): self._controller.set_manipulation_mode("MOVE")
        elif self.rad_scl.isChecked(): self._controller.set_manipulation_mode("SCALE")

    def _on_render_settings_changed(self, *args: Any) -> None:
        is_combined = (self.cmb_render.currentIndex() == 1)
        self.w_comb_opts.setVisible(is_combined)
        
        actual_render_mode = RENDER_MODE_COMBINED if is_combined else RENDER_MODE_FLAT
        
        self._controller.set_render_settings(
            wireframe=self.chk_wire.isChecked(), 
            mode=actual_render_mode, 
            output=self.cmb_output.currentIndex(), 
            light=self.chk_comb_light.isChecked(), 
            tex=self.chk_comb_tex.isChecked(), 
            vcolor=self.chk_comb_vcolor.isChecked()
        )

    def show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        has_selection = (ctx.engine.get_selected_entity_id() >= 0)
        
        can_toggle_vis = has_selection
        if has_selection:
            data = ctx.engine.get_selected_entity_data()
            if data and data.get("light") and data["light"].get("type") == "Directional":
                can_toggle_vis = False
        
        menu.addAction("Copy", self.action_copy)
        menu.addAction("Cut", self.action_cut)
        
        act_paste = QAction("Paste", self)
        act_paste.setEnabled(ctx.engine.has_clipboard())
        act_paste.triggered.connect(self.action_paste)
        menu.addAction(act_paste)
        
        menu.addSeparator()
        act_vis = QAction("Toggle Visibility", self)
        act_vis.setEnabled(can_toggle_vis)
        act_vis.triggered.connect(self.action_toggle_visibility)
        menu.addAction(act_vis)
        
        menu.addSeparator()
        menu.addAction("Delete", self.action_delete).setEnabled(has_selection)
        
        if hasattr(self, 'hierarchy_view') and self.sender() == self.hierarchy_view:
            menu.exec(self.hierarchy_view.mapToGlobal(pos))
        else:
            menu.exec(pos)

    def action_copy(self) -> None: self._controller.copy_selected()
    def action_cut(self) -> None: self._controller.cut_selected()
    def action_paste(self) -> None: self._controller.paste_copied()
    def action_delete(self) -> None: self._controller.delete_selected()
    def action_toggle_visibility(self) -> None: self._controller.toggle_visibility_selected()

    def action_change_bg_color(self) -> None:
        current_bg = getattr(self.gl_widget, 'bg_color', DEFAULT_BG_COLOR)
        color = QColorDialog.getColor(
            QColor.fromRgbF(current_bg[0], current_bg[1], current_bg[2]), 
            self, "Select Background Color"
        )
        if color.isValid():
            self.gl_widget.bg_color = (color.redF(), color.greenF(), color.blueF())
            self.gl_widget.update()