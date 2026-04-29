"""
Main Editor Window.
Assembles all dockable panels, viewport canvas, toolbars, and menus into the 
primary application interface.
"""

from PySide6.QtWidgets import (QMainWindow, QDockWidget, QWidget, QHBoxLayout, 
                               QRadioButton, QToolBar, QComboBox, QCheckBox, 
                               QLabel, QSpacerItem, QSizePolicy, 
                               QColorDialog, QMessageBox, QMenu)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QAction
from typing import Any, Optional

from src.app import ctx, AppEvent

# Import SSOT configuration
from src.app.config import (
    APP_TITLE, DEFAULT_WINDOW_SIZE, DEFAULT_BG_COLOR, 
    RENDER_MODE_COMBINED, RENDER_MODE_FLAT
)


class EditorMainWindow(QMainWindow):
    """
    Shell interface of the application.
    Assembles Docks, ToolBars, MenuBars and binds buttons to the MainController.
    """
    
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self.setWindowTitle(APP_TITLE)
        self.resize(*DEFAULT_WINDOW_SIZE)

        self.dock_math: Optional[QDockWidget] = None
        self.hierarchy_view: Optional[QWidget] = None

        self.create_menu_bar()
        self.create_toolbar()
        self._setup_shortcuts()
        
        # Listen to object selection event to lock Scale/Rotate buttons for Proxies
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)

    def _setup_shortcuts(self) -> None:
        """Configures global keyboard shortcuts."""
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self._controller.project_ctrl.undo)

        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self._controller.project_ctrl.redo)

    def set_central_viewport(self, viewport_widget: QWidget) -> None:
        """Assigns the 3D OpenGL viewport as the central widget."""
        self.gl_widget = viewport_widget
        self.setCentralWidget(self.gl_widget)

    def register_dock(self, panel_view: QWidget) -> None:
        """Wraps a panel view in a QDockWidget and docks it to the main window."""
        title = getattr(panel_view, "PANEL_TITLE", "Panel")
        area = getattr(panel_view, "PANEL_DOCK_AREA", Qt.RightDockWidgetArea)
        
        dock = QDockWidget(title, self)
        dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        dock.setWidget(panel_view)
        self.addDockWidget(area, dock)
        
        if title == "Hierarchy":
            self.hierarchy_view = panel_view
            self.hierarchy_view.setContextMenuPolicy(Qt.CustomContextMenu)
            self.hierarchy_view.customContextMenuRequested.connect(self.show_context_menu)
            
        elif title == "Math Generator":
            dock.setFeatures(dock.features() | QDockWidget.DockWidgetClosable)
            self.dock_math = dock
            self.dock_math.hide() 

    # =========================================================================
    # CATCH EVENTS FROM ENGINE -> LOCK PROXY UI BUTTONS
    # =========================================================================

    def _on_entity_selected(self, entity_id: int) -> None:
        """Checks the object type and locks the corresponding Transform tools."""
        if entity_id < 0:
            return
            
        data = ctx.engine.get_selected_entity_data()
        if not data: 
            return
        
        # Unlock all
        self.rad_mov.setEnabled(True)
        self.rad_rot.setEnabled(True)
        self.rad_scl.setEnabled(True)
        
        has_light = bool(data.get("light"))
        has_cam = bool(data.get("cam"))
        
        # Lock buttons based on light type (Proxy constraint)
        if has_light:
            light_type = data["light"]["type"]
            self.rad_scl.setEnabled(False) # Disable scaling for lights
            if self.rad_scl.isChecked(): 
                self.rad_mov.setChecked(True)
                
            if light_type == "Directional": 
                self.rad_mov.setEnabled(False) # Disable moving for the sun (only allow rotation)
                if self.rad_mov.isChecked(): 
                    self.rad_rot.setChecked(True)
                    
            elif light_type == "Point": 
                self.rad_rot.setEnabled(False) # Disable rotation for point lights (only allow movement)
                if self.rad_rot.isChecked(): 
                    self.rad_mov.setChecked(True)
                    
        # Lock buttons for Camera (Scaling is not allowed)
        elif has_cam:
            self.rad_scl.setEnabled(False) # Disable scaling for cameras
            if self.rad_scl.isChecked(): 
                self.rad_mov.setChecked(True)

    # =========================================================================
    # SETUP MENUBAR & TOOLBAR UI
    # =========================================================================

    def create_menu_bar(self) -> None:
        """Initializes the top application menu bar."""
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
            act.setData(n) 
            act.triggered.connect(self._on_spawn_primitive_3d)
            menu_3d.addAction(act)
            
        menu_2d = menu_add.addMenu("2D Primitives")
        for n in ctx.engine.get_2d_primitive_names():
            act = QAction(n, self)
            act.setData(n)
            act.triggered.connect(self._on_spawn_primitive_2d)
            menu_2d.addAction(act)
            
        menu_add.addSeparator()
        menu_add.addAction("3D Math Surface").triggered.connect(
            lambda: (self.dock_math.show(), self.dock_math.raise_()) if self.dock_math else None
        )
        menu_add.addSeparator()
        
        menu_light = menu_add.addMenu("Lights")
        for label, ltype in [("Point Light", "Point"), ("Spot Light", "Spot"), ("Directional Light (Sun)", "Directional")]:
            act = QAction(label, self)
            act.setData(ltype)
            act.triggered.connect(self._on_add_light)
            menu_light.addAction(act)
            
        menu_add.addSeparator()
        menu_add.addAction("Camera").triggered.connect(lambda: self._controller.add_camera(self.chk_global_proxy.isChecked()))

        menu_edit = menubar.addMenu("Edit")
        menu_edit.addAction("Undo").triggered.connect(self._controller.project_ctrl.undo)
        menu_edit.addAction("Redo").triggered.connect(self._controller.project_ctrl.redo)
        menu_edit.addSeparator()
        menu_edit.addAction("Copy", self.action_copy)
        menu_edit.addAction("Cut", self.action_cut)
        menu_edit.addAction("Paste", self.action_paste)
        menu_edit.addAction("Delete", self.action_delete)

        menu_settings = menubar.addMenu("Settings")
        menu_settings.addAction("Background Color", self.action_change_bg_color)

    # =========================================================================
    # EVENT HANDLERS DELEGATING TO CONTROLLER
    # =========================================================================

    def _on_spawn_primitive_3d(self) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            name = action.data()
            self._controller.spawn_primitive(name, False)

    def _on_spawn_primitive_2d(self) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            name = action.data()
            self._controller.spawn_primitive(name, True)

    def _on_add_light(self) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            ltype = action.data()
            self._controller.add_light(
                ltype, self.chk_global_proxy.isChecked(), self.chk_global_light.isChecked()
            )

    def show_context_menu(self, pos: QPoint) -> None:
        """Builds and displays a dynamic context menu for scene interactions."""
        menu = QMenu(self)
        has_selection = (ctx.engine.get_selected_entity_id() >= 0)
        
        # --- ADD LOGIC TO CHECK VISIBILITY TOGGLE PERMISSION ---
        can_toggle_vis = has_selection
        if has_selection:
            data = ctx.engine.get_selected_entity_data()
            if data:
                is_light = bool(data.get("light"))
                is_cam = bool(data.get("cam"))
                
                # 1. Directional Light has no Proxy Mesh -> ALWAYS DISABLED
                if is_light and data["light"].get("type") == "Directional":
                    can_toggle_vis = False
                    
                # 2. Other lights or Camera -> Disabled if Global Proxy on Toolbar is OFF
                elif (is_light or is_cam) and not self.chk_global_proxy.isChecked():
                    can_toggle_vis = False
        # ---------------------------------------------------------
        
        menu.addAction("Copy").triggered.connect(self.action_copy)
        menu.addAction("Cut").triggered.connect(self.action_cut)
        
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
        group_count = len(getattr(self._controller.hierarchy_ctrl, "selected_multi_ids", []))
        act_group = QAction("Group Selected", self)
        act_group.setEnabled(group_count > 1)
        act_group.triggered.connect(self._controller.group_selected)
        menu.addAction(act_group)

        act_ungroup = QAction("Ungroup", self)
        can_ungroup = False
        if has_selection:
            data = ctx.engine.get_selected_entity_data()
            can_ungroup = bool(data and data.get("is_group"))
        act_ungroup.setEnabled(can_ungroup)
        act_ungroup.triggered.connect(self._controller.ungroup_selected)
        menu.addAction(act_ungroup)
        
        menu.addSeparator()
        act_del = QAction("Delete", self)
        act_del.setEnabled(has_selection)
        act_del.triggered.connect(self.action_delete)
        menu.addAction(act_del)
        
        if hasattr(self, 'hierarchy_view') and self.sender() == self.hierarchy_view:
            menu.exec(self.hierarchy_view.mapToGlobal(pos))
        else:
            menu.exec(pos)

    def action_copy(self) -> None: self._controller.copy_selected()
    def action_cut(self) -> None: self._controller.cut_selected()
    def action_delete(self) -> None: self._controller.delete_selected()
    def action_toggle_visibility(self) -> None: self._controller.toggle_visibility_selected()

    def action_paste(self) -> None: 
        # Keep paste behavior centralized in the controller for consistent validation.
        self._controller.paste_copied()

    def action_change_bg_color(self) -> None:
        current_bg = getattr(self.gl_widget, 'bg_color', DEFAULT_BG_COLOR)
        color = QColorDialog.getColor(QColor(int(current_bg[0]*255), int(current_bg[1]*255), int(current_bg[2]*255)), 
                                      self, "Select Background Color")
        if color.isValid():
            self.gl_widget.bg_color = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            self.gl_widget.update() 

    # =========================================================================
    # SETUP TOOLBAR
    # =========================================================================

    def create_toolbar(self) -> None:
        """Initializes the top layout containing tools and global render settings."""
        toolbar_tools = QToolBar("Transform Tools")
        toolbar_tools.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar_tools)
        
        container_tools = QWidget()
        lay_tools = QHBoxLayout(container_tools)
        lay_tools.setContentsMargins(10, 2, 20, 2)
        lay_tools.addWidget(QLabel("Tools:"))
        
        self.rad_mov = QRadioButton("Move")
        self.rad_mov.setChecked(True)
        self.rad_rot = QRadioButton("Rotate")
        self.rad_scl = QRadioButton("Scale")
        
        for r in [self.rad_mov, self.rad_rot, self.rad_scl]: 
            r.toggled.connect(self._on_tool_changed)
            lay_tools.addWidget(r)
            
        toolbar_tools.addWidget(container_tools)

        toolbar_view = QToolBar("View Settings")
        toolbar_view.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar_view)
        
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
        self.chk_comb_light.stateChanged.connect(self._on_render_settings_changed)
        
        self.chk_comb_tex = QCheckBox("+Textures")
        self.chk_comb_tex.setChecked(True)
        self.chk_comb_tex.stateChanged.connect(self._on_render_settings_changed)
        
        self.chk_comb_vcolor = QCheckBox("+Vertex Colors")
        self.chk_comb_vcolor.setChecked(True)
        self.chk_comb_vcolor.stateChanged.connect(self._on_render_settings_changed)
        
        lay_comb.addWidget(self.chk_comb_light)
        lay_comb.addWidget(self.chk_comb_tex)
        lay_comb.addWidget(self.chk_comb_vcolor)
        layout.addWidget(self.w_comb_opts)
        
        self.cmb_output = QComboBox()
        self.cmb_output.addItems(["Color (RGB)", "Depth Map"])
        self.cmb_output.currentIndexChanged.connect(self._on_render_settings_changed)
        layout.addWidget(QLabel(" | Output: "))
        layout.addWidget(self.cmb_output)

        toolbar_view.addWidget(container)

    def _on_tool_changed(self) -> None:
        """Dispatches the selected Gizmo manipulation mode (Translate, Rotate, Scale)."""
        if self.rad_rot.isChecked(): 
            self._controller.set_manipulation_mode("ROTATE")
        elif self.rad_mov.isChecked(): 
            self._controller.set_manipulation_mode("MOVE")
        elif self.rad_scl.isChecked(): 
            self._controller.set_manipulation_mode("SCALE")

    def _on_render_settings_changed(self, *args: Any) -> None:
        """Dispatches global shader and pipeline rendering configuration updates."""
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