from PySide6.QtWidgets import (QMainWindow, QDockWidget, QWidget, QHBoxLayout, 
                               QRadioButton, QToolBar, QComboBox, QCheckBox, 
                               QLabel, QFrame, QSpacerItem, QSizePolicy, 
                               QColorDialog, QMessageBox, QMenu)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QAction
from typing import Any, Optional

from src.app import ctx, AppEvent

from src.app.config import (
    APP_TITLE, 
    DEFAULT_WINDOW_SIZE, 
    DEFAULT_BG_COLOR, 
    RENDER_MODE_COMBINED, 
    RENDER_MODE_FLAT
)

class EditorMainWindow(QMainWindow):
    """
    Main Application Shell.
    Responsible for assembling Toolbars, Menus, and the Docking system.
    Coordinates user interface interactions and delegates logic to the MainController.
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
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)

    def _setup_shortcuts(self) -> None:
        """Configures global hotkeys for Undo/Redo operations."""
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self._controller.project_ctrl.undo)
        
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self._controller.project_ctrl.redo)

    def set_central_viewport(self, viewport_widget: QWidget) -> None:
        self.gl_widget = viewport_widget
        self.setCentralWidget(self.gl_widget)

    def register_dock(self, panel_view: QWidget) -> None:
        """
        Wraps a Panel View into a QDockWidget and integrates it into the UI layout.
        """
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
    # ENGINE EVENT HANDLERS -> UI STATE LOCKING
    # =========================================================================

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

    # =========================================================================
    # MENU BAR & TOOLBAR CONSTRUCTION
    # =========================================================================

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

        toolbar_view.addWidget(container)

    # =========================================================================
    # CONTROLLER DELEGATES (Bridge logic)
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