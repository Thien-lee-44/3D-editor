from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.math_gen_view import MathGeneratorPanelView

class MathGenController:
    """Coordinates procedural math-surface generation requests from the UI."""
    def __init__(self) -> None:
        self.view = MathGeneratorPanelView(controller=self)

    @safe_execute(context="Generate Math Surface")
    def generate_surface(self, formula: str, xmin: float, xmax: float, ymin: float, ymax: float, res: int) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.spawn_math_surface(formula, xmin, xmax, ymin, ymax, res)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.spawn_math_surface(formula, xmin, xmax, ymin, ymax, res)
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)
