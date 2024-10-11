import src.sly_globals as g
import supervisely as sly
from supervisely.app.widgets import Container
import src.ui.handlers as handlers

layout = Container(widgets=[handlers.stepper])
app = sly.Application(layout=layout)

g.app = app
