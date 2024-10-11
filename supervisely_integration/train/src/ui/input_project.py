from supervisely.app.widgets import Card, ProjectThumbnail, Text, Checkbox, Container
from supervisely.project.download import is_cached
from src.sly_globals import PROJECT_ID, USE_CACHE, project_info


project_th = ProjectThumbnail(project_info)
if is_cached(PROJECT_ID):
    _text = "Use cached data stored on the agent to optimize project downlaod"
else:
    _text = "Cache data on the agent to optimize project download for future trainings"
use_cache_text = Text(_text)
use_cache_checkbox = Checkbox(use_cache_text, checked=USE_CACHE)

card = Card(
    title="Input project",
    description="Selected project from which images and annotations will be downloaded",
    content=Container(widgets=[project_th, use_cache_checkbox]),
)
