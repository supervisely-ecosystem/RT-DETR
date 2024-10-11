import os
import supervisely as sly
from pathlib import Path
from supervisely.app.widgets import AugmentationsWithTabs, Card, Container, Switch, Button

import src.sly_globals as g
from src.ui.task import task_selector
from src import sly_utils


def format_task_name(task: str):
    if "segmentation" in task.lower():
        task = "segmentation"
    else:
        task = "detection"
    return task


def name_from_path(aug_path):
    name = os.path.basename(aug_path).split(".json")[0].capitalize()
    name = " + ".join(name.split("_"))
    return name


template_dir = "supervisely_integration/train/aug_templates"
template_paths = list(map(str, Path(template_dir).glob("*.json")))
template_paths = sorted(template_paths, key=lambda x: x.replace(".", "_"))[::-1]

templates = [{"label": name_from_path(path), "value": path} for path in template_paths]


swithcer = Switch(True)
augments = AugmentationsWithTabs(
    g, task_type=format_task_name(task_selector.get_value()), templates=templates
)

select_btn = Button("Select")
container = Container([swithcer, augments, select_btn])

card = Card(
    title="Training augmentations",
    description="Choose one of the prepared templates or provide custom pipeline",
    content=container,
)
card.lock("Confirm splits.")


def update_task(task_name):
    task = format_task_name(task_name)
    augments._augs1._task_type = task
    augments._augs2._task_type = task


def reset_widgets():
    if swithcer.is_switched():
        augments.show()
    else:
        augments.hide()


def get_selected_aug():
    # path to aug pipline (.json file)
    if swithcer.is_switched():
        return augments._current_augs._template_path
    else:
        return None


@swithcer.value_changed
def on_switch(is_switched: bool):
    reset_widgets()


reset_widgets()
