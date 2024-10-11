from typing import List
import supervisely as sly
from supervisely.app.widgets import (
    RadioTabs,
    RadioTable,
    SelectString,
    Card,
    Container,
    Button,
    Text,
    Field,
    TeamFilesSelector,
    Switch,
)

from src.ui.task import task_selector
from src.ui.utils import update_custom_params
from src.sly_globals import TEAM_ID
from src.utils import parse_yaml_metafile
from rtdetr_pytorch.model_list import _models


def _get_table_data():
    columns = [
        "Name",
        "Dataset",
        "AP_Val",
        "Params(M)",
        "FRPS(T4)",
    ]

    # collect rows
    rows = [[value for key, value in model_info.items() if key != "meta"] for model_info in _models]
    subtitles = [None] * len(columns)
    return columns, rows, subtitles


def is_pretrained_model_selected():
    custom_path = get_selected_custom_path()
    if radio_tabs.get_active_tab() == "Pretrained models":
        if custom_path:
            raise Exception(
                "Active tab is Pretrained models, but the path to the custom weights is selected. This is ambiguous."
            )
        return True
    else:
        if custom_path:
            return False
        else:
            raise Exception(
                "Active tab is Custom weights, but the path to the custom weights isn't selected."
            )


def get_selected_pretrained_model() -> dict:
    global selected_model
    if selected_model:
        return selected_model


def get_selected_custom_path() -> str:
    paths = input_file.get_selected_paths()
    return paths[0] if len(paths) > 0 else ""


cur_task = task_selector.get_value()
selected_model: dict = None
models_meta: dict = None
models: list = None

table = RadioTable([""], [[""]])
text = Text()

load_from = Switch(True)
load_from_field = Field(
    load_from,
    "Download pre-trained model",
    "Whether to download pre-trained weights and finetune the model or train it from scratch.",
)

input_file = TeamFilesSelector(TEAM_ID, selection_file_type="file")
path_field = Field(
    title="Path to weights file",
    description="Copy path in Team Files",
    content=input_file,
)

radio_tabs = RadioTabs(
    titles=["Pretrained models", "Custom weights"],
    contents=[
        Container(widgets=[table, text, load_from_field]),
        path_field,
    ],
)

select_btn = Button(text="Select model")

card = Card(
    title=f"{cur_task} models",
    description="Choose model weights",
    content=Container([radio_tabs, select_btn]),
    lock_message="Select task to unlock.",
)
card.lock()


def update_models():
    columns, rows, subtitles = _get_table_data()
    table.set_data(columns, rows, subtitles)
    table.select_row(0)
    update_selected_model(table.get_selected_row())


def update_selected_model(selected_row):
    global selected_model, models
    idx = table.get_selected_row_index()
    selected_model = models[idx]
    text.text = f"Selected model: {selected_row[0]}"
