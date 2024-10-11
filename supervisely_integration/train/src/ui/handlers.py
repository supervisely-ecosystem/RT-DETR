import os
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Stepper, Container

import src.ui.train as train
import src.ui.classes as classes_ui
import src.ui.train_val_split as splits_ui
import src.ui.hyperparameters.handlers as handlers
import src.ui.models as models
import src.ui.input_project as input_project
import src.ui.task as task_ui

from src import sly_utils
from src.utils import parse_yaml_metafile
from src.train_parameters import TrainParameters

from src.ui import hyperparameters
from src.ui import augmentations
from src.ui import model_leaderboard
from src.ui.utils import wrap_button_click, button_clicked, set_stepper_step
import src.sly_globals as g


all_widgets = [
    input_project.card,
    Container(widgets=[task_ui.card, model_leaderboard.card]),
    models.card,
    classes_ui.card,
    splits_ui.card,
    augmentations.card,
    hyperparameters.card,
    train.card,
]

stepper = Stepper(widgets=all_widgets)
stepper.set_active_step(2)

train_start_callback = wrap_button_click(
    train.start_train_btn,
    cards_to_unlock=[],
    widgets_to_disable=[
        task_ui.select_btn,
        models.select_btn,
        classes_ui.select_btn,
        splits_ui.select_btn,
        augmentations.select_btn,
        hyperparameters.select_btn,
    ],
    upd_params=False,
)

hyperparameters_select_callback = wrap_button_click(
    hyperparameters.select_btn,
    cards_to_unlock=[train.card],
    widgets_to_disable=[
        hyperparameters.general_params,
        hyperparameters.checkpoint_params,
        *hyperparameters.optimizers_params.values(),
        hyperparameters.select_optim,
        hyperparameters.apply_clip_input,
        hyperparameters.clip_input,
        *hyperparameters.schedulers_params.values(),
        hyperparameters.select_scheduler,
        hyperparameters.tabs,
        hyperparameters.warmup,
        hyperparameters.enable_warmup_input,
    ],
)

augmentations_select_callback = wrap_button_click(
    augmentations.select_btn,
    cards_to_unlock=[hyperparameters.card],
    widgets_to_disable=[augmentations.augments, augmentations.swithcer],
    callback=hyperparameters_select_callback,
)

splits_select_callback = wrap_button_click(
    splits_ui.select_btn,
    cards_to_unlock=[augmentations.card],
    widgets_to_disable=[splits_ui.splits],
    callback=augmentations_select_callback,
)

classes_select_callback = wrap_button_click(
    classes_ui.select_btn,
    cards_to_unlock=[splits_ui.card],
    widgets_to_disable=[
        classes_ui.classes,
        classes_ui.filter_images_without_gt_input,
    ],
    callback=splits_select_callback,
)

models_select_callback = wrap_button_click(
    models.select_btn,
    cards_to_unlock=[classes_ui.card],
    widgets_to_disable=[
        models.radio_tabs,
        models.arch_select,
        models.path_field,
        models.table,
        models.load_from,
    ],
    callback=classes_select_callback,
)

task_select_callback = wrap_button_click(
    task_ui.select_btn,
    [models.card, model_leaderboard.card],
    [task_ui.task_selector],
    models_select_callback,
)


# TASK
def on_task_changed(selected_task):
    models.update_architecture(selected_task)
    augmentations.update_task(selected_task)
    model_leaderboard.update_table(models.models_meta, selected_task)


@task_ui.select_btn.click
def select_task():
    task_select_callback()
    set_stepper_step(
        stepper,
        task_ui.select_btn,
        next_pos=3,
    )

    if button_clicked[task_ui.select_btn.widget_id]:
        on_task_changed(task_ui.task_selector.get_value())
    else:
        model_leaderboard.table.read_json(None)
        model_leaderboard.table.sort(0)


# MODELS
models.update_architecture(task_ui.task_selector.get_value())


@models.arch_select.value_changed
def on_architecture_selected(selected_arch):
    models.update_models(selected_arch)


@models.table.value_changed
def update_selected_model(selected_row):
    models.update_selected_model(selected_row)


@models.select_btn.click
def on_model_selected():
    # unlock cards
    models_select_callback()
    set_stepper_step(
        stepper,
        models.select_btn,
        next_pos=4,
    )

    mode = model_mode.get_active_tab()
    if mode == g.MODEL_MODES[0]:
        pretrained: List[str] = pretrained_models_table.get_selected_row()
        custom = None
        sly.logger.debug(f"Selected mode: {mode}, selected pretrained model: {pretrained}")
    else:
        pretrained = None
        custom: List[str] = select_custom_weights.get_selected_items()
        # TODO: Add single-item mode to the widget and remove indexing
        custom = custom[0] if custom else None
        sly.logger.debug(f"Selected mode: {mode}, path to custom weights: {custom}")
    finetune = finetune_checkbox.is_checked()
    g.train_mode = TrainMode(pretrained, custom, finetune)

    # unlock cards
    sly.logger.debug(f"State {classes_ui.card.widget_id}: {StateJson()[classes_ui.card.widget_id]}")


@classes_ui.classes.value_changed
def change_selected_classes(selected):
    selected_num = len(selected)
    if selected_num == 0:
        classes_ui.select_btn.disable()
    else:
        classes_ui.select_btn.enable()


@classes_ui.select_btn.click
def select_classes():
    classes_select_callback()
    set_stepper_step(
        stepper,
        classes_ui.select_btn,
        next_pos=5,
    )


@splits_ui.select_btn.click
def select_splits():
    splits_select_callback()
    set_stepper_step(
        stepper,
        splits_ui.select_btn,
        next_pos=6,
    )


@augmentations.select_btn.click
def select_augs():
    augmentations_select_callback()
    stepper.set_active_step(7)
    set_stepper_step(
        stepper,
        augmentations.select_btn,
        next_pos=7,
    )


@hyperparameters.select_btn.click
def select_hyperparameters():
    hyperparameters_select_callback()
    set_stepper_step(
        stepper,
        hyperparameters.select_btn,
        next_pos=8,
    )


@train.start_train_btn.click
def start_train():
    train_start_callback()
    g.USE_CACHE = input_project.use_cache_checkbox.is_checked()
    train.start_train()


@train.stop_train_btn.click
def stop_train():
    train.stop_train()
