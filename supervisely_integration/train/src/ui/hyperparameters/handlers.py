import numpy as np
from typing import List
from supervisely.app.widgets import Widget

from src.ui.hyperparameters.checkpoints import (
    checkpoint_interval_text,
    checkpoint_interval_input,
    checkpoint_save_switch,
    checkpoint_save_count_input,
)
from src.ui.hyperparameters.general import (
    chart_update_text,
    val_text,
    chart_update_input,
    validation_input,
    epochs_input,
    smaller_size_input,
    bigger_size_input,
)

from src.ui.hyperparameters.optimizers import (
    optimizers_params,
    select_optim,
    apply_clip_input,
    clip_input,
    get_optimizer_params,
)

from src.ui.hyperparameters.lr_scheduler import (
    schedulers_params,
    select_scheduler,
    get_scheduler_params,
    enable_warmup_input,
    warmup,
    etamin_switch_input,
    etamin_input,
    etamin_ratio_input,
    preview_btn,
    preview_chart,
    clear_btn,
)

from src.train_parameters import TrainParameters
from src import visualize_scheduler
from src.sly_utils import get_images_count
from src.ui.hyperparameters import update_params_with_widgets


def show_hide_fields(fields: List[Widget], hide: bool = True):
    for field in fields:
        if hide:
            field.hide()
        else:
            field.show()


@epochs_input.value_changed
def epoch_num_changes(new_value):
    validation_input.max = new_value
    checkpoint_interval_input.max = new_value


@validation_input.value_changed
def update_validation_desc(new_value):
    val_text.text = f"Evaluate validation set every {new_value} epochs"


@chart_update_input.value_changed
def update_chart_desc(new_value):
    chart_update_text.text = f"Update chart every {new_value} iterations"


@checkpoint_interval_input.value_changed
def update_ch_interval_desc(new_value):
    checkpoint_interval_text.text = f"Save checkpoint every {new_value} epochs"


@apply_clip_input.value_changed
def enable_disable_clip(new_value):
    if new_value:
        clip_input.enable()
    else:
        clip_input.disable()


@select_scheduler.value_changed
def update_scheduler(new_value):
    for scheduler in schedulers_params.keys():
        if new_value == scheduler:
            schedulers_params[scheduler].show()
        else:
            schedulers_params[scheduler].hide()


@select_optim.value_changed
def update_optim(new_value):
    for optim in optimizers_params.keys():
        if new_value == optim:
            optimizers_params[optim].show()
        else:
            optimizers_params[optim].hide()


@smaller_size_input.value_changed
def set_min_for_max_size(new_min):
    bigger_size_input.min = new_min


@bigger_size_input.value_changed
def set_max_for_min_size(new_max):
    smaller_size_input.max = new_max


@enable_warmup_input.value_changed
def show_hide_scheduler(new_value):
    if new_value:
        warmup.show()
    else:
        warmup.hide()


@checkpoint_save_switch.value_changed
def on_checkpoint_save_limit_switch(is_switched):
    if is_switched:
        checkpoint_save_count_input.show()
    else:
        checkpoint_save_count_input.hide()


@etamin_switch_input.value_changed
def etamin_or_ratio_switch(new_value):
    if new_value:
        etamin_input.enable()
        etamin_ratio_input.disable()
    else:
        etamin_input.disable()
        etamin_ratio_input.enable()


@preview_btn.click
def on_preivew_scheduler():
    test_params = TrainParameters()
    update_params_with_widgets(test_params)
    param_scheduler = visualize_scheduler.get_param_scheduler(test_params)

    total_epochs = test_params.total_epochs
    batch_size = test_params.batch_size_train
    start_lr = test_params.optimizer["lr"]

    x, lrs = visualize_scheduler.test_schedulers(
        param_scheduler, start_lr, int(np.ceil(get_images_count() / batch_size)), total_epochs
    )
    name = get_preview_name(param_scheduler)
    preview_chart.add_series(f"{name}", x, lrs)
    preview_chart.show()


def get_preview_name(param_scheduler: list):
    name = ""
    if len(param_scheduler) == 2:
        # with warmup
        name += "warmup + "
    name += param_scheduler[-1]["type"]
    return name


@clear_btn.click
def on_clear_preview():
    preview_chart.set_series([])
