from supervisely.app.widgets import Tabs, Card, Button, Container

from src.ui.hyperparameters.checkpoints import checkpoints_tab, checkpoint_params
from src.ui.hyperparameters.general import general_tab, general_params
from src.ui.hyperparameters.optimizers import (
    optimizers_tab,
    optimizers_params,
    apply_clip_input,
    clip_input,
    select_optim,
)
from src.ui.hyperparameters.lr_scheduler import (
    schedulres_tab,
    schedulers_params,
    get_scheduler_params,
    warmup,
    enable_warmup_input,
    select_scheduler,
)
from src.train_parameters import TrainParameters

from src.ui.hyperparameters import general
from src.ui.hyperparameters import checkpoints
from src.ui.hyperparameters import optimizers
from src.ui.hyperparameters import lr_scheduler
from src.ui import classes
from src.ui import models

from src.ui.hyperparameters.general import run_model_benchmark_checkbox # , run_speedtest_checkbox


tabs = Tabs(
    labels=[
        "General",
        "Checkpoints",
        "Optimizer (Advanced)",
        "Learning rate scheduler (Advanced)",
    ],
    contents=[general_tab, checkpoints_tab, optimizers_tab, schedulres_tab],
)

select_btn = Button(text="Select")
content = Container(widgets=[tabs, select_btn])

card = Card(
    title="Training hyperparameters",
    description="Configure the training process.",
    content=content,
)
card.lock("Select augmentations.")


def update_widgets_with_params(params: TrainParameters):
    general.update_general_widgets_with_params(params)
    checkpoints.update_checkpoint_widgets_with_params(params)
    optimizers.update_optimizer_widgets_with_params(params)
    lr_scheduler.update_scheduler_widgets_with_params(params)


def update_params_with_widgets(params: TrainParameters):
    general.update_general_params_with_widgets(params)
    checkpoints.update_checkpoint_params_with_widgets(params)
    optimizers.update_optimizer_params_with_widgets(params)
    lr_scheduler.update_scheduler_params_with_widgets(params)
    params.load_from = models.load_from.is_switched()
    params.filter_images_without_gt = classes.filter_images_without_gt_input.is_switched()
