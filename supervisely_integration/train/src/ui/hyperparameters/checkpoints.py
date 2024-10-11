from supervisely.app.widgets import (
    Container,
    InputNumber,
    Field,
    Text,
    Switch,
)

from src.ui.utils import InputContainer, get_switch_value, set_switch_value
from src.train_parameters import TrainParameters


NUM_EPOCHS = 10

checkpoint_params = InputContainer()

# interval
checkpoint_interval_input = InputNumber(1, 1, NUM_EPOCHS)
checkpoint_interval_text = Text(
    f"Save checkpoint every {checkpoint_interval_input.get_value()} epochs",
    status="info",
)
checkpoint_interval = Container([checkpoint_interval_input, checkpoint_interval_text])
checkpoint_interval_field = Field(checkpoint_interval, title="Checkpoints interval")
checkpoint_params.add_input("checkpoint_interval", checkpoint_interval_input)

# max number of saves
checkpoint_save_count_input = InputNumber(1, 1)
checkpoint_save_switch = Switch(switched=True)
checkpoint_save_count = Container([checkpoint_save_switch, checkpoint_save_count_input])
checkpoint_save_count_field = Field(
    checkpoint_save_count,
    title="Max checkpoints to keep",
    description=(
        "The maximum checkpoints to keep. "
        "In some cases we want only the latest "
        "few checkpoints and would like to delete "
        "old ones to save the disk space. "
        "If option is disabled then it means unlimited."
    ),
)
checkpoint_params.add_input("max_keep_checkpoints", checkpoint_save_count_input)
checkpoint_params.add_input(
    "saves_limited",
    checkpoint_save_switch,
    get_switch_value,
    set_switch_value,
)

# save last
checkpoint_last_switch = Switch()
checkpoint_last_field = Field(
    checkpoint_last_switch,
    title="Save last checkpoint",
    description="Whether to force the last checkpoint to be saved regardless of interval",
)
checkpoint_params.add_input(
    "save_last",
    checkpoint_last_switch,
    get_switch_value,
    set_switch_value,
)

# save best
checkpoint_best_switch = Switch(True)
checkpoint_best_field = Field(
    checkpoint_best_switch,
    title="Save best checkpoint",
    description="Best checkpoint will be marked with prefix 'best_'",
)
checkpoint_params.add_input(
    "save_best",
    checkpoint_best_switch,
    get_switch_value,
    set_switch_value,
)

# save optim
checkpoint_optimizer_switch = Switch(False)
checkpoint_optimizer_field = Field(
    checkpoint_optimizer_switch,
    title="Save optimizer",
)
checkpoint_params.add_input(
    "save_optimizer",
    checkpoint_optimizer_switch,
    get_switch_value,
    set_switch_value,
)


checkpoints_tab = Container(
    [
        checkpoint_interval_field,
        checkpoint_save_count_field,
        checkpoint_last_field,
        checkpoint_best_field,
        checkpoint_optimizer_field,
    ]
)


def update_checkpoint_widgets_with_params(params: TrainParameters):
    checkpoint_params.set("checkpoint_interval", params.checkpoint_interval)

    if checkpoint_params.saves_limited:  # == checkpoint_save_switch.is_switched():
        checkpoint_params.set("max_keep_checkpoints", params.max_keep_checkpoints)

    checkpoint_params.set("save_best", params.save_best)
    checkpoint_params.set("save_last", params.save_last)
    checkpoint_params.set("save_optimizer", params.save_optimizer)


def update_checkpoint_params_with_widgets(params: TrainParameters) -> TrainParameters:
    params.checkpoint_interval = checkpoint_params.checkpoint_interval
    params.save_best = checkpoint_params.save_best
    params.save_last = checkpoint_params.save_last
    params.save_optimizer = checkpoint_params.save_optimizer
    params.max_keep_checkpoints = checkpoint_params.max_keep_checkpoints
    if not checkpoint_params.saves_limited:
        params.max_keep_checkpoints = -1
