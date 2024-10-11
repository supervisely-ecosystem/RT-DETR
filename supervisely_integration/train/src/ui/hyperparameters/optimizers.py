from typing import Optional, Tuple
from supervisely.app.widgets import SelectString, InputNumber, Switch, Field, Container

from src.ui.utils import OrderedWidgetWrapper, get_switch_value, set_switch_value
from src.train_parameters import TrainParameters


optimizers_names = []


def get_betas(widgets: Tuple[InputNumber, InputNumber]) -> Tuple[float, float]:
    return widgets[0].get_value(), widgets[1].get_value()


def set_betas(widgets: Tuple[InputNumber, InputNumber], betas: Tuple[float, float]):
    widgets[0].value = betas[0]
    widgets[1].value = betas[1]


lr = InputNumber(0.01, 0, step=5e-5)
lr_field = Field(lr, title="Learning rate")

wd = InputNumber(1e-4, 0, step=1e-5)
wd_field = Field(wd, title="Weight decay")

adam_beta1 = InputNumber(0.9, 0, step=1e-5)
adam_beta1_field = Field(adam_beta1, title="Beta 1")

adam_beta2 = InputNumber(0.999, 0, step=1e-5)
adam_beta2_field = Field(adam_beta2, title="Beta 2")

betas = Container([adam_beta1_field, adam_beta2_field])

amsgrad_input = Switch()
amsgrad_field = Field(amsgrad_input, title="Amsgrad")

sgd_momentum = InputNumber(0.9, 0, step=1e-2)
sgd_momentum_field = Field(sgd_momentum, title="Momentum")


adam = OrderedWidgetWrapper("Adam")
adam.add_input("lr", lr, lr_field)
adam.add_input("weight_decay", wd, wd_field)
# adam.add_input("beta1", adam_beta1, adam_beta1_field)
# adam.add_input("beta2", adam_beta2, adam_beta2_field)
adam.add_input("betas", (adam_beta1, adam_beta2), betas, get_betas, set_betas)
adam.add_input("amsgrad", amsgrad_input, amsgrad_field, get_switch_value, set_switch_value)
optimizers_names.append(repr(adam))


adamw = OrderedWidgetWrapper("AdamW")
adamw.add_input("lr", lr, lr_field)
adamw.add_input("weight_decay", wd, wd_field)
# adamw.add_input("beta1", adam_beta1, adam_beta1_field)
# adamw.add_input("beta2", adam_beta2, adam_beta2_field)
adam.add_input("betas", (adam_beta1, adam_beta2), betas, get_betas, set_betas)
optimizers_names.append(repr(adamw))


sgd = OrderedWidgetWrapper("SGD")
sgd.add_input("lr", lr, lr_field)
sgd.add_input("weight_decay", wd, wd_field)
sgd.add_input("momentum", sgd_momentum, sgd_momentum_field)
optimizers_names.append(repr(sgd))

optimizers_params = {
    repr(adam): adam,
    repr(adamw): adamw,
    repr(sgd): sgd,
}

link = "https://pytorch.org/docs/1.10.0/optim.html#algorithms"
select_optim = SelectString(
    optimizers_names,
    optimizers_names,
    items_links=[link for _ in range(len(optimizers_names))],
)
select_optim_field = Field(
    select_optim,
    title="Select optimizer",
)


apply_clip_input = Switch(True)
clip_input = InputNumber(0.1, 0, step=1e-2)
clip_container = Container([apply_clip_input, clip_input])
clip_field = Field(
    clip_container,
    title="Clip gradient norm",
    description="Select the highest gradient norm value.",
)

frozen_stages_switch = Switch(False)


@frozen_stages_switch.value_changed
def frozen_stages_switch_changed(is_on: bool):
    if is_on:
        frozen_stages_input.show()
    else:
        frozen_stages_input.hide()


frozen_stages_input = InputNumber(-1, min=-1)
frozen_stages_input.hide()
frozen_stages_field = Field(
    Container([frozen_stages_switch, frozen_stages_input]),
    "Override frozen stages parameter",
    description=(
        "The first k stages for the backbone will be frozen. Set to -1 to disable freezing."
    ),
)


optimizers_tab = Container(
    [
        frozen_stages_field,
        select_optim_field,
        adam.create_container(),
        adamw.create_container(True),
        sgd.create_container(True),
        clip_field,
    ]
)


def get_optimizer_params() -> OrderedWidgetWrapper:
    name = select_optim.get_value()
    return optimizers_params[name]


def get_clip() -> Optional[float]:
    if apply_clip_input.is_switched():
        return clip_input.get_value()
    return None


def update_optimizer_widgets_with_params(params: TrainParameters):
    name = params.optimizer["type"]
    select_optim.set_value(name)

    for param, value in params.optimizer.items():
        if param in optimizers_params[name].get_params():
            optimizers_params[name].set(param, value)

    if params.clip_grad_norm is not None:
        clip_input.value = params.clip_grad_norm
        apply_clip_input.on()
    else:
        apply_clip_input.off()


def update_optimizer_params_with_widgets(params: TrainParameters) -> TrainParameters:
    name = select_optim.get_value()
    new_params = optimizers_params[name].get_params()
    new_params["type"] = name
    params.optimizer = new_params
    params.clip_grad_norm = get_clip()
    if frozen_stages_switch.is_on():
        params.frozen_stages = frozen_stages_input.get_value()
    return params
