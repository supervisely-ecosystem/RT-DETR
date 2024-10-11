from typing import Tuple
from supervisely.app.widgets import (
    Checkbox,
    Container,
    InputNumber,
    BindedInputNumber,
    Select,
    Field,
    Text,
    Empty,
    Switch,
)

from src.ui.utils import InputContainer, get_switch_value, set_switch_value, get_devices
from src.train_parameters import TrainParameters

NUM_EPOCHS = 10
# General
general_params = InputContainer()

# device_names, torch_devices = get_devices()
# device_input = Select([Select.Item(v, l) for v, l in zip(torch_devices, device_names)])
# device_field = Field(
#     device_input,
#     title="Device",
#     description=(
#         "Run nvidia-smi or check agent page to "
#         "see how many devices your machine has "
#         "or keep by default"
#     ),
# )
# general_params.add_input("device", device_input)


epochs_input = InputNumber(NUM_EPOCHS, min=1)
epochs_field = Field(epochs_input, "Number of epochs")
general_params.add_input("total_epochs", epochs_input)


def size_and_prop(inp: BindedInputNumber) -> Tuple[Tuple[int, int], bool]:
    return inp.get_value(), inp.proportional


bigger_size_input = InputNumber(1333, 1)
bigger_size = Field(bigger_size_input, "Longer edge")
smaller_size_input = InputNumber(800, 1)
smaller_size = Field(smaller_size_input, "Shorter edge")
general_params.add_input("bigger_size", bigger_size_input)
general_params.add_input("smaller_size", smaller_size_input)
general_params.set("smaller_size", 800)

size_input = Container(
    [bigger_size, smaller_size, Empty()], direction="horizontal", fractions=[1, 1, 5]
)

size_field = Field(
    size_input,
    title="Input size",
    description="Images will be scaled approximately to the specified sizes while keeping the aspect ratio "
    "(internally, the sizes are passed as 'scale' parameter of the 'Resize' transform in mmcv).",
)


bs_train_input = InputNumber(1, 1)
bs_train_field = Field(bs_train_input, "Train batch size")
general_params.add_input("batch_size_train", bs_train_input)

bs_val_input = InputNumber(1, 1)
bs_val_field = Field(bs_val_input, "Validation batch size")
general_params.add_input("batch_size_val", bs_val_input)


validation_input = InputNumber(1, 1, general_params.total_epochs)
val_text = Text(
    f"Evaluate validation set every {validation_input.get_value()} epochs",
    status="info",
)
validation_interval = Container([validation_input, val_text])

validation_field = Field(
    validation_interval,
    title="Validation interval",
    description=(
        "By default we evaluate the model on the "
        "validation set after each epoch, you can "
        "change the evaluation interval"
    ),
)
general_params.add_input("val_interval", validation_input)


chart_update_input = InputNumber(50, 1)
chart_update_text = Text(
    f"Update chart every {chart_update_input.get_value()} iterations",
    status="info",
)
chart_update_interval = Container([chart_update_input, chart_update_text])

chart_update_field = Field(
    chart_update_interval,
    title="Chart update interval",
    description=("How often сhart should be updated."),
)
general_params.add_input("chart_update_interval", chart_update_input)

run_model_benchmark_checkbox = Checkbox(content="Run Model Benchmark evaluation", checked=True)
# run_speedtest_checkbox = Checkbox(content="Run speed test", checked=True)
model_benchmark_f = Field(
    Container(
        widgets=[
            run_model_benchmark_checkbox,
            # run_speedtest_checkbox,
        ]
    ),
    title="Model Evaluation Benchmark",
    description=f"Generate evalutaion dashboard with visualizations and detailed analysis of the model performance after training. The best checkpoint will be used for evaluation. You can also run speed test to evaluate model inference speed.",
)
docs_link = '<a href="https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/" target="_blank">documentation</a>'
model_benchmark_learn_more = Text(
    f"Learn more about Model Benchmark in the {docs_link}.", status="info"
)

general_tab = Container(
    [
        # device_field,
        epochs_field,
        size_field,
        bs_train_field,
        bs_val_field,
        validation_field,
        chart_update_field,
        model_benchmark_f,
        model_benchmark_learn_more,
    ]
)


def update_general_widgets_with_params(params: TrainParameters):
    general_params.set("total_epochs", params.total_epochs)
    general_params.set("val_interval", params.val_interval)
    general_params.set("batch_size_train", params.batch_size_train)
    general_params.set("batch_size_val", params.batch_size_val)
    general_params.set("bigger_size", max(params.input_size))
    general_params.set("smaller_size", min(params.input_size))
    general_params.set("chart_update_interval", params.chart_update_interval)

    chart_update_text.text = f"Update chart every {params.chart_update_interval} iterations"
    val_text.text = f"Evaluate validation set every {params.val_interval} epochs"


def update_general_params_with_widgets(params: TrainParameters):
    params.total_epochs = general_params.total_epochs
    params.val_interval = general_params.val_interval
    params.batch_size_train = general_params.batch_size_train
    params.batch_size_val = general_params.batch_size_val
    params.input_size = (general_params.bigger_size, general_params.smaller_size)
    params.chart_update_interval = general_params.chart_update_interval
    # params.device_name = general_params.device


# @run_model_benchmark_checkbox.value_changed
# def change_model_benchmark(value):
#     if value:
#         run_speedtest_checkbox.show()
#     else:
#         run_speedtest_checkbox.hide()
