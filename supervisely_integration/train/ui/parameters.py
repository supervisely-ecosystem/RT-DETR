import os
import shutil
from datetime import datetime
from typing import List

import supervisely as sly
import yaml
from pycocotools.coco import COCO
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Field,
    Input,
    InputNumber,
    RadioTabs,
    ReloadableArea,
    Select,
)

import rtdetr_pytorch.train as train_cli
import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.output as output

# region advanced widgets
advanced_mode_checkbox = Checkbox("Advanced mode")
advanced_mode_field = Field(
    advanced_mode_checkbox,
    title="Advanced mode",
    description="Enable advanced mode to specify custom training parameters manually.",
)
with open(g.default_config_path, "r") as f:
    default_config = f.read()
    sly.logger.debug(f"Loaded default config from {g.default_config_path}")
advanced_mode_editor = Editor(default_config, language_mode="yaml", height_lines=100)
advanced_mode_editor.hide()
# endregion

# region general widgets
number_of_epochs_input = InputNumber(value=20, min=1)
number_of_epochs_field = Field(
    number_of_epochs_input,
    title="Number of epochs",
    description="The number of epochs to train the model for",
)
input_size_input = InputNumber(value=1000)
input_size_field = Field(
    input_size_input,
    title="Input size",
    description="Images will be scaled to this size before training while keeping the aspect ratio.",
)

train_batch_size_input = InputNumber(value=2, min=1)
train_batch_size_field = Field(
    train_batch_size_input,
    title="Train batch size",
    description="The number of images in a batch during training",
)

val_batch_size_input = InputNumber(value=2, min=1)
val_batch_size_field = Field(
    val_batch_size_input,
    title="Validation batch size",
    description="The number of images in a batch during validation",
)

validation_interval_input = InputNumber(value=1, min=1)
validation_interval_field = Field(
    validation_interval_input,
    title="Validation interval",
    description="The number of epochs between each validation run",
)

general_tab = Container(
    [
        number_of_epochs_field,
        input_size_field,
        train_batch_size_field,
        val_batch_size_field,
        validation_interval_field,
    ]
)
# endregion

# region checkpoints widgets
checkpoints_interval_input = InputNumber(value=1, min=1)
checkpoints_interval_field = Field(
    checkpoints_interval_input,
    title="Checkpoints interval",
    description="The number of epochs between each checkpoint save",
)

save_last_checkpoint_checkbox = Checkbox("Save last checkpoint")
save_best_checkpoint_checkbox = Checkbox("Save best checkpoint")

save_checkpoint_field = Field(
    Container([save_last_checkpoint_checkbox, save_best_checkpoint_checkbox]),
    title="Save checkpoints",
    description="Choose which checkpoints to save",
)

checkpoints_tab = Container([checkpoints_interval_field, save_checkpoint_field])

# endregion

# region optimizer widgets
optimizer_select = Select([Select.Item(opt) for opt in g.OPTIMIZERS])
optimizer_field = Field(
    optimizer_select,
    title="Select optimizer",
    description="Choose the optimizer to use for training",
)
learning_rate_input = InputNumber(value=0.0001)
learning_rate_field = Field(
    learning_rate_input,
    title="Learning rate",
    description="The learning rate to use for the optimizer",
)
wight_decay_input = InputNumber(value=0.0001)
wight_decay_field = Field(
    wight_decay_input,
    title="Weight decay",
    description="The amount of L2 regularization to apply to the weights",
)
momentum_input = InputNumber(value=0.9)
momentum_field = Field(
    momentum_input,
    title="Momentum",
    description="The amount of momentum to apply to the weights",
)
momentum_field.hide()
beta1_input = InputNumber(value=0.9)
beta1_field = Field(
    beta1_input,
    title="Beta 1",
    description="The exponential decay rate for the first moment estimates",
)
beta2_input = InputNumber(value=0.999)
beta2_field = Field(
    beta2_input,
    title="Beta 2",
    description="The exponential decay rate for the second moment estimates",
)

amsgrad_checkbox = Checkbox("AMSGrad")

clip_gradient_norm_checkbox = Checkbox("Clip gradient norm")
clip_gradient_norm_input = InputNumber(value=0.1)
clip_gradient_norm_field = Field(
    Container([clip_gradient_norm_checkbox, clip_gradient_norm_input]),
    title="Clip gradient norm",
    description="Select the highest gradient norm to clip the gradients",
)


optimization_tab = Container(
    [
        optimizer_field,
        learning_rate_field,
        wight_decay_field,
        momentum_field,
        beta1_field,
        beta2_field,
        amsgrad_checkbox,
        clip_gradient_norm_field,
    ]
)

# endregion

# region scheduler widgets
scheduler_select = Select([Select.Item(sch) for sch in g.SCHEDULERS])

scheduler_widgets_container = Container()
scheduler_parameters_area = ReloadableArea(scheduler_widgets_container)

enable_warmup_checkbox = Checkbox("Enable warmup", True)
warmup_iterations_input = InputNumber(value=2)
warmup_iterations_field = Field(
    warmup_iterations_input,
    title="Warmup iterations",
    description="The number of iterations to warm up the learning rate",
)
warmup_ratio_input = InputNumber(value=0.001)
warmup_ratio_field = Field(
    warmup_ratio_input,
    title="Warmup ratio",
    description="The ratio of the initial learning rate to use for warmup",
)
warmup_container = Container([warmup_iterations_field, warmup_ratio_field])

learning_rate_scheduler_tab = Container(
    [
        scheduler_select,
        scheduler_parameters_area,
        enable_warmup_checkbox,
        warmup_container,
    ]
)

# endregion

run_button = Button("Run training")
stop_button = Button("Stop training", button_type="danger")
stop_button.hide()


parameters_tabs = RadioTabs(
    ["General", "Checkpoints", "Optimization", "Learning rate scheduler"],
    contents=[
        general_tab,
        checkpoints_tab,
        optimization_tab,
        learning_rate_scheduler_tab,
    ],
)

card = Card(
    title="5️⃣ Training hyperparameters",
    description="Specify training hyperparameters using one of the methods.",
    collapsable=True,
    content=Container(
        [advanced_mode_field, advanced_mode_editor, parameters_tabs, run_button],
    ),
    content_top_right=stop_button,
)
# card.lock()
# card.collapse()


@advanced_mode_checkbox.value_changed
def advanced_mode_changed(is_checked: bool):
    if is_checked:
        advanced_mode_editor.show()
        parameters_tabs.hide()
    else:
        advanced_mode_editor.hide()
        parameters_tabs.show()


@optimizer_select.value_changed
def optimizer_changed(optimizer: str):
    if optimizer == "Adam":
        beta1_field.show()
        beta2_field.show()
        amsgrad_checkbox.show()
        momentum_field.hide()
    elif optimizer == "AdamW":
        beta1_field.hide()
        beta2_field.hide()
        amsgrad_checkbox.hide()
        momentum_field.hide()
    elif optimizer == "SGD":
        beta1_field.hide()
        beta2_field.hide()
        amsgrad_checkbox.hide()
        momentum_field.show()


@scheduler_select.value_changed
def scheduler_changed(scheduler: str):
    # StepLR: by_epoch_checkbox, LR scheduler step (input number), gamma (input number)
    # MultiStepLR: by_epoch_checkbox, LR scheduler steps (input), gamma (input number)
    # ExponentialLR: by_epoch_checkbox, gamma (input number)
    # ReduceLROnPlateauLR: by_epoch_checkbox, factor (input number), patience (input number)
    # CosineAnnealingLR: by_epoch_checkbox, T_max (input number), min lr checkbox, min lr (input number), min lr ratio (input number)
    # CosineRestartLR: by_epoch_checkbox, periods (input), restarts (input), min lr checkbox, min lr (input number), min lr ratio (input number)

    scheduler_widgets_container._widgets.clear()
    scheduler_parameters_area.reload()
    g.widgets = None

    widgets = {
        "by_epoch": Checkbox("By epoch", True),
    }

    if scheduler == "StepLR":
        widgets["step"] = InputNumber(value=1)
        widgets["gamma"] = InputNumber(value=0.1)
    elif scheduler == "MultiStepLR":
        widgets["steps"] = Input("1,2,3")
        widgets["gamma"] = InputNumber(value=0.1)
    elif scheduler == "ExponentialLR":
        widgets["gamma"] = InputNumber(value=0.1)
    elif scheduler == "ReduceLROnPlateau":
        widgets["factor"] = InputNumber(value=0.1)
        widgets["patience"] = InputNumber(value=10)
    elif scheduler == "CosineAnnealingLR":
        widgets["T_max"] = InputNumber(value=10)
        widgets["min_lr"] = Checkbox("Min LR", True)
        widgets["min_lr_value"] = InputNumber(value=0.001)
        widgets["min_lr_ratio"] = InputNumber(value=0.1)
    elif scheduler == "CosineRestartLR":
        widgets["periods"] = Input("10,20,30")
        widgets["restarts"] = Input("2,3,4")
        widgets["min_lr"] = Checkbox("Min LR", True)
        widgets["min_lr_value"] = InputNumber(value=0.001)
        widgets["min_lr_ratio"] = InputNumber(value=0.1)

    g.widgets = widgets

    scheduler_widgets_container._widgets.extend([widget for widget in widgets.values()])
    scheduler_parameters_area.reload()


@run_button.click
def run_training():
    output.card.unlock()
    output.card.uncollapse()

    download_project()
    create_trainval()

    read_parameters()

    prepare_config()
    cfg = train()
    save_config(cfg)
    out_path = upload_model(cfg.output_dir)
    print(out_path)


@stop_button.click
def stop_training():
    # TODO: Implement the stop process
    pass


def read_parameters():
    pass


def prepare_config():
    custom_config_text = advanced_mode_editor.get_value()
    model_name = g.train_mode.pretrained[0]
    arch = model_name.split("_coco")[0]
    config_name = f"{arch}_6x_coco"
    sly.logger.info(f"Model name: {model_name}, arch: {arch}, config_name: {config_name}")

    custom_config = yaml.safe_load(custom_config_text) or {}
    custom_config["__include__"] = [f"{config_name}.yml"]
    custom_config["remap_mscoco_category"] = False
    custom_config["num_classes"] = len(g.selected_classes)

    custom_config["train_dataloader"] = {
        "dataset": {
            "img_folder": f"{g.train_dataset_path}/img",
            "ann_file": f"{g.train_dataset_path}/coco_anno.json",
        }
    }
    custom_config["val_dataloader"] = {
        "dataset": {
            "img_folder": f"{g.val_dataset_path}/img",
            "ann_file": f"{g.val_dataset_path}/coco_anno.json",
        }
    }
    selected_classes = [obj_class.name for obj_class in g.selected_classes]
    custom_config["sly_metadata"] = {
        "classes": selected_classes,
        "project_id": g.selected_project_id,
        "project_name": g.selected_project_info.name,
        "model": model_name,
    }

    g.custom_config_path = os.path.join(g.CONFIG_PATHS_DIR, "custom_config.yml")
    with open(g.custom_config_path, "w") as f:
        yaml.dump(custom_config, f)


def train():
    model = g.train_mode.pretrained[0]
    finetune = g.train_mode.finetune
    cfg = train_cli.train(model, finetune, g.custom_config_path)
    return cfg


def save_config(cfg):
    if "__include__" in cfg.yaml_cfg:
        cfg.yaml_cfg.pop("__include__")

    output_path = os.path.join(g.OUTPUT_DIR, "config.yml")

    with open(output_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)


def upload_model(output_dir):
    model_name = g.train_mode.pretrained[0]
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    team_files_dir = (
        f"/RT-DETR/{g.selected_project_info.name}_{g.selected_project_id}/{timestamp}_{model_name}"
    )
    local_dir = f"{output_dir}/upload"
    sly.fs.mkdir(local_dir)

    checkpoints = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    latest_checkpoint = sorted(checkpoints)[-1]
    shutil.move(f"{output_dir}/{latest_checkpoint}", f"{local_dir}/{latest_checkpoint}")
    shutil.move(f"{output_dir}/log.txt", f"{local_dir}/log.txt")
    shutil.move("output/config.yml", f"{local_dir}/config.yml")

    out_path = g.api.file.upload_directory(
        sly.env.team_id(),
        local_dir,
        team_files_dir,
    )
    return out_path


def download_project():
    g.project_dir = os.path.join(g.DOWNLOAD_DIR, g.selected_project_info.name)
    sly.logger.info(f"Downloading project to {g.project_dir}...")
    sly.Project.download(g.api, g.selected_project_info.id, g.project_dir)
    sly.logger.info(f"Project downloaded to {g.project_dir}.")
    g.project = sly.Project(g.project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project loaded from {g.project_dir}.")


def create_trainval():
    train_items, val_items = g.splits
    sly.logger.debug(f"Creating trainval datasets from splits: {g.splits}...")
    train_items: List[sly.project.project.ItemInfo]
    val_items: List[sly.project.project.ItemInfo]

    converted_project_dir = os.path.join(g.CONVERTED_DIR, g.selected_project_info.name)
    sly.logger.debug(f"Converted project will be saved to {converted_project_dir}.")
    sly.fs.mkdir(converted_project_dir)
    train_dataset_path = os.path.join(converted_project_dir, "train")
    val_dataset_path = os.path.join(converted_project_dir, "val")
    sly.logger.debug(
        f"Train dataset path: {train_dataset_path}, val dataset path: {val_dataset_path}."
    )

    g.train_dataset_path = train_dataset_path
    g.val_dataset_path = val_dataset_path

    project_meta_path = os.path.join(converted_project_dir, "meta.json")
    sly.json.dump_json_file(g.project.meta.to_json(), project_meta_path)

    for items, dataset_path in zip(
        [train_items, val_items], [train_dataset_path, val_dataset_path]
    ):
        prepare_dataset(dataset_path, items)

    g.converted_project = sly.Project(converted_project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project created in {converted_project_dir}")

    for dataset_fs in g.converted_project.datasets:
        dataset_fs: sly.Dataset
        selected_classes = [obj_class.name for obj_class in g.selected_classes]

        coco_anno = get_coco_annotations(dataset_fs, g.converted_project.meta, selected_classes)
        coco_anno_path = os.path.join(dataset_fs.directory, "coco_anno.json")
        sly.json.dump_json_file(coco_anno, coco_anno_path)

    sly.logger.info("COCO annotations created")


def prepare_dataset(dataset_path: str, items: List[sly.project.project.ItemInfo]):
    sly.logger.debug(f"Preparing dataset in {dataset_path}...")
    img_dir = os.path.join(dataset_path, "img")
    ann_dir = os.path.join(dataset_path, "ann")
    sly.fs.mkdir(img_dir)
    sly.fs.mkdir(ann_dir)
    for item in items:
        src_img_path = os.path.join(g.project_dir, fix_widget_path(item.img_path))
        src_ann_path = os.path.join(g.project_dir, fix_widget_path(item.ann_path))
        dst_img_path = os.path.join(img_dir, item.name)
        dst_ann_path = os.path.join(ann_dir, f"{item.name}.json")
        sly.fs.copy_file(src_img_path, dst_img_path)
        sly.fs.copy_file(src_ann_path, dst_ann_path)

    sly.logger.info(f"Dataset prepared in {dataset_path}")


def fix_widget_path(bugged_path: str) -> str:
    """Fixes the broken ItemInfo paths from TrainValSplits widget.
    Removes the first two folders from the path.

    Bugged path: app_data/1IkWRgJG62f1ZuZ/ds0/ann/pexels_2329440.jpeg.json
    Corrected path: ds0/ann/pexels_2329440.jpeg.json

    :param bugged_path: Path to fix
    :type bugged_path: str
    :return: Fixed path
    :rtype: str
    """
    return "/".join(bugged_path.split("/")[2:])


def get_coco_annotations(dataset: sly.Dataset, meta: sly.ProjectMeta, selected_classes: List[str]):
    coco_anno = {"images": [], "categories": [], "annotations": []}
    cat2id = {name: i for i, name in enumerate(selected_classes)}
    img_id = 1
    ann_id = 1
    for name in dataset.get_items_names():
        ann = dataset.get_ann(name, meta)
        img_dict = {
            "id": img_id,
            "height": ann.img_size[0],
            "width": ann.img_size[1],
            "file_name": name,
        }
        coco_anno["images"].append(img_dict)

        for label in ann.labels:
            if isinstance(label.geometry, (sly.Bitmap, sly.Polygon)):
                rect = label.geometry.to_bbox()
            elif isinstance(label.geometry, sly.Rectangle):
                rect = label.geometry
            else:
                continue
            class_name = label.obj_class.name
            if class_name not in selected_classes:
                continue
            x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
            ann_dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat2id[class_name],
                "bbox": [x, y, x2 - x, y2 - y],
                "area": (x2 - x) * (y2 - y),
                "iscrowd": 0,
            }
            coco_anno["annotations"].append(ann_dict)
            ann_id += 1

        img_id += 1

    coco_anno["categories"] = [{"id": i, "name": name} for name, i in cat2id.items()]
    # Test:
    coco_api = COCO()
    coco_api.dataset = coco_anno
    coco_api.createIndex()
    return coco_anno