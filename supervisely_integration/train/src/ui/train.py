import os
from pathlib import Path

import supervisely as sly
import torch
from mmdet.registry import RUNNERS
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config, ConfigDict
from mmengine.visualization import Visualizer
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DoneLabel,
    Empty,
    FolderThumbnail,
    Progress,
    ReportThumbnail,
    SlyTqdm,
    Text,
)
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.inference import SessionJSON

import src.sly_globals as g
import src.ui.models as models_ui
import src.workflow as w

# register modules (don't remove):
from src import sly_dataset, sly_hook, sly_imgaugs, sly_utils
from src.project_cached import download_project
from src.serve import MMDetectionModel
from src.train_parameters import TrainParameters
from src.ui.augmentations import get_selected_aug
from src.ui.classes import classes
from src.ui.graphics import add_classwise_metric, monitoring
from src.ui.hyperparameters import (  # run_speedtest_checkbox,
    run_model_benchmark_checkbox,
    update_params_with_widgets,
)
from src.ui.task import task_selector
from src.ui.train_val_split import dump_train_val_splits, splits

root_source_path = str(Path(__file__).parents[1])
app_data_dir = os.path.join(root_source_path, "tempfiles")


def get_task():
    if "segmentation" in task_selector.get_value().lower():
        return "instance_segmentation"
    else:
        return "object_detection"


def set_device_env(device_name: str):
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device_id = device_name.split(":")[1].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def get_train_params(cfg) -> TrainParameters:
    task = get_task()
    selected_classes = classes.get_selected_classes()
    augs_config_path = get_selected_aug()

    # create params from config
    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, g.app_dir)

    # update params with UI
    update_params_with_widgets(params)
    params.add_classwise_metric = len(selected_classes) <= g.MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC
    return params


def prepare_model():
    # download custom model if needed
    # returns config path and weights path
    if models_ui.is_pretrained_model_selected():
        selected_model = models_ui.get_selected_pretrained_model()
        config_path = selected_model["config"]
        weights_path_or_url = selected_model["weights"]
    else:
        remote_weights_path = models_ui.get_selected_custom_path()
        weights_path_or_url, config_path = sly_utils.download_custom_model(remote_weights_path)
    return config_path, weights_path_or_url


def add_metadata(cfg: Config):
    is_pretrained = models_ui.is_pretrained_model_selected()

    if not is_pretrained and not hasattr(cfg, "sly_metadata"):
        # realy custom model
        sly.logger.warn(
            "There are no sly_metadata in config, seems the custom model wasn't trained in Supervisely."
        )
        cfg.sly_metadata = {
            "model_name": "custom",
            "architecture_name": "custom",
            "task_type": get_task(),
        }

    if is_pretrained:
        selected_model = models_ui.get_selected_pretrained_model()
        metadata = {
            "model_name": selected_model["name"],
            "architecture_name": models_ui.get_selected_arch_name(),
            "task_type": get_task(),
        }
    else:
        metadata = cfg.sly_metadata

    metadata["project_id"] = g.PROJECT_ID
    metadata["project_name"] = g.project_info.name

    cfg.sly_metadata = ConfigDict(metadata)


def train():
    use_cache = g.USE_CACHE
    # download dataset
    project_dir = f"{g.app_dir}/sly_project"
    download_project(
        api=g.api,
        project_id=g.PROJECT_ID,
        project_dir=project_dir,
        use_cache=use_cache,
        progress=iter_progress,
    )

    # prepare split files
    try:
        dump_train_val_splits(project_dir)
    except Exception:
        if not use_cache:
            raise
        sly.logger.warn(
            "Failed to dump train/val splits. Trying to re-download project.", exc_info=True
        )
        download_project(
            api=g.api,
            project_id=g.PROJECT_ID,
            project_dir=project_dir,
            use_cache=False,
            progress=iter_progress,
        )
        dump_train_val_splits(project_dir)

    # prepare model files
    iter_progress(message="Preparing the model...", total=1)
    config_path, weights_path_or_url = prepare_model()

    w.workflow_input(g.api, g.PROJECT_ID)

    # create config
    cfg = Config.fromfile(config_path)
    params = get_train_params(cfg)

    # set device
    # set_device_env(params.device_name)
    # doesn't work :(
    # maybe because of torch has been imported earlier and it already read CUDA_VISIBLE_DEVICES

    ### TODO: debug
    # params.checkpoint_interval = 5
    # params.save_best = False
    # params.val_interval = 1
    # params.num_workers = 0
    # params.input_size = (409, 640)
    ###

    # If we won't do this, restarting the training will throw a error
    Visualizer._instance_dict.clear()
    DetLocalVisualizer._instance_dict.clear()

    # create config from params
    train_cfg = params.update_config(cfg)

    # update load_from with custom_weights_path
    if params.load_from and weights_path_or_url:
        train_cfg.load_from = weights_path_or_url

    # add sly_metadata
    add_metadata(train_cfg)

    # show classwise chart
    if params.add_classwise_metric:
        add_classwise_metric(classes.get_selected_classes())
        sly.logger.debug("Added classwise metrics")

    # update globals
    config_name = config_path.split("/")[-1]
    g.config_name = config_name
    g.params = params

    # clean work_dir
    if sly.fs.dir_exists(params.work_dir):
        sly.fs.remove_dir(params.work_dir)

    # TODO: debug
    # train_cfg.dump("debug_config.py")

    iter_progress(message="Preparing the model...", total=1)

    # Its grace, the Runner!
    try:
        runner = RUNNERS.build(train_cfg)
    except AttributeError:
        sly.logger.error(
            "Failed to build runner, it may be related to the incorrect "
            "frozen_stages parameter in the config or other parameters."
        )
        raise

    with g.app.handle_stop():
        runner.train()

    if g.stop_training is True:
        sly.logger.info("The training is stopped.")

    epoch_progress.hide()

    # uploading checkpoints and data
    # TODO: params.experiment_name
    if params.augs_config_path is not None:
        sly_utils.save_augs_config(params.augs_config_path, params.work_dir)
    if g.api.task_id is not None:
        sly_utils.save_open_app_lnk(params.work_dir)
    out_path = sly_utils.upload_artifacts(
        params.work_dir,
        params.experiment_name,
        get_task(),
        iter_progress,
    )

    # ------------------------------------- Model Benchmark ------------------------------------- #
    model_benchmark_done = False
    if run_model_benchmark_checkbox.is_checked():
        try:
            task_type = get_task().replace("_", " ")
            if task_type in [
                sly.nn.TaskType.INSTANCE_SEGMENTATION,
                sly.nn.TaskType.OBJECT_DETECTION,
            ]:
                creating_report.show()

                best_filename = None
                best_checkpoints = []
                for file_name in os.listdir(params.work_dir):
                    if file_name.endswith(".pth"):
                        if file_name.startswith("best_"):
                            best_checkpoints.append(file_name)

                if len(best_checkpoints) == 0:
                    raise ValueError("Best model checkpoint not found")
                elif len(best_checkpoints) > 1:
                    best_checkpoints = sorted(best_checkpoints, key=lambda x: x, reverse=True)

                best_filename = best_checkpoints[0]
                sly.logger.info(f"Creating the report for the best model: {best_filename!r}")

                # 0. Serve trained model
                m = MMDetectionModel(
                    model_dir=params.work_dir,
                    use_gui=False,
                    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yml"),
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sly.logger.info(f"Using device: {device}")

                checkpoint_path = os.path.join(out_path, best_filename)
                config_path = os.path.join(out_path, "config.py")
                deploy_params = dict(
                    device=device,
                    model_source="Custom models",
                    task_type=task_type,
                    checkpoint_name=best_filename,
                    checkpoint_url=checkpoint_path,
                    config_url=config_path,
                    arch_type=train_cfg.sly_metadata.architecture_name,
                )
                m._load_model(deploy_params)
                m.serve()
                session = SessionJSON(g.api, session_url="http://localhost:8000")
                if sly.fs.dir_exists(app_data_dir + "/benchmark"):
                    sly.fs.remove_dir(app_data_dir + "/benchmark")

                # 1. Init benchmark (todo: auto-detect task type)
                benchmark_dataset_ids = None
                benchmark_images_ids = None
                train_dataset_ids = None
                train_images_ids = None

                split_method = splits._content.get_active_tab()
                train_set, val_set = splits.get_splits()

                if split_method == "Based on datasets":
                    benchmark_dataset_ids = splits._val_ds_select.get_selected_ids()
                    train_dataset_ids = splits._train_ds_select.get_selected_ids()
                else:
                    dataset_infos = g.api.dataset.get_list(g.PROJECT_ID)

                    def get_image_infos_by_split(split: list):
                        ds_infos_dict = {ds_info.name: ds_info for ds_info in dataset_infos}
                        image_names_per_dataset = {}
                        for item in split:
                            image_names_per_dataset.setdefault(item.dataset_name, []).append(
                                item.name
                            )
                        image_infos = []
                        for dataset_name, image_names in image_names_per_dataset.items():
                            ds_info = ds_infos_dict[dataset_name]
                            image_infos.extend(
                                g.api.image.get_list(
                                    ds_info.id,
                                    filters=[
                                        {
                                            "field": "name",
                                            "operator": "in",
                                            "value": image_names,
                                        }
                                    ],
                                )
                            )
                        return image_infos

                    val_image_infos = get_image_infos_by_split(val_set)
                    train_image_infos = get_image_infos_by_split(train_set)
                    benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                    train_images_ids = [img_info.id for img_info in train_image_infos]

                if task_type == sly.nn.TaskType.OBJECT_DETECTION:
                    bm = ObjectDetectionBenchmark(
                        g.api,
                        g.project_info.id,
                        output_dir=app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        classes_whitelist=classes.get_selected_classes(),
                    )
                elif task_type == sly.nn.TaskType.INSTANCE_SEGMENTATION:
                    bm = InstanceSegmentationBenchmark(
                        g.api,
                        g.project_info.id,
                        output_dir=app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        classes_whitelist=classes.get_selected_classes(),
                    )
                else:
                    raise ValueError(
                        f"Model benchmark for task type {task_type} is not implemented (coming soon)"
                    )

                train_info = {
                    "app_session_id": sly.env.task_id(),
                    "train_dataset_ids": train_dataset_ids,
                    "train_images_ids": train_images_ids,
                    "images_count": len(train_set),
                }
                bm.train_info = train_info

                # 2. Run inference
                bm.run_inference(session)

                # 3. Pull results from the server
                gt_project_path, dt_project_path = bm.download_projects(save_images=False)

                # 4. Evaluate
                bm._evaluate(gt_project_path, dt_project_path)

                # 5. Upload evaluation results
                eval_res_dir = sly_utils.get_eval_results_dir_name(
                    g.api, sly.env.task_id(), g.project_info
                )
                bm.upload_eval_results(eval_res_dir + "/evaluation/")

                # # 6. Speed test
                # if run_speedtest_checkbox.is_checked():
                #     bm.run_speedtest(session, g.project_info.id)
                #     bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

                # 7. Prepare visualizations, report and
                bm.visualize()
                remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
                report = bm.upload_report_link(remote_dir)

                # 8. UI updates
                benchmark_report_template = g.api.file.get_info_by_path(
                    sly.env.team_id(), remote_dir + "template.vue"
                )
                model_benchmark_done = True
                creating_report.hide()
                model_benchmark_report.set(benchmark_report_template)
                model_benchmark_report.show()
                model_benchmark_pbar.hide()
                sly.logger.info(
                    f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
                )
                sly.logger.info(
                    f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
                )
        except Exception as e:
            sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            creating_report.hide()
            model_benchmark_pbar.hide()
            try:
                if bm.dt_project_info:
                    g.api.project.remove(bm.dt_project_info.id)
                if bm.diff_project_info:
                    g.api.project.remove(bm.diff_project_info.id)
            except Exception as re:
                pass

    if not model_benchmark_done:
        benchmark_report_template = None
    # ----------------------------------------------- - ---------------------------------------------- #

    w.workflow_output(g.api, g.mmdet_generated_metadata, benchmark_report_template)

    # set task results
    if sly.is_production():
        remote_file_path = g.sly_mmdet3.get_config_path(out_path)
        file_info = g.api.file.get_info_by_path(g.TEAM_ID, remote_file_path)

        # add link to artifacts
        folder_thumb.set(info=file_info)
        folder_thumb.show()

        # show success message
        success_msg.show()

        # disable buttons after training
        start_train_btn.hide()
        stop_train_btn.hide()

        # set link to artifacts in ws tasks
        g.api.task.set_output_directory(sly.env.task_id(), file_info.id, out_path)
        g.app.stop()


start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epochs")
epoch_progress.hide()

iter_progress = Progress("Iterations", hide_on_finish=False)
iter_progress.hide()

success_msg = DoneLabel("Training completed. Training artifacts were uploaded to Team Files.")
success_msg.hide()

folder_thumb = FolderThumbnail()
folder_thumb.hide()

model_benchmark_report = ReportThumbnail()
model_benchmark_report.hide()

model_benchmark_pbar = SlyTqdm()
creating_report = Text(status="info", text="Creating report on model...")
creating_report.hide()


btn_container = Container(
    [start_train_btn, stop_train_btn, Empty()],
    "horizontal",
    overflow="wrap",
    fractions=[1, 1, 10],
    gap=1,
)

container = Container(
    [
        success_msg,
        folder_thumb,
        creating_report,
        model_benchmark_report,
        btn_container,
        epoch_progress,
        iter_progress,
        model_benchmark_pbar,
        monitoring.compile_monitoring_container(True),
    ]
)

card = Card(
    "Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=container,
)
card.lock("Select hyperparameters.")


def start_train():
    g.stop_training = False
    monitoring.container.show()
    stop_train_btn.enable()
    # epoch_progress.show()
    iter_progress.show()
    train()


def stop_train():
    g.stop_training = True
    stop_train_btn.disable()
