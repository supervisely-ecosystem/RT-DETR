import os
import shutil
from pathlib import Path

from requests_toolbelt import MultipartEncoderMonitor
from tqdm import tqdm

import src.sly_globals as g
import supervisely as sly
from supervisely.app.widgets import Progress


def download_custom_config(remote_weights_path: str):
    # # download config_xxx.py
    # save_dir = remote_weights_path.split("checkpoints")
    # files = g.api.file.listdir(g.TEAM_ID, save_dir)
    # # find config by name in save_dir
    # remote_config_path = [f for f in files if f.endswith(".py")]
    # assert len(remote_config_path) > 0, f"Can't find config in {save_dir}."

    # download config.py
    remote_dir = os.path.dirname(remote_weights_path)
    remote_config_path = remote_dir + "/config.py"
    config_name = remote_config_path.split("/")[-1]
    config_path = g.app_dir + f"/{config_name}"
    g.api.file.download(g.TEAM_ID, remote_config_path, config_path)
    return config_path


def download_custom_model_weights(remote_weights_path: str):
    # download .pth
    file_name = os.path.basename(remote_weights_path)
    weights_path = g.app_dir + f"/{file_name}"
    g.api.file.download(g.TEAM_ID, remote_weights_path, weights_path)
    return weights_path


def download_custom_model(remote_weights_path: str):
    config_path = download_custom_config(remote_weights_path)
    weights_path = download_custom_model_weights(remote_weights_path)
    return weights_path, config_path


def upload_artifacts(
    work_dir: str,
    experiment_name: str = None,
    task_type: str = None,
    progress_widget: Progress = None,
):
    task_id = g.api.task_id or ""
    paths = [path for path in os.listdir(work_dir) if path.endswith(".py")]
    assert len(paths) > 0, "Can't find config file saved during training."
    assert len(paths) == 1, "Found more then 1 .py file"
    cfg_path = f"{work_dir}/{paths[0]}"
    shutil.move(cfg_path, f"{work_dir}/config.py")

    # rm symlink
    sly.fs.silent_remove(f"{work_dir}/last_checkpoint")

    if not experiment_name:
        experiment_name = f"{g.config_name.split('.py')[0]}"
    sly.logger.debug("Uploading checkpoints to Team Files...")

    if sly.is_community():
        convert_and_resize_images(work_dir)

    if progress_widget:
        progress_widget.show()
        size_bytes = sly.fs.get_directory_size(work_dir)
        pbar = progress_widget(
            message="Uploading to Team Files...",
            total=size_bytes,
            unit="b",
            unit_divisor=1024,
            unit_scale=True,
        )

        def cb(monitor: MultipartEncoderMonitor):
            pbar.update(int(monitor.bytes_read - pbar.n))

    else:
        cb = None

    framework_folder = g.sly_mmdet3.framework_folder
    remote_artifacts_dir = f"{framework_folder}/{task_id}_{experiment_name}"
    remote_weights_dir = g.sly_mmdet3.get_weights_path(remote_artifacts_dir)
    remote_config_dir = g.sly_mmdet3.get_config_path(remote_artifacts_dir)

    out_path = g.api.file.upload_directory(
        g.TEAM_ID,
        work_dir,
        remote_artifacts_dir,
        progress_size_cb=cb,
    )
    progress_widget.hide()

    # generate metadata
    g.mmdet_generated_metadata = g.sly_mmdet3.generate_metadata(
        app_name=g.sly_mmdet3.app_name,
        task_id=task_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=g.sly_mmdet3.weights_ext,
        project_name=g.api.project.get_info_by_id(g.PROJECT_ID).name,
        task_type=task_type,
        config_path=remote_config_dir,
    )

    return out_path


def convert_and_resize_images(work_dir: str):
    import cv2

    MAX_DIM = 2048

    for root, _, files in os.walk(work_dir):
        for file in files:
            if file.endswith(".png"):
                png_img_path = Path(root) / file
                parent_dir = png_img_path.parent
                if parent_dir.name == "vis_image":
                    jpg_img_path = png_img_path.with_suffix(".jpg")
                    img = cv2.imread(png_img_path.as_posix())
                    h, w = img.shape[:2]
                    if h > MAX_DIM or w > MAX_DIM:
                        out_size = (MAX_DIM, -1) if h > MAX_DIM else (-1, MAX_DIM)
                        img = sly.image.resize(img, out_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    sly.image.write(jpg_img_path.as_posix(), img)
                    sly.fs.silent_remove(png_img_path.as_posix())
                    img = None


def download_project(progress_widget):
    project_dir = f"{g.app_dir}/sly_project"

    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)

    n = get_images_count()
    with progress_widget(message="Downloading project...", total=n) as pbar:
        sly.Project.download(g.api, g.PROJECT_ID, project_dir, progress_cb=pbar.update)

    return project_dir


def get_images_count():
    return g.IMAGES_COUNT


def save_augs_config(augs_config_path: str, work_dir: str):
    sly.fs.copy_file(augs_config_path, work_dir + "/augmentations.json")


def save_open_app_lnk(work_dir: str):
    with open(work_dir + "/open_app.lnk", "w") as f:
        f.write(f"{g.api.server_address}/apps/sessions/{g.api.task_id}")


def get_eval_results_dir_name(api, task_id, project_info):
    task_info = api.task.get_info_by_id(task_id)
    task_dir = f"{task_id}_{task_info['meta']['app']['name']}"
    eval_res_dir = f"/model-benchmark/evaluation/{project_info.id}_{project_info.name}/{task_dir}/"
    eval_res_dir = api.storage.get_free_dir_name(sly.env.team_id(), eval_res_dir)
    return eval_res_dir
