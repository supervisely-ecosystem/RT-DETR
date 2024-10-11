import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import cv2
import numpy as np
import pkg_resources
import supervisely as sly
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmengine import Config
from mmengine.structures import InstanceData
from supervisely.nn.inference import CheckpointInfo, RuntimeType, TaskType
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask

root_source_path = str(Path(__file__).parents[1])
app_source_path = str(Path(__file__).parents[1])

configs_dir = os.path.join(root_source_path, "configs")
mmdet_ver = pkg_resources.get_distribution("mmdet").version
if os.path.isdir(f"/tmp/mmdet/mmdetection-{mmdet_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmdetection version {mmdet_ver}...")
    shutil.copytree(f"/tmp/mmdet/mmdetection-{mmdet_ver}/configs", configs_dir)


class MMDetectionModel(sly.nn.inference.InstanceSegmentation):

    def load_model_meta(self, model_source: str, cfg: Config, checkpoint_name: str = None):
        def set_common_meta(classes, task_type):
            obj_classes = [
                sly.ObjClass(
                    name,
                    (
                        sly.Bitmap
                        if task_type == sly.nn.TaskType.INSTANCE_SEGMENTATION
                        else sly.Rectangle
                    ),
                )
                for name in classes
            ]
            self.selected_model_name = cfg.sly_metadata.architecture_name
            self.checkpoint_name = checkpoint_name
            self.dataset_name = cfg.dataset_type
            self.class_names = classes
            self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
            self._get_confidence_tag_meta()

        if cfg.dataset_type != "SuperviselyDatasetSplit":

            # classes from .pth
            classes = self.model.dataset_meta.get("classes", [])
            if classes == []:
                # classes from config
                dataset_class_name = cfg.dataset_type
                dataset_meta = DATASETS.module_dict[dataset_class_name].METAINFO
                classes = dataset_meta.get("classes", [])
                if classes == []:
                    raise ValueError("Classes not found in the .pth and config file")
            self.dataset_name = cfg.dataset_type
            set_common_meta(classes, self.task_type)

        else:
            classes = cfg.train_dataloader.dataset.selected_classes
            self.checkpoint_name = checkpoint_name
            self.dataset_name = cfg.sly_metadata.project_name
            self.task_type = cfg.sly_metadata.task_type.replace("_", " ")
            set_common_meta(classes, self.task_type)

        self.model.test_cfg["score_thr"] = 0.45  # default confidence_threshold

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["object detection", "instance segmentation"],
        checkpoint_name: str,
        checkpoint_url: str,
        config_url: str,
        arch_type: str = None,
    ):
        """
        Load model method is used to deploy model.

        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param task_type: The type of the computer vision task the model is designed for.
        :type task_type: Literal["object detection", "instance segmentation"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model checkpoint can be downloaded.
        :type checkpoint_url: str
        :param config_url: The URL where the model config can be downloaded.
        :type config_url: str
        :param arch_type: The architecture type of the model.
        :type arch_type: str
        """
        self.device = device
        self.task_type = task_type
        self.runtime = RuntimeType.PYTORCH

        checkpoint_info = self.api.file.get_info_by_path(sly.env.team_id(), checkpoint_url)
        # config_info = self.api.file.get_info_by_path(sly.env.team_id(), config_url)

        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if not sly.fs.file_exists(local_weights_path):
            self.api.file.download(sly.env.team_id(), checkpoint_url, local_weights_path)

        local_config_path = os.path.join(configs_dir, "custom", "config.py")
        if sly.fs.file_exists(local_config_path):
            sly.fs.silent_remove(local_config_path)
        self.api.file.download(sly.env.team_id(), config_url, local_config_path)
        if not sly.fs.file_exists(local_config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_url}. "
                "Config should be placed in the same directory as the checkpoint file."
            )

        try:
            cfg = Config.fromfile(local_config_path)
            if "pretrained" in cfg.model:
                cfg.model.pretrained = None
            elif "init_cfg" in cfg.model.backbone:
                cfg.model.backbone.init_cfg = None
            cfg.model.train_cfg = None
            self.model = init_detector(
                cfg, checkpoint=local_weights_path, device=device, palette=[]
            )
            self.load_model_meta(model_source, cfg, checkpoint_name)
        except KeyError as e:
            raise KeyError(f"Error loading config file: {local_config_path}. Error: {e}")

        checkpoint_name = os.path.splitext(checkpoint_name)[0]
        if model_source == "Pretrained models":
            custom_checkpoint_path = None
        else:
            custom_checkpoint_path = checkpoint_url
            checkpoint_url = self.api.file.get_url(checkpoint_info.id)

        model_name = cfg.sly_metadata.model_name
        self.checkpoint_info = CheckpointInfo(
            model_name=model_name,
            checkpoint_name=checkpoint_name,
            architecture=self.selected_model_name,
            model_source=model_source,
            checkpoint_url=checkpoint_url,
            custom_checkpoint_path=custom_checkpoint_path,
        )

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionBBox, PredictionMask]]:
        # set confidence_threshold
        conf_tresh = settings.get("confidence_threshold", 0.45)
        if conf_tresh:
            # TODO: may be set recursively?
            self.model.test_cfg["score_thr"] = conf_tresh

        # set nms_iou_thresh
        nms_tresh = settings.get("nms_iou_thresh", 0.65)
        if nms_tresh:
            test_cfg = self.model.test_cfg
            if hasattr(test_cfg, "nms"):
                test_cfg["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rcnn") and hasattr(test_cfg["rcnn"], "nms"):
                test_cfg["rcnn"]["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rpn") and hasattr(test_cfg["rpn"], "nms"):
                test_cfg["rpn"]["nms"]["iou_threshold"] = nms_tresh

        # inference
        result: DetDataSample = inference_detector(self.model, image_path)
        preds = result.pred_instances.cpu().numpy()

        # collect predictions
        predictions = []
        for pred in preds:
            pred: InstanceData
            score = float(pred.scores[0])
            if conf_tresh is not None and score < conf_tresh:
                # filter by confidence
                continue
            class_name = self.class_names[pred.labels.astype(int)[0]]
            if self.task_type == "object detection":
                x1, y1, x2, y2 = pred.bboxes[0].astype(int).tolist()
                tlbr = [y1, x1, y2, x2]
                sly_pred = PredictionBBox(class_name=class_name, bbox_tlbr=tlbr, score=score)
            else:
                if pred.get("masks") is None:
                    raise Exception(
                        f'The model "{self.checkpoint_name}" can\'t predict masks. Please, try another model.'
                    )
                mask = pred.masks[0]
                sly_pred = PredictionMask(class_name=class_name, mask=mask, score=score)
            predictions.append(sly_pred)

        # TODO: debug
        # ann = self._predictions_to_annotation(image_path, predictions)
        # img = sly.image.read(image_path)
        # ann.draw_pretty(img, thickness=2, opacity=0.4, output_path="test.jpg")
        return predictions


# custom_settings_path = os.path.join(app_source_path, "custom_settings.yml")
