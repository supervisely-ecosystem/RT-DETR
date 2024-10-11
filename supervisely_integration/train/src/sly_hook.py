from typing import Dict, Optional, Sequence
from mmdet.registry import HOOKS
from mmengine.hooks import Hook  # LoggerHook, CheckpointHook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner
import supervisely as sly
import torch

import src.sly_globals as g
import src.ui.train as train_ui
from src.ui.graphics import monitoring


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        chart_update_interval: int = 5,
    ):
        self.chart_update_interval = chart_update_interval
        self.epoch_progress = None
        self.iter_progress = None

        if train_ui.get_task() == "instance_segmentation":
            self.task = "segm"
        else:
            self.task = "bbox"

    def before_train(self, runner: Runner) -> None:
        train_ui.epoch_progress.show()
        self.epoch_progress = train_ui.epoch_progress(message="Epochs", total=runner.max_epochs)
        self.iter_progress = train_ui.iter_progress(
            message="Iterations", total=len(runner.train_dataloader)
        )

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: dict = None
    ) -> None:
        # Check nans
        if not torch.isfinite(outputs["loss"]):
            sly.logger.warn("The loss is NaN.")
            outputs["loss"] = torch.tensor(0.0, device=outputs["loss"].device)

        # Update progress bars
        self.iter_progress.update(1)

        # Update train charts
        if self.every_n_train_iters(runner, self.chart_update_interval):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
            i = runner.iter + 1
            monitoring.add_scalar("train", "Loss", "loss", i, tag["loss"])
            monitoring.add_scalar("train", "Learning Rate", "lr", i, tag["lr"])

        # Stop training
        if g.app.is_stopped() or g.stop_training:
            sly.logger.info("The training is stopped.")
            raise g.app.StopException("This error is expected")

    def after_val_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        if g.app.is_stopped():
            raise g.app.StopException("This error is expected")
        return super().after_val_iter(runner, batch_idx, data_batch, outputs)

    def after_train_epoch(self, runner: Runner) -> None:
        # Update progress bars
        self.epoch_progress.update(1)
        self.iter_progress = train_ui.iter_progress(
            message="Iterations", total=len(runner.train_dataloader)
        )

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        if not metrics:
            return

        # Add mAP metrics
        metric_keys = [f"coco/{self.task}_{metric}" for metric in g.COCO_MTERIC_KEYS]
        for metric_key, metric_name in zip(metric_keys, g.COCO_MTERIC_KEYS):
            value = metrics[metric_key]
            monitoring.add_scalar("val", "Metrics", metric_name, runner.epoch, value)

        # Add classwise metrics
        if g.params.add_classwise_metric:
            # colors = runner.val_dataloader.dataset.metainfo["palette"]
            classwise_metrics = {
                k.split("_precision")[0][5:]: v
                for k, v in metrics.items()
                if k.endswith("_precision")
            }
            for class_name, value in classwise_metrics.items():
                monitoring.add_scalar("val", "Classwise mAP", class_name, runner.epoch, value)
