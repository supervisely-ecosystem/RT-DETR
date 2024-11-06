"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import pathlib
import sys
from typing import Iterable

import torch
import torch.amp
import utils
from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict
from src.misc.sly_logger import LOGS

from supervisely.app.widgets import Progress


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()

    progress_bar_iters: Progress = kwargs.get("progress_bar_iters")

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Epoch: [{}]".format(epoch)
    print_freq = kwargs.get("print_freq", 10)
    LOGS.epoch = epoch
    grad_norm = None

    ema = kwargs.get("ema", None)
    scaler = kwargs.get("scaler", None)
    lr_warmup = kwargs.get("lr_warmup", None)
    lr_scheduler = kwargs.get("lr_scheduler", None)

    with progress_bar_iters(message=f"Iterations", total=len(data_loader)) as iters_pbar:
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets)

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets)

                loss = sum(loss_dict.values())
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)

                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()

                if max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()

            # ema
            if ema is not None:
                ema.update(model)

            # lr scheduler
            if lr_warmup is not None:
                lr_warmup.step()
            if lr_scheduler is not None and not utils.is_by_epoch(lr_scheduler):
                lr_scheduler.step()

            loss_dict_reduced = reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # import wandb

            # from supervisely.train import train_logger

            # # wandb.log({
            # #     "Train/loss":
            # #     })
            # train_logger.log({"Train/loss": loss_value.item()})

            # Update supervisely logs
            LOGS.loss = loss_value.item()
            LOGS.grad_norm = grad_norm.item() if grad_norm is not None else None
            lrs = {}
            for i, param_group in enumerate(optimizer.param_groups):
                lrs[f"lr{i}"] = param_group["lr"]
            LOGS.lrs = lrs
            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                LOGS.cuda_memory = torch.cuda.max_memory_allocated() / MB
            LOGS.iter_idx += 1
            iters_pbar.update(1)

    # Draw training loss
    LOGS.log_train_iter()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    imgs, predictions = [], []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

        # Draw predictions
        if len(imgs) < LOGS.n_preview_imgs:
            for sample, result, orig_target_size in zip(samples, results, orig_target_sizes):
                img, prediction = utils.prepare_result(sample, result, orig_target_size, base_ds)
                imgs.append(img)
                predictions.append(prediction)
                if len(imgs) == LOGS.n_preview_imgs:
                    break
        if len(imgs) == LOGS.n_preview_imgs:
            LOGS.log_preview(imgs, predictions)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            class_ap, class_ar = utils.collect_per_class_metrics(coco_evaluator, base_ds)
            LOGS.log_evaluation(stats["coco_eval_bbox"], class_ap, class_ar)
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator
