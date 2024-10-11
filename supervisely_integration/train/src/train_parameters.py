from mmengine import Config, ConfigDict
import multiprocessing
from copy import deepcopy
import supervisely as sly


class TrainParameters:
    ACCEPTABLE_TASKS = ["object_detection", "instance_segmentation"]

    def __init__(self) -> None:
        # required
        self.task = None
        self.selected_classes = None
        self.augs_config_path = None
        self.app_dir = None
        self.work_dir = None

        # general
        self.epoch_based_train = True
        self.total_epochs = 20
        self.val_interval = 1 if self.epoch_based_train else 1000
        self.batch_size_train = 2
        self.batch_size_val = 1
        self.input_size = (1000, 600)
        self.num_workers = min(4, multiprocessing.cpu_count())
        self.load_from: bool = True  # load weights to continue training (path or url in config)
        self.log_interval = 50  # for text logger
        self.chart_update_interval = 1
        self.filter_images_without_gt = True
        self.experiment_name = None
        self.add_classwise_metric = True
        self.device_name = "cuda:0"  # selecting device doesn't supported now

        # checkpoints
        self.checkpoint_interval = 1
        self.max_keep_checkpoints = 3
        self.save_last = True
        self.save_best = True
        self.save_optimizer = False

        # optimizer
        self.optim_wrapper = None
        self.optimizer = dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
        self.clip_grad_norm = None

        # scheduler
        self.warmup = "linear"
        self.warmup_iters = 100  # by default it will be changed to 1 epoch
        self.warmup_ratio = 0.001
        self.scheduler = None

        # self.losses = None

    @classmethod
    def from_config(cls, config: Config):
        self = cls()
        size_from_cfg = try_get_size_from_config(config)
        if size_from_cfg:
            self.input_size = size_from_cfg
        self.optim_wrapper = config.optim_wrapper
        self.optimizer = config.optim_wrapper.optimizer
        try:
            if self.optim_wrapper.clip_grad.max_norm is not None:
                self.clip_grad_norm = self.optim_wrapper.clip_grad.max_norm
                self.optim_wrapper.clip_grad.norm_type = 2
        except AttributeError:
            self.optim_wrapper.clip_grad = None
        # TODO: load general params if it is custom config
        return self

    def init(self, task, selected_classes, augs_config_path, app_dir):
        self.task = task
        self.selected_classes = selected_classes
        self.augs_config_path = augs_config_path
        self.app_dir = app_dir
        self.work_dir = app_dir + "/work_dir"

    def update_config(self, config: Config):
        cfg = deepcopy(config)
        assert self.is_inited(), "TrainParameters: wrong initialization parameters."

        # change model num_classes
        num_classes = len(self.selected_classes)
        modify_num_classes_recursive(cfg.model, num_classes)
        modify_num_classes_recursive(cfg.model, num_classes, "num_things_classes")
        modify_num_classes_recursive(cfg.model, 0, "num_stuff_classes")

        # can we just remove batch_augments?
        if hasattr(cfg.model.data_preprocessor, "batch_augments"):
            cfg.model.data_preprocessor.batch_augments = None

        # Mask2Former fix:
        if cfg.model.type == "Mask2Former":
            cfg.model.panoptic_head.loss_cls["class_weight"] = [1.0] * num_classes + [0.1]

        # RTMDet and YOLOX fix:
        if cfg.model.type in ["RTMDet", "YOLOX"]:
            cfg.model.data_preprocessor.pad_mask = True
            cfg.model.data_preprocessor.pad_size_divisor = 32

        if hasattr(self, "frozen_stages") and self.frozen_stages is not None:
            cfg.model.backbone.frozen_stages = self.frozen_stages
            sly.logger.debug(f"Frozen stages set to: {self.frozen_stages}")

        # pipelines
        train_pipeline, test_pipeline = get_default_pipelines(
            with_mask=self.task == "instance_segmentation"
        )
        img_aug = dict(type="SlyImgAugs", config_path=self.augs_config_path)
        idx_insert = find_index_for_imgaug(train_pipeline)  # 2 by default
        train_pipeline.insert(idx_insert, img_aug)
        train_pipeline[3]["scale"] = self.input_size
        test_pipeline[1]["scale"] = self.input_size
        if cfg.get("train_pipeline"):
            cfg.train_pipeline = train_pipeline
        if cfg.get("test_pipeline"):
            cfg.test_pipeline = test_pipeline

        # datasets
        train_dataset = dict(
            type="SuperviselyDatasetSplit",
            data_root=f"{self.app_dir}/sly_project",
            split_file=f"{self.app_dir}/train_split.json",
            task=self.task,
            selected_classes=self.selected_classes,
            filter_images_without_gt=self.filter_images_without_gt,
            pipeline=train_pipeline,
        )
        val_dataset = dict(
            type="SuperviselyDatasetSplit",
            data_root=f"{self.app_dir}/sly_project",
            split_file=f"{self.app_dir}/val_split.json",
            task=self.task,
            save_coco_ann_file=f"{self.app_dir}/val_coco_instances.json",
            selected_classes=self.selected_classes,
            filter_images_without_gt=self.filter_images_without_gt,
            pipeline=test_pipeline,
            test_mode=True,
        )

        # dataloaders
        train_dataloader, val_dataloader = get_default_dataloaders()

        train_dataloader.batch_size = self.batch_size_train
        train_dataloader.num_workers = self.num_workers
        train_dataloader.persistent_workers = self.num_workers != 0
        val_dataloader.batch_size = self.batch_size_val
        val_dataloader.num_workers = self.num_workers
        val_dataloader.persistent_workers = self.num_workers != 0

        train_dataloader.dataset = train_dataset
        val_dataloader.dataset = val_dataset

        cfg.train_dataloader = train_dataloader
        cfg.val_dataloader = val_dataloader
        cfg.test_dataloader = cfg.val_dataloader.copy()

        # evaluators
        # from mmdet.evaluation.metrics import CocoMetric
        coco_metric = "segm" if self.task == "instance_segmentation" else "bbox"

        cfg.val_evaluator = dict(
            type="CocoMetric",
            ann_file=f"{self.app_dir}/val_coco_instances.json",
            metric=coco_metric,
            classwise=self.add_classwise_metric,
        )

        cfg.test_evaluator = cfg.val_evaluator.copy()

        # train/val
        cfg.train_cfg = dict(
            by_epoch=self.epoch_based_train,
            max_epochs=self.total_epochs,
            val_interval=self.val_interval,
        )

        # hooks
        # from sly_hook import SuperviselyHook
        # from mmdet.engine.hooks import CheckInvalidLossHook, MeanTeacherHook, NumClassCheckHook
        # from mmengine.hooks import CheckpointHook
        save_best = "auto" if self.save_best else None
        cfg.default_hooks.checkpoint = dict(
            type="CheckpointHook",
            interval=self.checkpoint_interval,
            by_epoch=self.epoch_based_train,
            max_keep_ckpts=self.max_keep_checkpoints,
            save_last=self.save_last,
            save_best=save_best,
            save_optimizer=self.save_optimizer,
        )
        cfg.log_processor = dict(
            type="LogProcessor", window_size=self.chart_update_interval, by_epoch=True
        )
        cfg.default_hooks.logger["interval"] = self.log_interval
        cfg.custom_hooks = [
            dict(type="NumClassCheckHook"),
            # dict(type="CheckInvalidLossHook", interval=1),
            dict(type="SuperviselyHook", chart_update_interval=self.chart_update_interval),
        ]

        # visualization
        # from mmdet.engine.hooks import DetVisualizationHook
        # TODO: debug
        cfg.default_hooks.visualization = dict(type="DetVisualizationHook", draw=True, interval=100)

        # optimizer
        # from mmengine.optim.optimizer import OptimWrapper
        cfg.optim_wrapper.optimizer = self.optimizer
        if self.clip_grad_norm:
            cfg.optim_wrapper.clip_grad = dict(max_norm=self.clip_grad_norm)

        # scheduler
        # from mmengine.optim.scheduler import ConstantLR, LinearLR
        cfg.param_scheduler = []
        if self.warmup_iters:
            warmup = dict(
                type="LinearLR",
                start_factor=self.warmup_ratio,
                by_epoch=False,
                begin=0,
                end=self.warmup_iters,
            )
            cfg.param_scheduler.append(warmup)
        if self.scheduler:
            if self.scheduler["by_epoch"] is False:
                self.scheduler["begin"] = self.warmup_iters
            cfg.param_scheduler.append(self.scheduler)

        # TODO: loss. can we correctly change losses?

        # cfg.load_from = self.load_from  # will set later as we don't know 'weights_path' so far
        cfg.work_dir = self.work_dir
        cfg.experiment_name = self.experiment_name
        cfg.launcher = "none"
        # cfg.env_cfg.mp_cfg.mp_start_method = "spawn"
        cfg.num_classes = len(self.selected_classes)

        return cfg

    def is_inited(self):
        need_to_check = [self.task, self.selected_classes, self.work_dir]
        return all([bool(x) for x in need_to_check]) and self.task in self.ACCEPTABLE_TASKS


def modify_num_classes_recursive(d, num_classes, key="num_classes"):
    if isinstance(d, ConfigDict):
        if d.get(key) is not None:
            d[key] = num_classes
        for k, v in d.items():
            modify_num_classes_recursive(v, num_classes, key)
    elif isinstance(d, (list, tuple)):
        for v in d:
            modify_num_classes_recursive(v, num_classes, key)


def find_index_for_imgaug(pipeline):
    # return index after LoadImageFromFile and LoadAnnotations
    i1, i2 = -1, -1
    types = [p["type"] for p in pipeline]
    if "LoadImageFromFile" in types:
        i1 = types.index("LoadImageFromFile")
    if "LoadAnnotations" in types:
        i2 = types.index("LoadAnnotations")
    idx_insert = max(i1, i2)
    if idx_insert != -1:
        idx_insert += 1
    return idx_insert


def get_default_pipelines(with_mask: bool):
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=with_mask),
        # *imgagus will be here
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=with_mask),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    return train_pipeline, test_pipeline


def get_default_dataloaders():
    train_dataloader = dict(
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=None,
    )

    val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=None,
    )

    return ConfigDict(train_dataloader), ConfigDict(val_dataloader)


def try_get_size_from_config(config: Config):
    try:
        pipeline = (
            getattr(config, "train_pipeline", None) or config.train_dataloader.dataset.pipeline
        )
        for transform in pipeline:
            if transform["type"] == "Resize":
                return transform["scale"]
    except Exception as exc:
        print(f"can't get size from config: {exc}")
    return None
