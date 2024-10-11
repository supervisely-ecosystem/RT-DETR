from mmengine.registry import PARAM_SCHEDULERS
from mmengine.optim.optimizer import OptimWrapper
from src.train_parameters import TrainParameters


class DummyOptimWrapper(OptimWrapper):
    @property
    def param_groups(self):
        return self._param_groups

    def __init__(self, start_lr) -> None:
        self._param_groups = [{"lr": start_lr}]

    def get(self):
        return self.param_groups[0]["lr"]


def step_by_epoch(schedulers):
    for scheduler in schedulers:
        if scheduler.by_epoch:
            scheduler.step()


def step_by_iter(schedulers):
    for scheduler in schedulers:
        if not scheduler.by_epoch:
            scheduler.step()


def test_schedulers(schedulers_cfg: list, start_lr, dataloader_len, total_epochs):
    optim_wrapper = DummyOptimWrapper(start_lr)

    schedulers = []
    for scheduler in schedulers_cfg:
        default_end = (
            total_epochs if scheduler.get("by_epoch", True) else dataloader_len * total_epochs
        )
        scheduler.setdefault("end", default_end)
        schedulers.append(
            PARAM_SCHEDULERS.build(
                scheduler, default_args=dict(optimizer=optim_wrapper, epoch_length=dataloader_len)
            )
        )

    lrs = []
    lrs.append(optim_wrapper.get())
    for ep in range(total_epochs):
        for i in range(dataloader_len):
            step_by_iter(schedulers)
            lrs.append(optim_wrapper.get())
        step_by_epoch(schedulers)
        lrs[-1] = optim_wrapper.get()
    lrs.pop(-1)
    x = list(range(1, len(lrs) + 1))
    return x, lrs


def get_param_scheduler(params: TrainParameters):
    param_scheduler = []
    if params.warmup_iters:
        warmup = dict(
            type="LinearLR",
            start_factor=params.warmup_ratio,
            by_epoch=False,
            begin=0,
            end=params.warmup_iters,
        )
        param_scheduler.append(warmup)
    if params.scheduler:
        if params.scheduler["by_epoch"] is False:
            params.scheduler["begin"] = params.warmup_iters
        param_scheduler.append(params.scheduler)
    return param_scheduler
