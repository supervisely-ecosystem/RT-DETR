import os
import torch
from src.solver import DetSolver
from src.core import YAMLConfig
from checkpoints import checkpoints


def train(model: str, finetune: bool, config_path: str):

    if finetune:
        checkpoint_url = checkpoints[model]
        name = os.path.basename(checkpoint_url)
        checkpoint_path = f"models/{name}"
        if not os.path.exists(checkpoint_path):
            torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)
        tuning = checkpoint_path
    else:
        tuning = ''

    cfg = YAMLConfig(
        config_path,
        # resume='',
        tuning=tuning
    )

    import yaml
    with open("output/config.yml", 'w') as f:
        yaml.dump(cfg.yaml_cfg, f)

    solver = DetSolver(cfg)
    solver.fit()
