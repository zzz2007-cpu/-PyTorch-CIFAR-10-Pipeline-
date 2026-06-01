from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "run_name": None,
    "seed": 42,
    "data": {
        "data_dir": "./datasets",
        "batch_size": 128,
        "num_workers": 2,
        "data_aug": False,
        "label_noise_rate": 0.0,
        "label_noise_seed": 42,
    },
    "model": {
        "name": "simple_cnn",
    },
    "train": {
        "epochs": 20,
        "output_dir": "./outputs",
    },
    "optimizer": {
        "name": "sgd",
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    },
    "scheduler": {
        "name": "none",
        "step_size": 10,
        "gamma": 0.1,
        "t_max": None,
    },
}


def deep_update(base: dict, updates: dict) -> dict:
    result = deepcopy(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = deep_update(DEFAULT_CONFIG, user_config)
    config["config_path"] = str(path)
    return config
