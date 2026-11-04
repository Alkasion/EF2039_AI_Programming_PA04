"""Shared helpers: paths, config access, and seeding."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_experiment(cfg, name):
    for exp in cfg["experiments"]:
        if exp["name"] == name:
            return exp
    known = ", ".join(e["name"] for e in cfg["experiments"])
    raise KeyError(f"Unknown experiment '{name}'. Known: {known}")


def per_class_count(exp, class_name):
    spec = exp["train_per_class"]
    if class_name in spec:
        return int(spec[class_name])
    if "default" in spec:
        return int(spec["default"])
    raise KeyError(f"No train count for class '{class_name}' in {exp['name']}")


def set_seed(seed):
    """Seed every RNG we touch so splits and training are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # not supported on every backend
        pass


def splits_dir(cfg):
    return ROOT / cfg["data"]["splits_dir"]


def results_dir(exp_name=None):
    d = ROOT / "results" if exp_name is None else ROOT / "results" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(exp_name):
    d = ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exp_name}.keras"
