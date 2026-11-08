"""Shared helpers: paths, config access, seeding and device selection."""
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


def set_seed(seed):
    """Seed every RNG we touch so splits and training are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """CUDA GPU > Apple MPS > CPU. Override with PA04_DEVICE=cpu|mps|cuda."""
    import torch

    forced = os.environ.get("PA04_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def raw_dir(cfg):
    return ROOT / cfg["data"]["raw_dir"]


def processed_dir(cfg):
    return ROOT / cfg["data"]["processed_dir"]


def splits_dir(cfg):
    return ROOT / cfg["data"]["splits_dir"]


def results_dir(exp_name=None):
    d = ROOT / "results" if exp_name is None else ROOT / "results" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_name(exp_name, seed):
    """Results of one training run live in results/<exp>/seed<seed>/."""
    return f"{exp_name}/seed{seed}"


def model_path(exp_name, seed):
    d = ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exp_name}_seed{seed}.pt"
