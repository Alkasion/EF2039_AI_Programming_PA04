"""Reproduce every number and figure in the paper with one command.

Usage: python -m src.run_all [--skip-prepare]
"""
import argparse
import json

from src.baseline_laplacian import run as run_baseline
from src.evaluate import evaluate, summarize
from src.make_figures import make_all as make_figures
from src.prepare_data import build_splits, index_raw, preprocess
from src.train import train
from src.utils import get_device, load_config, results_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true",
                    help="reuse data/processed and the committed split CSVs")
    args = ap.parse_args()
    cfg = load_config()
    if not args.skip_prepare:
        scenes = index_raw(cfg)
        preprocess(cfg, scenes)
        build_splits(cfg, scenes)
    run_baseline(cfg)
    timing = {"device": str(get_device())}
    for exp in cfg["experiments"]:
        timing[exp["name"] + "_train_seconds"] = round(train(cfg, exp["name"]), 1)
        evaluate(cfg, exp["name"])
    with open(results_dir() / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    summarize(cfg)
    make_figures(cfg)


if __name__ == "__main__":
    main()
