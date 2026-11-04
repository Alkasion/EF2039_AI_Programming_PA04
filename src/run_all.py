"""Reproduce every number in the paper with one command.

Usage: python -m src.run_all [--skip-prepare]
"""
import argparse

from src.evaluate import evaluate, summarize
from src.prepare_data import build_splits
from src.train import train
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true",
                    help="reuse the committed split CSVs instead of rebuilding them")
    args = ap.parse_args()
    cfg = load_config()
    if not args.skip_prepare:
        build_splits(cfg)
    for exp in cfg["experiments"]:
        train(cfg, exp["name"])
        evaluate(cfg, exp["name"])
    summarize(cfg)


if __name__ == "__main__":
    main()
