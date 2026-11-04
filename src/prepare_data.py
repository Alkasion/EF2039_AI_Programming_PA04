"""Build the fixed test/val splits and one training subset per experiment.

Output (committed to git so anyone gets identical splits):
    data/splits/test.csv, data/splits/val.csv
    data/splits/train_<experiment>.csv
Each CSV has columns: filepath (relative to data/raw), label.

Usage:
    python -m src.prepare_data            # all experiments
    python -m src.prepare_data --download # TODO(topic): fetch dataset first
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, per_class_count, splits_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download_dataset(cfg):
    # TODO(topic): implement once the Kaggle dataset is chosen
    # (e.g. `kaggle datasets download -d <owner>/<name>` then unzip into data/raw/<class>/).
    raise NotImplementedError("Dataset not chosen yet; see README 'Data' section.")


def list_images(raw_dir, class_name):
    folder = raw_dir / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    readable = [p for p in files if cv2.imread(str(p)) is not None]
    dropped = len(files) - len(readable)
    if dropped:
        print(f"[{class_name}] skipped {dropped} unreadable files")
    return [str(p.relative_to(raw_dir)) for p in readable]


def build_splits(cfg):
    seed = cfg["seed"]
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    out_dir = splits_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["data"]["classes"]
    if not classes:
        raise ValueError("configs/experiments.yaml: data.classes is empty (topic not set).")

    n_test = cfg["data"]["test_per_class"]
    n_val = cfg["data"]["val_per_class"]
    rng = np.random.default_rng(seed)

    test_rows, val_rows, pools = [], [], {}
    for c in classes:
        files = np.array(list_images(raw_dir, c))
        files = files[rng.permutation(len(files))]
        if len(files) < n_test + n_val:
            raise ValueError(f"[{c}] only {len(files)} images; need >= {n_test + n_val}")
        test_rows += [(f, c) for f in files[:n_test]]
        val_rows += [(f, c) for f in files[n_test:n_test + n_val]]
        pools[c] = files[n_test + n_val:]  # training pool, disjoint from test/val

    pd.DataFrame(test_rows, columns=["filepath", "label"]).to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame(val_rows, columns=["filepath", "label"]).to_csv(out_dir / "val.csv", index=False)

    summary = []
    for exp in cfg["experiments"]:
        rows = []
        for c in classes:
            k = per_class_count(exp, c)
            if k > len(pools[c]):
                raise ValueError(f"{exp['name']}: wants {k} '{c}' images, pool has {len(pools[c])}")
            # Taking a prefix of one fixed shuffle makes smaller subsets nested in larger ones.
            rows += [(f, c) for f in pools[c][:k]]
            summary.append({"experiment": exp["name"], "class": c, "n_train": k})
        pd.DataFrame(rows, columns=["filepath", "label"]).to_csv(
            out_dir / f"train_{exp['name']}.csv", index=False
        )

    pd.DataFrame(summary).to_csv(out_dir / "split_summary.csv", index=False)
    print(f"Wrote splits to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.download:
        download_dataset(cfg)
    build_splits(cfg)


if __name__ == "__main__":
    main()
