# EF2039 PA04 — Data-centric Image Classification

> **Status:** skeleton. Dataset, classes and research questions are still TBD; sections marked `TODO(topic)` will be filled in once they are chosen.

Course: EF2039 Introduction to AI Programming (KENTECH)

## Research questions

- **RQ1:** TODO(topic)
- **RQ2:** TODO(topic)

All experiments share the same model, hyperparameters, seed, and **fixed test set**; they differ only in their training data.

## Repository layout

```
configs/experiments.yaml   # dataset, hyperparameters, and every experiment (single source of truth)
src/prepare_data.py        # download (TODO) + build fixed test/val splits and per-experiment train subsets
src/train.py               # train one experiment -> models/<exp>.keras
src/evaluate.py            # inference on the fixed test set -> results/<exp>/*, results/summary.csv
src/run_all.py             # one command: prepare -> train -> evaluate every experiment -> summary
data/splits/               # committed split CSVs (filepath,label) — exact files used
models/                    # checkpoints (see models/README.md)
results/                   # confusion matrices, per-class metrics, summary table
```

## Hardware

TODO: fill in after the first full run (machine used, runtime per experiment).
Developed on Apple Silicon (arm64, macOS). A GPU is not required.

## Setup

```bash
git clone https://github.com/Alkasion/EF2039_AI_Programming_PA04.git
cd EF2039_AI_Programming_PA04
conda env create -f environment.yml
conda activate ef2039-pa04
```

## Data

- **Source:** TODO(topic) (Kaggle link, license)
- **Classes:** TODO(topic)
- **Preprocessing:** images decoded as RGB, resized to 224×224; unreadable files are skipped.
- **Splits:** per class, a fixed test set (`test_per_class`) and validation set (`val_per_class`) are drawn first with seed 42; each experiment's training subset is drawn from the remaining pool. Smaller subsets are nested inside larger ones. Exact file lists are in `data/splits/`.
- **Class distribution:** see `data/splits/split_summary.csv`.

Download:

```bash
# TODO(topic): exact download command
```

Expected layout: `data/raw/<class_name>/<image files>`.

## Reproduce all results

```bash
python -m src.run_all --skip-prepare
```

`--skip-prepare` reuses the committed split files. Drop it to rebuild the splits from `data/raw` (same seed gives the same files).

Individual steps:

```bash
python -m src.prepare_data
python -m src.train --exp <name>
python -m src.evaluate --exp <name>
python -m src.evaluate --summary
```

## Expected results

TODO: filled from `results/summary.csv` after the final run.

## Pretrained weights

TODO: link or committed files (see `models/README.md`).
