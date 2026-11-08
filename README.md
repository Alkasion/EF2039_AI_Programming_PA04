# EF2039 PA04 — Data-centric Blur Detection for Photo Culling

Course: EF2039 Introduction to AI Programming (KENTECH)

Photo culling means throwing away the unusable shots from a photo session. This repository trains a
binary classifier that decides **keep (sharp)** or **reject (blurry)** for a photo, and studies how the
*training data* changes the result. Model, hyperparameters, seed, training scenes and the test set are
the same in every experiment; only the blurry training photos change.

## Research questions

- **RQ1 (synthetic vs. real data):** If we only have sharp photos and create blurry ones artificially
  (disk / motion kernels), how well does the model detect *real* blurry photos, compared with training
  on real blurry photos?
- **RQ2 (diversity of blur types):** If the training data contains only one kind of blur (motion or
  defocus), does the model still detect the other kind?

| Experiment | RQ | Blurry half of the training set (200 photos) |
|---|---|---|
| `real_mixed` | RQ1, RQ2 | 100 real defocus + 100 real motion |
| `synth_mixed` | RQ1 | 100 synthetic defocus + 100 synthetic motion (made from the sharp photos) |
| `real_motion_only` | RQ2 | 200 real motion |
| `real_defocus_only` | RQ2 | 200 real defocus |
| `laplacian_baseline` | — | no training; Laplacian-variance threshold tuned on `val_real_mixed` |

Every experiment also uses the same 200 real sharp photos, so each training set is 200 sharp + 200 blurry.

## Repository layout

```
configs/experiments.yaml   # dataset, preprocessing, hyperparameters, every experiment (single source of truth)
src/prepare_data.py        # download, preprocess, synthetic blur, scene-level splits
src/train.py               # train one experiment -> models/<exp>.pt
src/evaluate.py            # inference on the fixed test set -> results/<exp>/*, results/summary.csv
src/baseline_laplacian.py  # classical Laplacian-variance baseline
src/make_figures.py        # figures in results/figures/
src/run_all.py             # one command: (prepare) -> baseline -> train + evaluate all -> summary -> figures
data/splits/               # committed split CSVs — the exact files used
models/                    # checkpoints (download links in models/README.md)
results/                   # metrics, confusion matrices, predictions, figures
```

## Setup

```bash
git clone https://github.com/Alkasion/EF2039_AI_Programming_PA04.git
cd EF2039_AI_Programming_PA04
conda env create -f environment.yml
conda activate ef2039-pa04
```

The code picks the device automatically: CUDA GPU, then Apple MPS, then CPU. Force one with
`PA04_DEVICE=cpu` (or `mps`, `cuda`). ImageNet weights for ResNet-18 are downloaded once to
`$TORCH_HOME` (default `~/.cache/torch`).

<details>
<summary>Optional: keep the environment and caches on another drive</summary>

```bash
D=/path/to/drive   # e.g. an external SSD
CONDA_PKGS_DIRS=$D/.cache/conda_pkgs PIP_CACHE_DIR=$D/.cache/pip \
  conda env create -f environment.yml -p $D/conda_envs/ef2039-pa04
conda env config vars set -p $D/conda_envs/ef2039-pa04 \
  TORCH_HOME=$D/.cache/torch MPLCONFIGDIR=$D/.cache/matplotlib \
  PIP_CACHE_DIR=$D/.cache/pip CONDA_PKGS_DIRS=$D/.cache/conda_pkgs TMPDIR=$D/.cache/tmp
conda activate $D/conda_envs/ef2039-pa04
```
</details>

### Running on Backend.AI (or any Linux server)

1. Start a session with a Python/conda-capable image and mount a **virtual folder** (vfolder). Files
   outside the vfolder are lost when the session ends, so clone the repository *into* the vfolder.
2. In the session terminal:
   ```bash
   cd /home/work/<your-vfolder>
   git clone https://github.com/Alkasion/EF2039_AI_Programming_PA04.git
   cd EF2039_AI_Programming_PA04
   conda env create -f environment.yml -p ./env     # environment inside the vfolder, kept between sessions
   conda activate ./env
   python -m src.prepare_data --download
   nohup python -u -m src.run_all --skip-prepare > run_all.log 2>&1 &   # keeps running if the terminal closes
   tail -f run_all.log
   ```
3. `results/timing.json` records which device was used and how long each experiment took.

If `conda` is not in the image, `python -m venv env && . env/bin/activate && pip install` the packages
listed under `pip:` in `environment.yml` gives the same versions.

## Data

- **Source:** [Blur dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) on Kaggle
  (`kwentar/blur-dataset`), scaled version (longest side 2048 px). 350 scenes photographed with 66
  devices (mostly smartphones); each scene has one sharp (S), one defocus-blurred (F) and one
  motion-blurred (M) photo → 1,050 photos.
- **Labels:** `sharp` (keep) for S; `blurry` (reject) for F and M. The blur type is kept for analysis.
- **Preprocessing:** resize so the short side is 448 px, then center-crop 224×224. Resizing the whole
  photo straight to 224×224 hides much of the blur (on 120 scenes, the Laplacian-variance AUC for motion
  blur drops from 0.92 to 0.88), so we keep more pixels per object and crop.
- **Synthetic blur:** each sharp photo blurred at the 448 px scale with a disk kernel (radius 2–8 px)
  or a linear motion kernel (length 5–25 px, angle 0–180°), seeded per scene. The ranges are generic
  choices and were **not** tuned on the real blurry photos.
- **Splits (by scene, seed 42):** test 100 scenes, validation 30, training pool 220 (200 used). No scene
  appears in two splits, so a test photo's sharp/blurred twins are never seen in training.
  The validation set follows each experiment's own recipe, so e.g. `synth_mixed` never sees a real
  blurry photo, even for checkpoint selection.
- **Test set (fixed):** 100 sharp + 100 defocus + 100 motion real photos (`data/splits/test.csv`).
- **Class distribution per experiment:** `data/splits/split_summary.csv`.

Download (~1.6 GB, no Kaggle account needed) and build everything:

```bash
python -m src.prepare_data --download
```

## Model and training

ImageNet-pretrained ResNet-18 (torchvision), all layers fine-tuned, 2-class head. AdamW, learning rate
1e-4, weight decay 1e-4, batch size 32, 15 epochs, horizontal-flip augmentation, cross-entropy loss.
The epoch with the lowest validation loss is kept. Seed 42 everywhere.

## Evaluation

On the fixed test set, with `blurry` as the positive class: share of each photo type classified
correctly (sharp kept, defocus rejected, motion rejected), balanced accuracy (the test set has twice
as many blurry as sharp photos), accuracy, precision / recall / F1 for `blurry`, and ROC-AUC.

## Reproduce all results

```bash
python -m src.prepare_data --download   # once: data + processed images + splits
python -m src.run_all --skip-prepare    # baseline, 4 x (train + evaluate), summary, figures
```

Individual steps:

```bash
python -m src.baseline_laplacian
python -m src.train --exp real_mixed
python -m src.evaluate --exp real_mixed
python -m src.evaluate --summary
python -m src.make_figures
```

To evaluate the released checkpoints without training, download them (see `models/README.md`) and run
`python -m src.evaluate --exp <name>` for each experiment.

## Hardware

Results above were produced on an Apple M4 MacBook (10-core CPU, 16 GB RAM) with PyTorch MPS. Training took about 68–78 s per experiment (15 epochs, 400 images); preprocessing about 1 min. A GPU is not required; CPU-only training works but is slower (not yet timed). About 2 GB of disk (1.6 GB download + images) and 4 GB of RAM are enough.

## Expected results

| Method | Train blur | Sharp kept | Defocus rejected | Motion rejected | Balanced acc. | F1 (blurry) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| `laplacian_baseline` | none (threshold tuned on val_real_mixed) | 0.82 | 0.99 | 0.84 | 0.87 | 0.91 | 0.95 |
| `real_mixed` | real defocus + real motion | 0.96 | 1.00 | 0.99 | 0.98 | 0.99 | 1.00 |
| `synth_mixed` | synthetic defocus + synthetic motion | 0.99 | 0.99 | 0.96 | 0.98 | 0.98 | 1.00 |
| `real_motion_only` | real motion | 0.95 | 1.00 | 0.99 | 0.97 | 0.99 | 1.00 |
| `real_defocus_only` | real defocus | 0.99 | 0.99 | 0.91 | 0.97 | 0.97 | 1.00 |

Test set: 100 sharp + 100 defocus + 100 motion real photos; one run, seed 42. Full numbers: `results/summary.csv`.

Small differences (about ±1 photo) are possible across devices, because GPU kernels are not bit-exact.

## License and citation

The images belong to the Blur dataset authors (see the Kaggle page for its license) and are not
redistributed here; `src/prepare_data.py --download` fetches them from Kaggle.
