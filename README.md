# EF2039 PA04 — Data-centric Blur Detection for Photo Culling

Course: EF2039 Introduction to AI Programming (KENTECH)

Photo culling means throwing away the unusable shots from a photo session. This repository trains a
binary classifier that decides **keep (sharp)** or **reject (blurry)** for a photo, and studies how the
training data changes the result. Model, hyperparameters, seed, training scenes and the test set are
the same in every experiment; only the blurry training photos change.

## Research questions

- **RQ1 (synthetic vs. real data):** If we only have sharp photos and create blurry ones artificially
  (disk / motion kernels), how well does the model detect real blurry photos, compared with training
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
src/train.py               # train one experiment -> models/<exp>_seed<seed>.pt
src/evaluate.py            # inference on the fixed test set -> results/<exp>/seed<seed>/*, results/summary.csv
src/predict.py             # inference on your own photos (keep / reject)
src/baseline_laplacian.py  # classical Laplacian-variance baseline
src/check_preprocessing.py # the resize-vs-crop comparison quoted under "Data"
src/make_figures.py        # figures in results/figures/
src/run_all.py             # one command: (prepare) -> baseline -> train + evaluate all -> summary -> figures
data/splits/               # committed split CSVs — the exact files used
models/                    # checkpoints (download links in models/README.md)
results/                   # metrics, confusion matrices, predictions, figures (Mac M4, MPS)
results_backendai/         # the same experiments re-run on Backend.AI (A100 fGPU, CUDA)
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
  photo straight to 224×224 hides much of the blur: on the 250 train/validation scenes, the ROC-AUC of
  Laplacian variance drops from 0.94 to 0.90 for motion blur and from 0.99 to 0.96 for defocus
  (`python -m src.check_preprocessing`), so we keep more pixels per object and crop.
- **Synthetic blur:** each sharp photo blurred at the 448 px scale with a disk kernel (radius 2–8 px)
  or a linear motion kernel (length 5–25 px, angle 0–180°), seeded per scene. The ranges are generic
  choices and were **not** tuned on the real blurry photos.
- **Splits (by scene, seed 42):** test 100 scenes, validation 30, training pool 220 (200 used). No scene
  appears in two splits, so a test photo's sharp/blurred twins are never seen in training.
  The validation set follows each experiment's own recipe, so e.g. `synth_mixed` never sees a real
  blurry photo, even for checkpoint selection.
- **Test set (fixed):** 100 sharp + 100 defocus + 100 motion real photos (`data/splits/test.csv`).
- **Class distribution per experiment (train and validation):** `data/splits/split_summary.csv`.
  Every training set is 200 sharp + 200 blurry, every validation set 30 sharp + 30 blurry.

Download (~1.6 GB, no Kaggle account needed) and build everything:

```bash
python -m src.prepare_data --download
```

## Model and training

ImageNet-pretrained ResNet-18 (torchvision), all layers fine-tuned, 2-class head. AdamW, learning rate
1e-4, weight decay 1e-4, batch size 32, 15 epochs, horizontal-flip augmentation, cross-entropy loss.
The epoch with the lowest validation loss is kept.

**Seeds.** Data splits and synthetic blur use seed 42 and never change. Every experiment is trained
5 times with training seeds 42–46 (classifier-head initialisation, batch order, flips), and results are
reported as mean ± standard deviation over the 5 runs. The Laplacian baseline is deterministic (1 run).

## Evaluation

On the fixed test set, with `blurry` as the positive class: share of each photo type classified
correctly (sharp kept, defocus rejected, motion rejected), balanced accuracy (the test set has twice
as many blurry as sharp photos), accuracy, precision / recall / F1 for `blurry`, and ROC-AUC.

## Reproduce all results

```bash
python -m src.prepare_data --download   # once: data + processed images + splits
python -m src.run_all --skip-prepare    # baseline, 4 experiments x 5 seeds (train + evaluate), summary, figures
```

Individual steps:

```bash
python -m src.baseline_laplacian
python -m src.train --exp real_mixed              # all 5 seeds (add --seed 42 for one)
python -m src.evaluate --exp real_mixed           # all 5 seeds (add --seed 42 for one)
python -m src.evaluate --summary
python -m src.make_figures
```

## Trained model weights

The 20 checkpoints behind `results/` (4 experiments x seeds 42–46, ResNet-18 `state_dict`, ~45 MB each)
are attached to the GitHub Release
[v1.0](https://github.com/Alkasion/EF2039_AI_Programming_PA04/releases/tag/v1.0).
To evaluate them without training:

```bash
for e in real_mixed synth_mixed real_motion_only real_defocus_only; do
  for s in 42 43 44 45 46; do
    curl -L -o models/${e}_seed$s.pt \
      https://github.com/Alkasion/EF2039_AI_Programming_PA04/releases/download/v1.0/${e}_seed$s.pt
  done
done
python -m src.prepare_data --download                     # test images (skip if already done)
for e in real_mixed synth_mixed real_motion_only real_defocus_only; do python -m src.evaluate --exp $e; done
python -m src.baseline_laplacian && python -m src.evaluate --summary
```

`results/summary.csv` should then match the "Expected results" table below.

## Inference on your own photos

```bash
python -m src.predict photo.jpg my_photos/                 # real_mixed, seed 42
python -m src.predict my_photos/ --exp synth_mixed --seed 43 --csv decisions.csv
```

Prints P(blurry) and KEEP / REJECT for each photo. A photo gets the same preprocessing as the training
data, so the decision is based on the 224×224 centre crop.

## Hardware

The results in `results/` were produced on an Apple M4 MacBook (10-core CPU, 16 GB RAM) with PyTorch MPS. One training run took 59–84 s (15 epochs, 400 images); all 20 runs plus evaluation took about 28 min. Preprocessing takes about 1 min. A GPU is not required; CPU-only training works but is slower (not yet timed). About 2 GB of disk (1.6 GB download + images) and 4 GB of RAM are enough; the 20 checkpoints need another 0.9 GB.

## Expected results

| Method | Train blur | Sharp kept | Defocus rejected | Motion rejected | Balanced acc. | F1 (blurry) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| `laplacian_baseline` | none (threshold tuned on val_real_mixed) | 0.820 | 0.990 | 0.840 | 0.868 | 0.913 | 0.953 |
| `real_mixed` | real defocus + real motion | 0.966 ± 0.005 | 1.000 ± 0.000 | 0.984 ± 0.005 | 0.979 ± 0.001 | 0.988 ± 0.000 | 0.999 ± 0.000 |
| `synth_mixed` | synthetic defocus + synthetic motion | 0.988 ± 0.004 | 0.986 ± 0.005 | 0.936 ± 0.015 | 0.975 ± 0.005 | 0.977 ± 0.004 | 0.997 ± 0.001 |
| `real_motion_only` | real motion | 0.958 ± 0.013 | 0.998 ± 0.004 | 0.988 ± 0.008 | 0.976 ± 0.007 | 0.986 ± 0.004 | 0.999 ± 0.000 |
| `real_defocus_only` | real defocus | 0.986 ± 0.009 | 0.998 ± 0.004 | 0.916 ± 0.026 | 0.972 ± 0.007 | 0.975 ± 0.007 | 0.996 ± 0.001 |

Mean ± standard deviation over 5 training seeds (42–46); the baseline is deterministic. Test set: 100 sharp + 100 defocus + 100 motion real photos. Per-run numbers: `results/summary_per_seed.csv`.

Small differences (about ±1 photo per run) are possible across devices, because GPU kernels are not bit-exact.

## Cross-device check (Backend.AI)

All experiments were also run from scratch on KENTECH Backend.AI (fractional A100, CUDA, PyTorch 2.8 NGC image),
with data downloaded and split on the server. The rebuilt split files were byte-identical to `data/splits/`,
and the Laplacian baseline gave exactly the same numbers. Trained models differ slightly because GPU kernels
and library versions differ, but every conclusion is the same on both devices. Full outputs: `results_backendai/`.

| Method | Device | Sharp kept | Defocus rejected | Motion rejected | Balanced acc. |
|---|---|---|---|---|---|
| `laplacian_baseline` | Mac M4 (MPS) | 0.820 | 0.990 | 0.840 | 0.868 |
| `laplacian_baseline` | A100 fGPU (CUDA) | 0.820 | 0.990 | 0.840 | 0.868 |
| `real_mixed` | Mac M4 (MPS) | 0.966 ± 0.005 | 1.000 ± 0.000 | 0.984 ± 0.005 | 0.979 ± 0.001 |
| `real_mixed` | A100 fGPU (CUDA) | 0.960 ± 0.012 | 1.000 ± 0.000 | 0.986 ± 0.005 | 0.977 ± 0.005 |
| `synth_mixed` | Mac M4 (MPS) | 0.988 ± 0.004 | 0.986 ± 0.005 | 0.936 ± 0.015 | 0.975 ± 0.005 |
| `synth_mixed` | A100 fGPU (CUDA) | 0.982 ± 0.013 | 0.990 ± 0.007 | 0.948 ± 0.023 | 0.976 ± 0.005 |
| `real_motion_only` | Mac M4 (MPS) | 0.958 ± 0.013 | 0.998 ± 0.004 | 0.988 ± 0.008 | 0.976 ± 0.007 |
| `real_motion_only` | A100 fGPU (CUDA) | 0.960 ± 0.012 | 0.998 ± 0.004 | 0.988 ± 0.008 | 0.977 ± 0.005 |
| `real_defocus_only` | Mac M4 (MPS) | 0.986 ± 0.009 | 0.998 ± 0.004 | 0.916 ± 0.026 | 0.972 ± 0.007 |
| `real_defocus_only` | A100 fGPU (CUDA) | 0.986 ± 0.009 | 0.998 ± 0.004 | 0.916 ± 0.019 | 0.972 ± 0.006 |

## License and citation

The images belong to the Blur dataset authors (see the Kaggle page for its license) and are not
redistributed here; `src/prepare_data.py --download` fetches them from Kaggle.
