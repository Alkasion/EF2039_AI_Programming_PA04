# Results on Backend.AI (cross-device check)

The same code, config and seeds as `results/`, run from scratch on KENTECH Backend.AI
(`ai-gpu.kentech.ac.kr`) on 2026-11-08: data downloaded from Kaggle, preprocessed, split,
then `python -m src.run_all --skip-prepare`. Layout is identical to `results/`.

- Hardware: fractional NVIDIA A100 (0.2 fGPU, 16 GB visible), 6 CPU cores, 50 GiB RAM
- Software (container image PyTorch NGC 25.05, packages from the image, not `environment.yml`):
  Python 3.12.3, PyTorch 2.8.0a0 (NGC build), torchvision 0.22.0a0, NumPy 1.26.4,
  OpenCV 4.10.0, scikit-learn 1.6.1 — see the `ENV` line at the top of `run_all.log`
- The rebuilt split files had the same SHA-256 as the committed ones in `data/splits/`
  (`test.csv` e8d04910…, `train_real_mixed.csv` 52e9aae8…)
- Training: 17–22 s per run, 20 runs in about 7 min (`timing.json`)
