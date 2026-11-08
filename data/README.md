# Data

Everything here except `splits/` is created by `python -m src.prepare_data --download` and is git-ignored.

- `raw/` — the Kaggle Blur dataset (`kwentar/blur-dataset`, scaled version: longest side 2048 px), as
  `raw/{sharp,defocused_blurred,motion_blurred}/<scene>_<device>_<S|F|M>.<ext>`. 350 scenes x 3 photos.
- `processed/real/{sharp,defocus,motion}/<scene>.png` — every real photo resized so the short side is 448 px,
  then center-cropped to 224x224 (lossless PNG).
- `processed/synthetic/{defocus,motion}/<scene>.png` — the scene's **sharp** photo blurred with a random disk
  kernel (radius 2–8 px) or linear motion kernel (length 5–25 px, random angle) at the 448 px scale, then
  cropped the same way. The random parameters are seeded per scene, so a rebuild gives identical files.

`splits/` holds the exact file lists used by every experiment (committed). Columns:
`filepath` (relative to `processed/`), `label` (sharp/blurry), `blur_type` (sharp/defocus/motion),
`source` (real/synthetic), `scene`, `device`.

- `test.csv` — fixed, shared by all experiments: 100 scenes x {sharp, defocus, motion} = 300 real photos
- `train_<experiment>.csv`, `val_<experiment>.csv` — 200 train scenes / 30 val scenes, built with the
  experiment's recipe (one sharp + one blurry photo per scene)
- `split_summary.csv` — number of training images per blur type and source for every experiment

Scenes, not photos, are split, so no scene appears in more than one of test / val / train.
