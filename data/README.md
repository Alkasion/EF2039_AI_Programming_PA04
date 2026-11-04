# Data

Raw images live in `data/raw/` (git-ignored, downloaded by script). See the main README "Data" section.

`splits/` holds the exact file lists used by every experiment (committed):
- `test.csv`, `val.csv` — fixed, shared by all experiments
- `train_<experiment>.csv` — training subset per experiment
- `split_summary.csv` — training images per class per experiment
