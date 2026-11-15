"""Classical baseline: variance of the Laplacian (a standard sharpness measure, no learning).

A photo is called blurry when its Laplacian variance is below a threshold. The threshold is the
one that maximises balanced accuracy on the val_real_mixed split; the test set is never used to
choose it.

Usage: python -m src.baseline_laplacian
"""
import cv2
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.data import read_split
from src.evaluate import score_predictions
from src.utils import load_config, processed_dir

NAME = "laplacian_baseline"


def laplacian_variance(cfg, df):
    root = processed_dir(cfg)
    vals = []
    for p in df["filepath"]:
        gray = cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Unreadable image: {root / p}. Run `python -m src.prepare_data` first.")
        vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.array(vals)


def run(cfg):
    val_df = read_split(cfg, "val_real_mixed")
    test_df = read_split(cfg, "test")
    val_score = -np.log(laplacian_variance(cfg, val_df) + 1e-6)  # higher = blurrier
    y_val = (val_df["label"] == "blurry").astype(int).to_numpy()
    candidates = np.unique(val_score)
    best_t = max(candidates, key=lambda t: balanced_accuracy_score(y_val, (val_score >= t).astype(int)))

    test_score = -np.log(laplacian_variance(cfg, test_df) + 1e-6)
    # Map to (0, 1) around the threshold so the shared scorer can use 0.5 and ROC-AUC is unchanged.
    p_blurry = 1.0 / (1.0 + np.exp(-(test_score - best_t)))
    return score_predictions(NAME, test_df, p_blurry,
                             extra={"laplacian_var_threshold": float(np.exp(-best_t))})


def main():
    run(load_config())


if __name__ == "__main__":
    main()
