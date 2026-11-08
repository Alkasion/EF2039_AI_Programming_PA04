"""Inference + evaluation on the shared fixed test set (100 scenes x {sharp, defocus, motion}).

Usage:
    python -m src.evaluate --exp <experiment_name>   # one model
    python -m src.evaluate --summary                 # merge all metrics into results/summary.csv

"blurry" (reject) is the positive class. Per experiment, writes to results/<exp>/:
    predictions.csv  per-image prediction and P(blurry)
    metrics.json     accuracy, balanced accuracy, precision/recall/F1 (blurry), ROC-AUC,
                     and the share of each blur type classified correctly
    confusion_matrix.csv|png   rows = true blur type (sharp/defocus/motion), cols = predicted label
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from src.data import make_loader, read_split
from src.model import load_trained
from src.utils import find_experiment, get_device, load_config, model_path, results_dir, set_seed

BLUR_TYPES = ["sharp", "defocus", "motion"]
BASELINES = ["laplacian_baseline"]


def plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    rate = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["sharp\n(keep)", "blurry\n(reject)"])
    ax.set_yticks(range(len(BLUR_TYPES)), BLUR_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True photo type")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({rate[i, j]:.0%})", ha="center", va="center", fontsize=9,
                    color="white" if rate[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def score_predictions(name, test_df, p_blurry, threshold=0.5, extra=None):
    """Compute and save every metric for one method. Shared by models and baselines."""
    out = results_dir(name)
    y_true = (test_df["label"] == "blurry").astype(int).to_numpy()
    y_pred = (p_blurry >= threshold).astype(int)

    pred_df = test_df.copy()
    pred_df["p_blurry"] = p_blurry
    pred_df["pred"] = np.where(y_pred == 1, "blurry", "sharp")
    pred_df["correct"] = y_pred == y_true
    pred_df.to_csv(out / "predictions.csv", index=False, float_format="%.4f")

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "experiment": name,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_blurry": float(prec),
        "recall_blurry": float(rec),
        "f1_blurry": float(f1),
        "roc_auc": float(roc_auc_score(y_true, p_blurry)),
    }
    cm = np.zeros((len(BLUR_TYPES), 2), dtype=int)
    for i, bt in enumerate(BLUR_TYPES):
        mask = (test_df["blur_type"] == bt).to_numpy()
        cm[i, 0] = int((y_pred[mask] == 0).sum())
        cm[i, 1] = int((y_pred[mask] == 1).sum())
        correct = cm[i, 0] if bt == "sharp" else cm[i, 1]
        metrics[f"acc_{bt}"] = float(correct / mask.sum())
    if extra:
        metrics.update(extra)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=BLUR_TYPES, columns=["pred_sharp", "pred_blurry"]).to_csv(out / "confusion_matrix.csv")
    plot_confusion(cm, name, out / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics


@torch.no_grad()
def predict(model, loader, device):
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def evaluate(cfg, exp_name):
    find_experiment(cfg, exp_name)
    set_seed(cfg["seed"])
    device = get_device()
    test_df = read_split(cfg, "test")
    model = load_trained(cfg, model_path(exp_name), device)
    p_blurry = predict(model, make_loader(cfg, test_df), device)
    return score_predictions(exp_name, test_df, p_blurry)


def summarize(cfg):
    rows = []
    for name in BASELINES + [e["name"] for e in cfg["experiments"]]:
        path = results_dir(name) / "metrics.json"
        if not path.exists():
            print(f"skip {name}: no metrics.json")
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        exp = next((e for e in cfg["experiments"] if e["name"] == name), None)
        m["rq"] = exp["rq"] if exp else "baseline"
        m["train_blur"] = (" + ".join(f"{s} {t}" for t, s in exp["blur"].items()) if exp
                           else "none (threshold tuned on val_real_mixed)")
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["experiment", "rq", "train_blur", "acc_sharp", "acc_defocus", "acc_motion",
            "balanced_accuracy", "accuracy", "precision_blurry", "recall_blurry", "f1_blurry", "roc_auc"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    df.to_csv(results_dir() / "summary.csv", index=False, float_format="%.4f")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp")
    g.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.summary:
        summarize(cfg)
    else:
        evaluate(cfg, args.exp)


if __name__ == "__main__":
    main()
