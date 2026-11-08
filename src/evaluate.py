"""Inference + evaluation on the shared fixed test set (100 scenes x {sharp, defocus, motion}).

Usage:
    python -m src.evaluate --exp <experiment_name>             # every seed in train.seeds
    python -m src.evaluate --exp <experiment_name> --seed 42   # one trained model
    python -m src.evaluate --summary                           # results/summary.csv (mean ± std over seeds)

"blurry" (reject) is the positive class. Per trained model, writes to results/<exp>/seed<seed>/:
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
from src.utils import (find_experiment, get_device, load_config, model_path, results_dir, run_name,
                       set_seed)

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


def score_predictions(name, test_df, p_blurry, threshold=0.5, extra=None, out_name=None):
    """Compute and save every metric for one method. Shared by models and baselines."""
    out = results_dir(out_name or name)
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
    plot_confusion(cm, out_name or name, out / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics


@torch.no_grad()
def predict(model, loader, device):
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def evaluate(cfg, exp_name, seed):
    find_experiment(cfg, exp_name)
    set_seed(seed)
    device = get_device()
    test_df = read_split(cfg, "test")
    model = load_trained(cfg, model_path(exp_name, seed), device)
    p_blurry = predict(model, make_loader(cfg, test_df), device)
    return score_predictions(exp_name, test_df, p_blurry, extra={"seed": seed},
                             out_name=run_name(exp_name, seed))


METRIC_COLS = ["acc_sharp", "acc_defocus", "acc_motion", "balanced_accuracy", "accuracy",
               "precision_blurry", "recall_blurry", "f1_blurry", "roc_auc"]


def collect_runs(cfg):
    """One row per (method, seed): the baseline once, every experiment once per seed."""
    rows = []
    runs = [(b, None, results_dir(b)) for b in BASELINES]
    runs += [(e["name"], s, results_dir(run_name(e["name"], s)))
             for e in cfg["experiments"] for s in cfg["train"]["seeds"]]
    for name, seed, d in runs:
        path = d / "metrics.json"
        if not path.exists():
            print(f"skip {name} seed {seed}: no metrics.json")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        m["experiment"], m["seed"] = name, seed
        rows.append(m)
    return pd.DataFrame(rows)


def summarize(cfg):
    runs = collect_runs(cfg)
    runs["seed"] = runs["seed"].astype("Int64")  # baseline has no seed
    runs[["experiment", "seed"] + METRIC_COLS].to_csv(
        results_dir() / "summary_per_seed.csv", index=False, float_format="%.4f")

    rows = []
    for name, g in runs.groupby("experiment", sort=False):
        exp = next((e for e in cfg["experiments"] if e["name"] == name), None)
        row = {"experiment": name,
               "rq": exp["rq"] if exp else "baseline",
               "train_blur": (" + ".join(f"{s} {t}" for t, s in exp["blur"].items()) if exp
                              else "none (threshold tuned on val_real_mixed)"),
               "n_runs": len(g)}
        for c in METRIC_COLS:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / "summary.csv", index=False, float_format="%.4f")

    show = df[["experiment", "n_runs"]].copy()
    for c in ["acc_sharp", "acc_defocus", "acc_motion", "balanced_accuracy", "f1_blurry", "roc_auc"]:
        show[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df[f"{c}_mean"], df[f"{c}_std"])]
    print(show.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp")
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--seed", type=int, help="default: every seed in train.seeds")
    args = ap.parse_args()
    cfg = load_config()
    if args.summary:
        summarize(cfg)
        return
    for seed in [args.seed] if args.seed is not None else cfg["train"]["seeds"]:
        evaluate(cfg, args.exp, seed)


if __name__ == "__main__":
    main()
