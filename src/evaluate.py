"""Inference + evaluation on the shared fixed test set.

Usage:
    python -m src.evaluate --exp <experiment_name>   # one model
    python -m src.evaluate --summary                 # merge all metrics into results/summary.csv

Per experiment, writes to results/<exp>/:
    predictions.csv, metrics.json, per_class.csv, confusion_matrix.csv, confusion_matrix.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data import make_dataset, read_split
from src.utils import find_experiment, load_config, model_path, results_dir, set_seed


def plot_confusion(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.2 * len(classes) + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate(cfg, exp_name):
    find_experiment(cfg, exp_name)
    set_seed(cfg["seed"])
    classes = cfg["data"]["classes"]
    out = results_dir(exp_name)

    test_df = read_split(cfg, "test")
    model = tf.keras.models.load_model(model_path(exp_name))
    probs = model.predict(make_dataset(cfg, test_df), verbose=0)

    y_true = np.array([classes.index(l) for l in test_df["label"]])
    y_pred = probs.argmax(axis=1)

    pred_df = test_df.copy()
    pred_df["pred"] = [classes[i] for i in y_pred]
    pred_df["confidence"] = probs.max(axis=1)
    pred_df.to_csv(out / "predictions.csv", index=False)

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame({
        "class": classes,
        "accuracy": cm.diagonal() / cm.sum(axis=1).clip(min=1),  # = per-class recall
        "precision": prec, "recall": rec, "f1": f1, "support": support,
    })
    per_class.to_csv(out / "per_class.csv", index=False, float_format="%.4f")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")

    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    metrics = {
        "experiment": exp_name,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(m_prec),
        "macro_recall": float(m_rec),
        "macro_f1": float(m_f1),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(cm, classes, exp_name, out / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics


def summarize(cfg):
    rows = []
    for exp in cfg["experiments"]:
        path = results_dir(exp["name"]) / "metrics.json"
        if not path.exists():
            print(f"skip {exp['name']}: no metrics.json")
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        m["rq"] = exp.get("rq", "")
        pc = pd.read_csv(results_dir(exp["name"]) / "per_class.csv")
        for _, r in pc.iterrows():
            m[f"acc_{r['class']}"] = r["accuracy"]
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / "summary.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False))


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
