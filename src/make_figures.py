"""Figures for the paper, built only from data/processed and results/.

Usage: python -m src.make_figures
Writes to results/figures/:
    samples.png               real vs synthetic blur on two test scenes
    per_type_accuracy.png     share of sharp / defocus / motion test photos classified correctly (mean ± std)
    culling_demo.png          keep / reject decisions of real_mixed (first seed) on test photos, incl. mistakes
"""
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config, processed_dir, results_dir, run_name, splits_dir


def _rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def samples(cfg, out):
    test = pd.read_csv(splits_dir(cfg) / "test.csv")
    scenes = test["scene"].drop_duplicates().iloc[:2].tolist()
    cols = [("real", "sharp", "Real sharp"), ("real", "defocus", "Real defocus"),
            ("real", "motion", "Real motion"), ("synthetic", "defocus", "Synthetic defocus"),
            ("synthetic", "motion", "Synthetic motion")]
    fig, axes = plt.subplots(len(scenes), len(cols), figsize=(2.0 * len(cols), 2.1 * len(scenes)))
    for r, sid in enumerate(scenes):
        for c, (src, bt, title) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(_rgb(processed_dir(cfg) / src / bt / f"{sid}.png"))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=200)
    plt.close(fig)


def per_type_accuracy(cfg, out):
    """Bars = mean over seeds, error bars = std, dots = individual runs."""
    summary = pd.read_csv(results_dir() / "summary.csv")
    runs = pd.read_csv(results_dir() / "summary_per_seed.csv")
    types = [("acc_sharp", "sharp (kept)"), ("acc_defocus", "defocus (rejected)"),
             ("acc_motion", "motion (rejected)")]
    x = np.arange(len(summary))
    w = 0.26
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for i, (key, label) in enumerate(types):
        means, stds = summary[f"{key}_mean"], summary[f"{key}_std"]
        pos = x + (i - 1) * w
        bars = ax.bar(pos, means, w, yerr=stds, capsize=2, label=label, color=colors[i],
                      error_kw={"elinewidth": 0.8})
        ax.bar_label(bars, labels=[f"{m:.2f}" for m in means], fontsize=6.5, padding=2,
                     label_type="edge")
        for k, name in enumerate(summary["experiment"]):
            vals = runs.loc[runs["experiment"] == name, key].to_numpy()
            if len(vals) > 1:
                ax.scatter(np.full(len(vals), pos[k]), vals, s=4, color="black", zorder=3, linewidths=0)
    ax.set_xticks(x, [n.replace("_", "\n", 1) for n in summary["experiment"]], fontsize=8)
    ax.set_ylabel("Correct on test photos")
    ax.set_ylim(0.6, 1.05)
    ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "per_type_accuracy.png", dpi=200)
    plt.close(fig)


def culling_demo(cfg, out, exp="real_mixed", n_correct=6, n_wrong=4):
    p = results_dir(run_name(exp, cfg["train"]["seeds"][0])) / "predictions.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    rng = np.random.default_rng(cfg["seed"])
    ok = df[df["correct"]]
    picks = [ok[ok["blur_type"] == bt].sample(n_correct // 3, random_state=cfg["seed"]) for bt in
             ("sharp", "defocus", "motion")]
    wrong = df[~df["correct"]]
    if len(wrong):
        picks.append(wrong.sample(min(n_wrong, len(wrong)), random_state=cfg["seed"]))
    sel = pd.concat(picks).reset_index(drop=True)
    sel = sel.iloc[rng.permutation(len(sel))] if len(sel) else sel
    ncol = 5
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.35 * nrow))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), sel.iterrows()):
        ax.imshow(_rgb(processed_dir(cfg) / r["filepath"]))
        decision = "REJECT" if r["pred"] == "blurry" else "KEEP"
        color = "#2E7D32" if r["correct"] else "#C62828"
        ax.set_title(f"{decision}  p={r['p_blurry']:.2f}\ntrue: {r['blur_type']}", fontsize=8, color=color)
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(color), s.set_linewidth(2)
        ax.axis("on"), ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "culling_demo.png", dpi=200)
    plt.close(fig)


def make_all(cfg):
    out = results_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    samples(cfg, out)
    per_type_accuracy(cfg, out)
    culling_demo(cfg, out)
    print(f"Wrote figures to {out}")


def main():
    make_all(load_config())


if __name__ == "__main__":
    main()
