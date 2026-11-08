"""Train one experiment; keep the epoch with the lowest validation loss.

Usage:
    python -m src.train --exp <experiment_name>              # every seed in train.seeds
    python -m src.train --exp <experiment_name> --seed 42    # one seed
"""
import argparse
import time

import pandas as pd
import torch

from src.data import make_loader, read_split
from src.model import build_model
from src.utils import (find_experiment, get_device, load_config, model_path, results_dir, run_name,
                       set_seed)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(cfg, exp_name, seed):
    find_experiment(cfg, exp_name)
    set_seed(seed)
    device = get_device()
    tcfg = cfg["train"]

    train_df = read_split(cfg, f"train_{exp_name}")
    val_df = read_split(cfg, f"val_{exp_name}")
    counts = train_df.groupby(["blur_type", "source"]).size().to_dict()
    print(f"[{exp_name} seed {seed}] device={device} train={len(train_df)} {counts} val={len(val_df)}")

    train_loader = make_loader(cfg, train_df, shuffle=True, augment=tcfg.get("augment", False), seed=seed)
    val_loader = make_loader(cfg, val_df)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])

    history, best = [], float("inf")
    start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "loss": tr_loss, "accuracy": tr_acc,
                        "val_loss": va_loss, "val_accuracy": va_acc})
        flag = ""
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), model_path(exp_name, seed))
            flag = " *"
        print(f"  epoch {epoch:2d}  loss {tr_loss:.4f} acc {tr_acc:.3f}  "
              f"val_loss {va_loss:.4f} val_acc {va_acc:.3f}{flag}")
    elapsed = time.time() - start
    pd.DataFrame(history).to_csv(results_dir(run_name(exp_name, seed)) / "history.csv", index=False, float_format="%.4f")
    print(f"[{exp_name} seed {seed}] {elapsed:.0f}s on {device}; best checkpoint -> {model_path(exp_name, seed)}")
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, help="default: every seed in train.seeds")
    args = ap.parse_args()
    cfg = load_config()
    for seed in [args.seed] if args.seed is not None else cfg["train"]["seeds"]:
        train(cfg, args.exp, seed)


if __name__ == "__main__":
    main()
