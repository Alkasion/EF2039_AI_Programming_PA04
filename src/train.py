"""Train one experiment.

Usage: python -m src.train --exp <experiment_name>
"""
import argparse

import pandas as pd
import tensorflow as tf

from src.data import make_dataset, read_split
from src.model import build_model
from src.utils import find_experiment, load_config, model_path, results_dir, set_seed


def train(cfg, exp_name):
    exp = find_experiment(cfg, exp_name)
    set_seed(cfg["seed"])

    train_df = read_split(cfg, exp["name"])
    val_df = read_split(cfg, "val")
    print(f"[{exp_name}] train counts:\n{train_df['label'].value_counts().to_string()}")

    train_ds = make_dataset(cfg, train_df, shuffle=True, augment=cfg["train"].get("augment", False))
    val_ds = make_dataset(cfg, val_df)

    model = build_model(cfg)
    ckpt = model_path(exp_name)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg["train"]["epochs"], callbacks=callbacks, verbose=2
    )
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(results_dir(exp_name) / "history.csv")
    print(f"[{exp_name}] best checkpoint -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    train(load_config(), args.exp)


if __name__ == "__main__":
    main()
