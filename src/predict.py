"""Inference on your own photos: keep (sharp) or reject (blurry).

Each photo gets the same preprocessing as the training data (short side to 448 px, center-crop
224x224), so the decision is based on the centre of the photo.

Usage:
    python -m src.predict photo1.jpg photo2.jpg          # real_mixed, seed 42
    python -m src.predict my_photos/ --exp synth_mixed --seed 43 --threshold 0.5 --csv decisions.csv

Needs the checkpoint models/<exp>_seed<seed>.pt (train it or download it, see models/README.md).
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from src.data import normalize, to_tensor
from src.model import load_trained
from src.prepare_data import preprocess_image
from src.utils import find_experiment, get_device, load_config, model_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(inputs):
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)
    return paths


@torch.no_grad()
def predict_paths(cfg, model, paths, device, threshold=0.5):
    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip {p}: not a readable image")
            continue
        x = normalize(to_tensor(preprocess_image(img, cfg))).unsqueeze(0).to(device)
        p_blurry = torch.softmax(model(x), dim=1)[0, 1].item()
        rows.append({"file": str(p), "p_blurry": p_blurry,
                     "decision": "REJECT (blurry)" if p_blurry >= threshold else "KEEP (sharp)"})
    return pd.DataFrame(rows, columns=["file", "p_blurry", "decision"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="image files and/or folders of images")
    ap.add_argument("--exp", default="real_mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="reject when P(blurry) >= threshold")
    ap.add_argument("--csv", help="also write the decisions to this CSV file")
    args = ap.parse_args()
    cfg = load_config()
    find_experiment(cfg, args.exp)
    ckpt = model_path(args.exp, args.seed)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found. Train it or download it, see models/README.md.")
    device = get_device()
    df = predict_paths(cfg, load_trained(cfg, ckpt, device), list_images(args.inputs), device, args.threshold)
    if df.empty:
        raise SystemExit("No readable images given.")
    print(df.to_string(index=False, float_format="%.3f"))
    if args.csv:
        df.to_csv(args.csv, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
