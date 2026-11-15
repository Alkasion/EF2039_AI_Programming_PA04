"""Download the Blur dataset, preprocess it, make synthetic blur, and build every split.

Steps
    1. --download: fetch kwentar/blur-dataset from Kaggle into data/raw/ (no Kaggle login needed).
    2. Preprocess every real photo: resize so the short side is 448 px, center-crop 224x224,
       save as lossless PNG in data/processed/real/<sharp|defocus|motion>/<scene>.png.
    3. Synthetic blur: blur each SHARP photo at the 448 px scale with a random disk (defocus)
       or linear (motion) kernel, then crop the same way
       -> data/processed/synthetic/<defocus|motion>/<scene>.png.
    4. Split the 350 scenes (seed 42) into test / val / train pool and write CSVs to data/splits/.

Usage:
    python -m src.prepare_data --download   # first time
    python -m src.prepare_data              # rebuild processed images + splits from data/raw
"""
import argparse
import re
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, processed_dir, raw_dir, splits_dir

RAW_FOLDERS = {"sharp": "sharp", "defocus": "defocused_blurred", "motion": "motion_blurred"}
NAME_RE = re.compile(r"^(\d+)_(.+)_([SFM])\.(jpe?g|png)$", re.IGNORECASE)
COLUMNS = ["filepath", "label", "blur_type", "source", "scene", "device"]


# ---------------------------------------------------------------- download
def download_dataset(cfg):
    out = raw_dir(cfg)
    if all((out / f).is_dir() and any((out / f).iterdir()) for f in RAW_FOLDERS.values()):
        print(f"{out} already populated; skipping download")
        return
    url = f"https://www.kaggle.com/api/v1/datasets/download/{cfg['data']['kaggle_id']}"
    dl_dir = ROOT / "data" / "download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "blur-dataset.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"{zip_path} is not a valid zip (interrupted download?); downloading again")
        zip_path.unlink()
    if not zip_path.exists():
        print(f"Downloading {url} (~1.6 GB) ...")
        # Write to a temporary name first, so an interrupted download is never mistaken for the archive.
        part = zip_path.with_name(zip_path.name + ".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if not zipfile.is_zipfile(part):
            part.unlink()
            raise IOError(f"Download from {url} is not a zip archive. Download the dataset by hand from "
                          f"https://www.kaggle.com/datasets/{cfg['data']['kaggle_id']} and save it as {zip_path}.")
        part.replace(zip_path)
    # The archive holds the same images twice; use the 'blur_dataset_scaled/' copy.
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if (len(parts) == 3 and parts[0] == "blur_dataset_scaled" and parts[1] in RAW_FOLDERS.values()
                    and parts[2] not in ("", ".", "..")):
                dst = out / parts[1] / parts[2]
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
    print(f"Extracted images to {out}")


# ---------------------------------------------------------------- image ops
def resize_short_side(img, short):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def center_crop(img, size):
    h, w = img.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return img[y:y + size, x:x + size]


def preprocess_image(img, cfg):
    """The preprocessing every photo goes through: short side to 448 px, then center-crop 224x224."""
    return center_crop(resize_short_side(img, cfg["data"]["resize_short_side"]), cfg["data"]["image_size"])


def disk_kernel(radius):
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    return k / k.sum()


def motion_kernel(length, angle_deg):
    length = int(length) | 1  # odd size so the line is centred
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length), flags=cv2.INTER_LINEAR)
    return k / k.sum()


def synthetic_blur(img448, blur_type, rng, syn_cfg):
    if blur_type == "defocus":
        lo, hi = syn_cfg["defocus_radius_px"]
        kernel = disk_kernel(rng.uniform(lo, hi))
    else:
        lo, hi = syn_cfg["motion_length_px"]
        kernel = motion_kernel(rng.integers(lo, hi + 1), rng.uniform(0, 180))
    return cv2.filter2D(img448, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- preprocessing
def index_raw(cfg):
    """Return {scene_id: {'device': str, 'sharp': path, 'defocus': path, 'motion': path}}."""
    scenes = {}
    for blur_type, folder in RAW_FOLDERS.items():
        d = raw_dir(cfg) / folder
        if not d.is_dir():
            raise FileNotFoundError(f"Missing {d}. Run `python -m src.prepare_data --download`.")
        for p in sorted(d.iterdir()):
            m = NAME_RE.match(p.name)
            if not m:
                continue
            sid = int(m.group(1))
            scenes.setdefault(sid, {"device": m.group(2)})[blur_type] = p
    complete = {s: v for s, v in scenes.items() if all(t in v for t in RAW_FOLDERS)}
    print(f"Found {len(complete)} complete scenes (sharp + defocus + motion)")
    return complete


def preprocess(cfg, scenes):
    short, size = cfg["data"]["resize_short_side"], cfg["data"]["image_size"]
    out = processed_dir(cfg)
    for sid, entry in sorted(scenes.items()):
        for blur_type in RAW_FOLDERS:
            img = cv2.imread(str(entry[blur_type]))
            if img is None:
                raise IOError(f"Unreadable image: {entry[blur_type]}")
            img448 = resize_short_side(img, short)  # synthetic blur is applied at this scale
            dst = out / "real" / blur_type / f"{sid}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), center_crop(img448, size))
            if blur_type != "sharp":
                continue
            for k, syn_type in enumerate(("defocus", "motion")):
                rng = np.random.default_rng([cfg["seed"], sid, k])  # same blur every rebuild
                syn = synthetic_blur(img448, syn_type, rng, cfg["synthetic"])
                dst = out / "synthetic" / syn_type / f"{sid}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), center_crop(syn, size))
    print(f"Wrote processed images to {out}")


# ---------------------------------------------------------------- splits
def row(scenes, sid, blur_type, source):
    label = "sharp" if blur_type == "sharp" else "blurry"
    path = f"{source}/{blur_type}/{sid}.png"
    return (path, label, blur_type, source, sid, scenes[sid]["device"])


def recipe_rows(scenes, scene_ids, blur_spec):
    """One sharp photo per scene + one blurry photo per scene, blur types split over the scenes."""
    rows = [row(scenes, s, "sharp", "real") for s in scene_ids]
    for chunk, (blur_type, source) in zip(np.array_split(scene_ids, len(blur_spec)), blur_spec.items()):
        rows += [row(scenes, s, blur_type, source) for s in chunk]
    return rows


def save(rows, path):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def build_splits(cfg, scenes):
    out = splits_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    n_test, n_val = cfg["data"]["n_test_scenes"], cfg["data"]["n_val_scenes"]
    ids = np.array(sorted(scenes))
    ids = ids[np.random.default_rng(cfg["seed"]).permutation(len(ids))]
    test_ids, val_ids, pool = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]

    # Fixed test set: every blur type, real photos only.
    save([row(scenes, s, t, "real") for s in test_ids for t in RAW_FOLDERS], out / "test.csv")

    summary = []
    for exp in cfg["experiments"]:
        n = exp["train_scenes"]
        if n > len(pool):
            raise ValueError(f"{exp['name']}: wants {n} train scenes, pool has {len(pool)}")
        train_rows = recipe_rows(scenes, pool[:n], exp["blur"])
        val_rows = recipe_rows(scenes, val_ids, exp["blur"])
        save(train_rows, out / f"train_{exp['name']}.csv")
        save(val_rows, out / f"val_{exp['name']}.csv")
        for split, rows in (("train", train_rows), ("val", val_rows)):
            df = pd.DataFrame(rows, columns=COLUMNS)
            for (bt, src), k in df.groupby(["blur_type", "source"]).size().items():
                summary.append({"experiment": exp["name"], "split": split, "blur_type": bt,
                                "source": src, "n_images": k})
    for bt in RAW_FOLDERS:
        summary.append({"experiment": "ALL", "split": "test", "blur_type": bt,
                        "source": "real", "n_images": len(test_ids)})
    pd.DataFrame(summary).to_csv(out / "split_summary.csv", index=False)
    print(f"Scenes: test {len(test_ids)}, val {len(val_ids)}, train pool {len(pool)}")
    print(f"Wrote splits to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--splits-only", action="store_true", help="skip image preprocessing")
    args = ap.parse_args()
    cfg = load_config()
    if args.download:
        download_dataset(cfg)
    scenes = index_raw(cfg)
    if not args.splits_only:
        preprocess(cfg, scenes)
    build_splits(cfg, scenes)


if __name__ == "__main__":
    main()
