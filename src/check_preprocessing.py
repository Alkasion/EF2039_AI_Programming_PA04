"""Why resize-then-crop instead of resizing the whole photo to 224x224?

Compares how well a plain sharpness measure (variance of the Laplacian) separates sharp from blurred
photos under both preprocessings, as ROC-AUC. Only the 250 train/validation scenes are used; the
test scenes are never touched. Prints the numbers quoted in the README (Data > Preprocessing).

Usage: python -m src.check_preprocessing      (needs data/raw, about 1 minute)
"""
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import read_split
from src.prepare_data import index_raw, preprocess_image
from src.utils import load_config


def laplacian_var(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    cfg = load_config()
    size = cfg["data"]["image_size"]
    scenes = index_raw(cfg)
    test_scenes = set(read_split(cfg, "test")["scene"])
    ids = [s for s in sorted(scenes) if s not in test_scenes]
    variants = {"resize short side + center crop": lambda im: preprocess_image(im, cfg),
                "resize whole photo": lambda im: cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)}
    val = {(v, t): [] for v in variants for t in ("sharp", "defocus", "motion")}
    for sid in ids:
        for t in ("sharp", "defocus", "motion"):
            img = cv2.imread(str(scenes[sid][t]))
            for v, fn in variants.items():
                val[v, t].append(laplacian_var(fn(img)))
    print(f"Laplacian-variance ROC-AUC, sharp vs blurred, {len(ids)} train/val scenes")
    y = np.r_[np.zeros(len(ids)), np.ones(len(ids))]
    for v in variants:
        aucs = {t: roc_auc_score(y, -np.r_[val[v, "sharp"], val[v, t]]) for t in ("defocus", "motion")}
        print(f"  {v:32s} defocus {aucs['defocus']:.3f}   motion {aucs['motion']:.3f}")


if __name__ == "__main__":
    main()
