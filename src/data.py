"""Turn a split CSV into a tf.data pipeline."""
import pandas as pd
import tensorflow as tf

from src.utils import ROOT


def read_split(cfg, name):
    """name: 'test', 'val' or an experiment name (-> train_<name>.csv)."""
    fname = f"{name}.csv" if name in ("test", "val") else f"train_{name}.csv"
    path = ROOT / cfg["data"]["splits_dir"] / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare_data` first.")
    return pd.read_csv(path)


def make_dataset(cfg, df, shuffle=False, augment=False):
    classes = cfg["data"]["classes"]
    h, w = cfg["data"]["image_size"]
    raw_dir = str(ROOT / cfg["data"]["raw_dir"])

    paths = [f"{raw_dir}/{p}" for p in df["filepath"]]
    labels = [classes.index(l) for l in df["label"]]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (h, w))
        return tf.cast(img, tf.float32), label  # model handles its own scaling

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        flip = tf.keras.layers.RandomFlip("horizontal", seed=cfg["seed"])
        ds = ds.map(lambda x, y: (flip(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg["train"]["batch_size"]).prefetch(tf.data.AUTOTUNE)
