"""
F20DL Group 15 - shared data loader. EVERYONE should load data through this,
so all models use the same images, labels, split and preprocessing.

The cleaned images are stored at their original size. This loader centre-crops
each one to a square and resizes it (default 128x128) as it loads.

SETUP: unzip Fitzpatrick17k_processed.zip next to your notebook, then at the top
of the notebook:

    import sys
    sys.path.append("Fitzpatrick17k_processed")   # where load_data.py lives
    from load_data import load_split, CLASS_NAMES

(If your notebook is INSIDE the Fitzpatrick17k_processed folder, you can skip the
two sys lines.)

Usage:

    X_train, y_train = load_split("train")                 # (n, 128, 128, 3), values 0-1
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    # Classic models (R3) need flat feature vectors and work better on smaller images:
    X_train, y_train = load_split("train", size=32, flatten=True)   # (n, 3072)

    # Pretrained CNNs usually want 224x224:
    X_train, y_train = load_split("train", size=224)

    # Cancer vs not-cancer instead of 5 diseases:
    X_train, y_train = load_split("train", binary=True)    # y: 1 = basal cell carcinoma

"""
import csv
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works whether this file sits INSIDE the Fitzpatrick17k_processed folder
# or NEXT TO it.
if os.path.exists(os.path.join(_HERE, "dataset.csv")):
    DATA_DIR = _HERE
else:
    DATA_DIR = os.path.join(_HERE, "Fitzpatrick17k_processed")


def _rows():
    with open(os.path.join(DATA_DIR, "dataset.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CLASS_NAMES = [name for _, name in sorted({(int(r["label_id"]), r["label"]) for r in _rows()})]


def _square(im, size):
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def load_split(split, size=128, flatten=False, binary=False):
    """Return (X, y) for split = 'train', 'val' or 'test'. Images are centre-cropped
    to a square and resized to size x size."""
    rows = [r for r in _rows() if r["split"] == split]
    if not rows:
        raise ValueError("split must be 'train', 'val' or 'test'")
    X = []
    for r in rows:
        with Image.open(os.path.join(DATA_DIR, r["file"])) as im:
            X.append(np.asarray(_square(im.convert("RGB"), size), dtype=np.float32) / 255.0)
    X = np.stack(X)
    if flatten:
        X = X.reshape(len(X), -1)
    key = "is_malignant" if binary else "label_id"
    y = np.array([int(r[key]) for r in rows])
    return X, y


def load_info(split=None):
    """The dataset.csv rows (label, skin type, source...) as a list of dicts."""
    rows = _rows()
    return [r for r in rows if split is None or r["split"] == split]
