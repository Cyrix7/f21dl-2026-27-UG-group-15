"""
F20DL Group 15 - clean the downloaded skin images and make ONE shared train/val/test split.

HOW TO RUN (Windows), after download_images.py has finished:
  1. Put this file in the Datasets folder (same place as download_images.py).
  2. In cmd, in that folder, run:  python clean_and_split.py
  Needs: numpy, pillow  (both come with Anaconda; otherwise: pip install numpy pillow)

WHAT IT DOES (each step is logged so we can justify it in the report):
  1. Drops images from source folders that are a different disease from their label
     (e.g. "grovers_disease" filed under allergic contact dermatitis) - see WRONG_SOURCE_FOLDERS.
  2. Drops images the dataset's own dermatologist check marked "3 Wrongly labelled".
  3. Drops files that can't be opened as images.
  4. Finds exact and near-duplicate photos (perceptual hash). Keeps one copy per group;
     if copies have DIFFERENT labels, drops them all (we can't tell which label is right).
  5. Saves the cleaned images at their ORIGINAL size (load_data.py crops/resizes when
     loading). Set IMG_SIZE below to save pre-resized square copies instead.
  6. Stratified 70/15/15 train/val/test split with a fixed seed, so everyone gets
     exactly the same split.

WHAT IT MAKES (inside the Datasets folder):
  Fitzpatrick17k_processed/
      images/<disease>/<disease>_0001.jpg ...   <- cleaned images (original size)
      dataset.csv          <- ONE row per image: file, label, label_id, is_malignant,
                              fitzpatrick_scale, source_folder, split   (USE THIS)
      removed_images.csv   <- every dropped image and why
      duplicate_groups.csv <- the duplicate groups found (check a few by eye)
"""
import csv
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------- settings ---
SEED = 42
IMG_SIZE = None              # None = keep original size (load_data.py resizes when loading).
                             # Set e.g. 128 to save pre-resized square copies instead.
SPLIT = (0.70, 0.15, 0.15)   # train / val / test
NEAR_DUP_BITS = 12           # max hash difference (out of 256 bits) to even consider a pair
NEAR_DUP_PIXEL = 0.03        # ...and the photos must ALSO match pixel-by-pixel (mean diff < 3%)

# Source folders (from the old DermaAmin URLs) whose disease does NOT match the label.
# Edit this list if the team disagrees with any of these.
WRONG_SOURCE_FOLDERS = {
    "allergic contact dermatitis": {
        "grovers_disease_transient_acantholytic_dermatosis",  # Grover's disease
        "salmon-patches",                                      # a birthmark
        "juvenile-rheumatoid-arthritis",                       # joint disease
        "bee_sting",                                           # insect sting
        "allergic-reaction-to-medication",                     # drug reaction, not contact
    },
    "psoriasis": {
        "filariasis",                       # parasitic worm infection
        "oxyuriasis",                       # pinworm infection
        "parapsoriasis-large-plaque",       # parapsoriasis is a different disease
        "parapsoriasis-small-plaque-type",
        "parapsoriasis-en-goutte",
    },
    "seborrheic dermatitis": {
        "seborrhoeic-warts",                # seborrhoeic keratosis, a different disease
    },
}
# ------------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "Fitzpatrick17k_CNN_Images")
OUT_DIR = os.path.join(BASE, "Fitzpatrick17k_processed")
OUT_IMG = os.path.join(OUT_DIR, "images")


def source_folder(url):
    if "dermaamin.com/site/images/" in url:
        return url.rsplit("/", 2)[-2]
    if "atlasdermatologico" in url:
        return "atlasdermatologico"
    return "unknown"


def dhash(img, size=16):
    """256-bit difference hash: similar-looking photos get similar hashes."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def thumb(img, size=32):
    """Small colour copy used to confirm a suspected duplicate pixel-by-pixel."""
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0


def centre_square(img, size):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def main():
    manifest = os.path.join(RAW_DIR, "manifest.csv")
    if not os.path.exists(manifest):
        raise SystemExit(f"Can't find {manifest}. Run download_images.py first.")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # qc column lives in the original CSVs, not the manifest
    qc = {}
    for name in os.listdir(BASE):
        if name.endswith(".csv"):
            with open(os.path.join(BASE, name), newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if "md5hash" in r and "qc" in r:
                        qc[r["md5hash"]] = r["qc"]

    if os.path.isdir(OUT_DIR):  # start fresh so no stale images are left from older runs
        import shutil
        shutil.rmtree(OUT_DIR)

    print(f"Starting with {len(rows)} downloaded images\n")
    removed = []

    def drop(r, reason):
        removed.append({"file": r["file"], "label": r["label"], "reason": reason})

    # Steps 1-3: wrong source folder, dermatologist-flagged, unreadable
    kept, hashes, thumbs = [], [], []
    for r in rows:
        r["source_folder"] = source_folder(r["source_url"])
        if r["source_folder"] in WRONG_SOURCE_FOLDERS.get(r["label"], set()):
            drop(r, f"wrong disease in source folder ({r['source_folder']})")
            continue
        if qc.get(r["md5hash"], "").startswith("3"):
            drop(r, "dataset QC: wrongly labelled")
            continue
        path = os.path.join(RAW_DIR, r["file"])
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                if min(im.size) < 64:
                    drop(r, f"too small ({im.size[0]}x{im.size[1]})")
                    continue
                hashes.append(dhash(im))
                thumbs.append(thumb(im))
        except Exception as e:
            drop(r, f"unreadable image ({e.__class__.__name__})")
            continue
        kept.append(r)

    # Step 4: near-duplicates via union-find on hash distance
    n = len(kept)
    H = np.array(hashes, dtype=bool)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 256):
        block = H[start:start + 256]
        dist = (block[:, None, :] != H[None, :, :]).sum(-1)
        for bi, j in zip(*np.nonzero(dist <= NEAR_DUP_BITS)):
            i = start + bi
            # only link a pair if the photos ALSO match pixel-by-pixel,
            # so lookalike skin photos don't get chained together
            if i < j and np.abs(thumbs[i] - thumbs[j]).mean() < NEAR_DUP_PIXEL:
                parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    final, dup_rows, gid = [], [], 0
    for members in groups.values():
        if len(members) == 1:
            final.append(kept[members[0]])
            continue
        gid += 1
        labels = {kept[i]["label"] for i in members}
        for k, i in enumerate(sorted(members, key=lambda i: kept[i]["file"])):
            dup_rows.append({"group": gid, "file": kept[i]["file"], "label": kept[i]["label"]})
            if len(labels) > 1:
                drop(kept[i], f"duplicate with conflicting labels (group {gid})")
            elif k == 0:
                final.append(kept[i])
            else:
                drop(kept[i], f"duplicate of another image (group {gid})")

    # Step 5: save resized images
    labels_sorted = sorted({r["label"] for r in final})
    label_id = {l: i for i, l in enumerate(labels_sorted)}
    for r in final:
        src = os.path.join(RAW_DIR, r["file"])
        dest_rel = os.path.splitext(r["file"])[0] + ".jpg"
        dest = os.path.join(OUT_IMG, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if IMG_SIZE:
                im = centre_square(im, IMG_SIZE)
            im.save(dest, quality=95)
        r["out_file"] = "images/" + dest_rel.replace("\\", "/")

    # Step 6: stratified split, fixed seed
    rng = np.random.default_rng(SEED)
    by_label = defaultdict(list)
    for r in sorted(final, key=lambda r: r["file"]):
        by_label[r["label"]].append(r)
    for label, items in by_label.items():
        idx = rng.permutation(len(items))
        n_train = round(len(items) * SPLIT[0])
        n_val = round(len(items) * SPLIT[1])
        for rank, i in enumerate(idx):
            items[i]["split"] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    # Write outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["file", "label", "label_id", "is_malignant", "fitzpatrick_scale",
            "source_folder", "md5hash", "split"]
    with open(os.path.join(OUT_DIR, "dataset.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(final, key=lambda r: r["out_file"]):
            w.writerow({"file": r["out_file"], "label": r["label"], "label_id": label_id[r["label"]],
                        "is_malignant": int(r["three_partition_label"] == "malignant"),
                        "fitzpatrick_scale": r["fitzpatrick_scale"],
                        "source_folder": r["source_folder"], "md5hash": r["md5hash"],
                        "split": r["split"]})
    for name, data, fields in (("removed_images.csv", removed, ["file", "label", "reason"]),
                               ("duplicate_groups.csv", dup_rows, ["group", "file", "label"])):
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    # Summary
    print("Removed images:")
    for reason, c in Counter(r["reason"].split(" (")[0] for r in removed).most_common():
        print(f"  {c:4d}  {reason}")
    print(f"\nFinal dataset: {len(final)} images  (label_id in brackets)\n")
    print(f"  {'disease':32s}{'train':>7s}{'val':>6s}{'test':>6s}{'total':>7s}")
    for l in labels_sorted:
        c = Counter(r["split"] for r in by_label[l])
        print(f"  {l + ' [' + str(label_id[l]) + ']':32s}{c['train']:7d}{c['val']:6d}{c['test']:6d}"
              f"{len(by_label[l]):7d}")
    c = Counter(r["split"] for r in final)
    print(f"  {'TOTAL':32s}{c['train']:7d}{c['val']:6d}{c['test']:6d}{len(final):7d}")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
