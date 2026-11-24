"""
F20DL Group 15 - download the Fitzpatrick17k skin images for our 5 conditions.

HOW TO RUN (Windows):
  1. Put this file in your Datasets folder (next to the 5 CSVs).
  2. Click the folder's address bar, type  cmd  and press Enter.
  3. Run:  python download_images.py
     (if lots of DermaAmin images are still missing, try:  python download_images.py --deep
      - slower, searches more upload months on the new DermaAmin site)

WHAT IT MAKES (inside the Datasets folder):
  Fitzpatrick17k_CNN_Images/
      allergic_contact_dermatitis/  allergic_contact_dermatitis_0001.jpg ...
      basal_cell_carcinoma/         basal_cell_carcinoma_0001.jpg ...
      psoriasis/  seborrheic_dermatitis/  vitiligo/
      manifest.csv          <- every image: file name, label, md5hash, skin type, url
      failed_downloads.csv  <- broken links and why

The number in each file name is the row number in that disease's CSV, so it
never changes between runs (gaps = broken links). Images already in the
"Images" folder (named by md5hash) are copied instead of re-downloaded.
Safe to re-run: anything already saved is skipped.
"""
import csv
import glob
import os
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

# DermaAmin moved to WordPress in 2026. Old links (dermaamin.com/site/images/...)
# are dead, but the same file names now live at
#   https://dermaamin.com/wp-content/uploads/<year>/<month>/<file name>
# By default we only try 2026/02 (where the migrated images seem to be).
# Run with  --deep  to also search every month from 2025/01 to 2026/11.
DEEP = "--deep" in sys.argv
if DEEP:
    DERMAAMIN_MONTHS = ["2026/02"] + [f"{y}/{m:02d}" for y in (2026, 2025)
                                      for m in range(12, 0, -1)
                                      if not (y == 2026 and m in (2, 12))]
else:
    DERMAAMIN_MONTHS = ["2026/02"]
month_hits = Counter()
hits_lock = threading.Lock()

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "Fitzpatrick17k_CNN_Images")
OLD_IMAGES = os.path.join(BASE, "Images")  # the 46 images we already had
HEADERS = {"User-Agent": "Mozilla/5.0 (F20DL coursework, Heriot-Watt University)"}


def image_ext(data):
    """Return a file extension if the bytes are a real image, else None."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"GIF8":
        return ".gif"
    return None  # probably an HTML error page


def load_jobs():
    jobs = []
    csv_files = sorted(glob.glob(os.path.join(BASE, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {BASE}. Put this script next to the CSVs.")
    for path in csv_files:
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=1):
                label = row["label"].strip().replace(" ", "_")
                row["folder"] = label
                row["stem"] = f"{label}_{i:04d}"
                jobs.append(row)
    return jobs


def already_saved(row):
    for ext in (".jpg", ".png", ".gif"):
        p = os.path.join(OUT_DIR, row["folder"], row["stem"] + ext)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def fetch(row):
    out_dir = os.path.join(OUT_DIR, row["folder"])
    os.makedirs(out_dir, exist_ok=True)

    existing = already_saved(row)
    if existing:
        return row, "skipped", os.path.basename(existing)

    # Reuse an image we already have locally (saved as <md5hash>.jpg)
    for old in glob.glob(os.path.join(OLD_IMAGES, "**", row["md5hash"] + ".*"), recursive=True):
        dest = os.path.join(out_dir, row["stem"] + os.path.splitext(old)[1].lower())
        shutil.copyfile(old, dest)
        return row, "copied", os.path.basename(dest)

    if "dermaamin.com/site/images/" in row["url"]:
        file_name = row["url"].rsplit("/", 1)[-1]
        with hits_lock:  # try the months that have worked most often first
            months = sorted(DERMAAMIN_MONTHS, key=lambda m: -month_hits[m])
        candidates = [(m, f"https://dermaamin.com/wp-content/uploads/{m}/{file_name}")
                      for m in months]
    else:
        candidates = [(None, row["url"])]

    last_error = "not found"
    for month, url in candidates:
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=25) as r:
                data = r.read()
        except Exception as e:
            last_error = str(e)
            continue
        ext = image_ext(data)
        if ext is None:
            last_error = "not an image (dead link or error page)"
            continue
        dest = os.path.join(out_dir, row["stem"] + ext)
        with open(dest, "wb") as f:
            f.write(data)
        if month:
            with hits_lock:
                month_hits[month] += 1
        row["url_used"] = url
        return row, "downloaded", os.path.basename(dest)

    return row, f"failed: {last_error}", ""


def main():
    jobs = load_jobs()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(jobs)} images listed. Saving to: {OUT_DIR}\n")

    results, counts = [], {}
    with ThreadPoolExecutor(max_workers=8) as pool:  # be gentle with the sites
        futures = [pool.submit(fetch, r) for r in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            row, status, fname = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            results.append((row, status, fname))
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {counts}")

    results.sort(key=lambda x: (x[0]["folder"], x[0]["stem"]))
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "md5hash", "fitzpatrick_scale",
                    "three_partition_label", "source_url", "downloaded_from"])
        for row, status, fname in results:
            if fname:
                w.writerow([f"{row['folder']}/{fname}", row["label"], row["md5hash"],
                            row["fitzpatrick_scale"], row["three_partition_label"], row["url"],
                            row.get("url_used", row["url"])])

    with open(os.path.join(OUT_DIR, "failed_downloads.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "md5hash", "url", "reason"])
        for row, status, fname in results:
            if status.startswith("failed"):
                w.writerow([row["label"], row["md5hash"], row["url"], status])

    print("\nImages per disease:")
    for folder in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, folder)
        if os.path.isdir(p):
            print(f"  {folder:30s} {len(os.listdir(p))}")
    if month_hits:
        print("\nDermaAmin images found per upload month:", dict(month_hits))
    print("\nDone. Totals:", counts)


if __name__ == "__main__":
    main()
