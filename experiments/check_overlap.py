"""
Check whether bright-field images from an external analysis (e.g. the cells behind a figure) are part of the
ML training, validation or test sets.

    python -m experiments.check_overlap --query "Fig3_1kPa=D:/Lautaro/Fig3/1kPa" --query "Fig3_10kPa=D:/Lautaro/Fig3/10kPa"

Each --query is LABEL=FOLDER. The folder is searched recursively for bright-field images (file name containing
"BF", .tif/.tiff/.png/.jpg). Every image is compared with the images listed in experiments/splits/*.csv:
  1. exact match: identical file content (MD5, the `bf_md5` column of the split files);
  2. near match: same image re-saved, converted to 8 bit or renamed. Both images are reduced to 128 x 128 px
     and compared by Pearson correlation; r >= --near (default 0.95) counts as the same image.
Near matching needs the dataset images (--dataset-root, default experiments/layout.DEFAULT_DATASET_ROOT).
The thumbnails of the dataset are cached in experiments/runs/bf_thumbs.npz (not tracked by git).

Output: one row per query image in experiments/results/dataset_audit/overlap_<LABEL>.csv and a summary of how
many images of each label fall in the training, validation or test sets, or in none of them.
"""
import argparse
import glob
import hashlib
import os

import cv2
import numpy as np
import pandas as pd

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, RUNS_DIR

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
AUDIT_DIR = os.path.join(RESULTS_DIR, "dataset_audit")
EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
THUMB = 128


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thumb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    t = cv2.resize(im.astype(np.float32), (THUMB, THUMB), interpolation=cv2.INTER_AREA).ravel()
    s = t.std()
    return (t - t.mean()) / s if s > 0 else None


def load_splits():
    rows = []
    for f in sorted(glob.glob(os.path.join(SPLITS_DIR, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("zenodo"):
            continue
        d = pd.read_csv(f)
        d["split_file"] = name
        d["role"] = "train" if name.endswith("_train") else ("validation" if name.endswith("_val") else "test")
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def dataset_thumbs(splits, dataset_root):
    cache = os.path.join(RUNS_DIR, "bf_thumbs.npz")   # not tracked by git
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if set(z["md5"]) >= set(splits["bf_md5"]):
            return list(z["md5"]), z["thumbs"]
    keys, thumbs, missing = [], [], 0
    for h, g in splits.groupby("bf_md5"):
        d = os.path.join(dataset_root, g["sample_dir"].iloc[0])
        bf = None
        if os.path.isdir(d):
            bf = next((os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith("BF_001.tif")), None)
        t = thumb(bf) if bf else None
        if t is None:
            missing += 1
            continue
        keys.append(h)
        thumbs.append(t)
    if missing:
        print(f"  note: {missing} dataset images not found under {dataset_root}; near matching skips them")
    thumbs = np.stack(thumbs) if thumbs else np.zeros((0, THUMB * THUMB), np.float32)
    os.makedirs(RUNS_DIR, exist_ok=True)
    np.savez_compressed(cache, md5=np.array(keys), thumbs=thumbs)
    return keys, thumbs


def describe(splits, h):
    g = splits[splits["bf_md5"] == h]
    return "; ".join(sorted(set(g["role"] + ":" + g["set"].astype(str)))), ";".join(sorted(set(g["role"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", required=True, help="LABEL=FOLDER (repeatable)")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--near", type=float, default=0.95, help="correlation threshold for a near match")
    a = ap.parse_args()

    splits = load_splits()
    md5_set = set(splits["bf_md5"])
    keys, thumbs = dataset_thumbs(splits, a.dataset_root)
    summary = []
    for q in a.query:
        label, folder = q.split("=", 1)
        files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                 if os.path.isfile(f) and "bf" in os.path.basename(f).lower() and f.lower().endswith(EXTS)]
        out = []
        for f in sorted(files):
            h = md5(f)
            row = {"label": label, "file": f, "md5": h, "match": "none", "roles": "", "where": "", "near_r": np.nan}
            if h in md5_set:
                row["where"], row["roles"] = describe(splits, h)
                row["match"] = "exact"
            elif len(keys):
                t = thumb(f)
                if t is not None:
                    r = thumbs @ t / t.size
                    k = int(np.argmax(r))
                    row["near_r"] = float(r[k])
                    if r[k] >= a.near:
                        row["where"], row["roles"] = describe(splits, keys[k])
                        row["match"] = "near"
            out.append(row)
        df = pd.DataFrame(out)
        os.makedirs(AUDIT_DIR, exist_ok=True)
        df.to_csv(os.path.join(AUDIT_DIR, f"overlap_{label}.csv"), index=False)
        n = len(df)
        cnt = {r: int(df["roles"].str.contains(r).sum()) for r in ("train", "validation", "test")}
        summary.append({"label": label, "images": n, **cnt, "not_in_ML_dataset": int((df["match"] == "none").sum()),
                        "exact": int((df["match"] == "exact").sum()), "near": int((df["match"] == "near").sum())})
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    print(f"\nper-image results: {AUDIT_DIR}/overlap_<label>.csv")
    if s["train"].sum():
        print("WARNING: some query images are in the TRAINING set (see the 'where' column).")


if __name__ == "__main__":
    main()
