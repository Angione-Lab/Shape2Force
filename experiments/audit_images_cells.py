"""
Audit of the cell folders behind the paper figures (`images_cells/`) against the ML dataset.

    python -m experiments.audit_images_cells

Defaults: images_cells = <dataset root>/images_cells and the ML dataset = the sample folders listed in
experiments/splits/*.csv under the same dataset root (Shape2Force/dataset, i.e. layout.DEFAULT_DATASET_ROOT;
the folders that experiments/prepare_dataset.py copied from the legacy tree). Optionally add
--legacy-root C:/Users/Kaveh/Documents/cellforce/dataset/single_cell to also search folders of the old tree
that were never part of the ML splits.

For every position folder in images_cells/<condition>/<set>/<Position_*>/ it answers:
  1. Is the bright-field image (BF_001.tif) part of the ML dataset, and in which role
     (training / validation / independent test)?  Exact match = identical file (MD5);
     near match = same image re-saved (128 x 128 thumbnails, Pearson r >= --near).
  2. If not in the ML dataset, is it in the legacy tree (only with --legacy-root)? If not, it is NEW.
  3. Does its condition (1 / 6 / 10 kPa, DMSO, Y-27632, TGF-b) agree with the condition of the
     matching ML sample? A mismatch means the same image is used under two conditions.
  4. Is the same bright-field image used in more than one condition or set inside images_cells?
  5. For matched samples, is the traction map the same? The colour map
     (Cell1_NobarUnconstrainedTractions_1.jpg) is converted to grey by inverting the jet colour
     map and correlated with the ML grey map (gt_map_r; about 1 = same TFM result).

Outputs in experiments/results/images_cells_audit/ :
  positions.csv            one row per position folder (all checks above + Manual_Quantification values)
  summary.csv              per condition x set: counts by status
  condition_mismatch.csv   positions whose image appears in the ML dataset or images_cells under another condition
  new_samples.csv          positions not found anywhere (candidates for an extended dataset)
  report.txt               the same, in words

Caches (MD5s and thumbnails) are written to experiments/runs/ (not tracked by git).
"""
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
import pandas as pd

from experiments.check_overlap import THUMB, load_splits, md5, thumb
from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, RUNS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "images_cells_audit")
CACHE = os.path.join(RUNS_DIR, "audit_images_cells_cache.json")
THUMBS = os.path.join(RUNS_DIR, "audit_images_cells_thumbs.npz")

CONDITION_OF_FOLDER = {          # images_cells condition folder -> condition label used in the ML split files
    "01kPa": "1 kPa collagen", "06kPa_Untreated": "6 kPa collagen", "10_kPa": "10 kPa collagen",
    "06kPa_DMSO": "DMSO", "06kPa_Y27632": "Y-27632", "06kPa_TGF-b": "TGF-beta",
}
QUANT_LABELS = ["Cell Area/1000", "Cell Area", "GT Force (Lauti) / Area", "GT Force (Lauti)",
                "Fred Force / Area", "Pred Force / Area", "Pred Force"]


# ----------------------------------------------------------------------------- helpers
def is_real(name):
    return not name.startswith("._") and not name.startswith(".")


def find_file(folder, pattern):
    hits = [f for f in sorted(os.listdir(folder)) if is_real(f) and re.fullmatch(pattern, f, flags=re.I)]
    return os.path.join(folder, hits[0]) if hits else None


def parse_quant(path):
    """Manual_Quantification.txt -> dict. Handles 'label: value', 'label value' and label/value on separate lines."""
    if not path or not os.path.exists(path):
        return {}
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
    num = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
    out, pending = {}, []
    for l in lines[1:] if lines and "QUANTIFICATION" in lines[0].upper() else lines:
        lab = next((q for q in QUANT_LABELS if l.lower().startswith(q.lower())), None)
        rest = l[len(lab):] if lab else l
        m = num.findall(rest)
        if lab and m:
            out[lab] = float(m[-1].replace(",", "."))
        elif lab:
            pending.append(lab)
        elif m and pending:
            out[pending.pop(0)] = float(m[-1].replace(",", "."))
    return {("q_" + k.replace(" ", "_").replace("/", "per").replace("(", "").replace(")", "")): v for k, v in out.items()}


def jet_to_gray(path, size=256):
    """Invert a MATLAB-style jet colour map (nearest colour) -> values in [0, 1]."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].reshape(-1, 3).astype(np.float32)
    x = np.linspace(0, 1, 256)
    lut = np.stack([np.clip(1.5 - np.abs(4 * x - 3), 0, 1), np.clip(1.5 - np.abs(4 * x - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * x - 1), 0, 1)], 1) * 255
    idx = np.empty(len(im), np.int32)
    for s in range(0, len(im), 16384):
        d = ((im[s:s + 16384, None, :] - lut[None]) ** 2).sum(-1)
        idx[s:s + 16384] = d.argmin(1)
    return (idx / 255.0).reshape(size, size)


def gray_map(path, size=256):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if im is None else cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) / 255.0


def corr(a, b):
    if a is None or b is None or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


class Cache:
    """MD5 and thumbnail cache keyed by (path, size, mtime)."""
    def __init__(self):
        self.md5 = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        self.th = {}
        if os.path.exists(THUMBS):
            z = np.load(THUMBS, allow_pickle=True)
            self.th = dict(zip(z["keys"], z["thumbs"]))

    @staticmethod
    def key(p):
        st = os.stat(p)
        return f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}"

    def get_md5(self, p):
        k = self.key(p)
        if k not in self.md5:
            self.md5[k] = md5(p)
        return self.md5[k]

    def get_thumb(self, p):
        k = self.key(p)
        if k not in self.th:
            t = thumb(p)
            self.th[k] = t if t is not None else np.zeros(THUMB * THUMB, np.float32)
        return self.th[k]

    def save(self):
        os.makedirs(RUNS_DIR, exist_ok=True)
        json.dump(self.md5, open(CACHE, "w"))
        np.savez_compressed(THUMBS, keys=np.array(list(self.th)), thumbs=np.stack(list(self.th.values())))


# ----------------------------------------------------------------------------- collection
def walk_images_cells(root):
    rows = []
    for cond in sorted(d for d in os.listdir(root) if is_real(d) and os.path.isdir(os.path.join(root, d))):
        for sset in sorted(d for d in os.listdir(os.path.join(root, cond))
                           if is_real(d) and os.path.isdir(os.path.join(root, cond, d))):
            for pos in sorted(d for d in os.listdir(os.path.join(root, cond, sset))
                              if is_real(d) and os.path.isdir(os.path.join(root, cond, sset, d))):
                f = os.path.join(root, cond, sset, pos)
                m = re.match(r"Position_(\d+)(.*)", pos)
                rows.append(dict(
                    condition_folder=cond, condition=CONDITION_OF_FOLDER.get(cond, cond), set_folder=sset,
                    position=pos, pos_num=int(m.group(1)) if m else np.nan,
                    b_copy=bool(m and m.group(2).strip("_").lower() == "b"),
                    bf=find_file(f, r"BF_001\.tif{1,2}"), traction=find_file(f, r"Cell1_NobarUnconstrainedTractions_1\.jpe?g"),
                    rank_png=find_file(f, r"rank.*\.png"), quant=find_file(f, r"Manual_Quantification\.txt"),
                    rel=os.path.relpath(f, root)))
    return pd.DataFrame(rows)


def walk_legacy(root):
    """Every folder in the legacy tree that holds a bright-field image."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if is_real(d)]
        bf = [f for f in filenames if is_real(f) and re.fullmatch(r".*BF_001\.tif{1,2}", f, flags=re.I)]
        if bf:
            gray = [f for f in filenames if is_real(f) and f.lower().endswith("_gray.jpg")]
            out.append(dict(folder=os.path.relpath(dirpath, root), bf=os.path.join(dirpath, bf[0]),
                            gray=os.path.join(dirpath, gray[0]) if gray else None))
    return pd.DataFrame(out)


def walk_ml(splits, dataset_root):
    """The ML samples listed in the split files, as found under the dataset root."""
    out, missing = [], 0
    for h, g in splits.groupby("bf_md5"):
        d = os.path.join(dataset_root, g["sample_dir"].iloc[0])
        bf = find_file(d, r".*BF_001\.tif{1,2}") if os.path.isdir(d) else None
        if bf is None:
            missing += 1
            continue
        gray = find_file(d, r".*_gray\.jpe?g")
        out.append(dict(folder="ML:" + g["sample_dir"].iloc[0], bf=bf, gray=gray, md5=h))
    return pd.DataFrame(out), missing


def ml_where(splits):
    """md5 -> (roles, description, conditions, sample_dir of one copy)."""
    s = splits.copy()
    six = s["condition"] == "6 kPa collagen"
    s["label"] = s["set"].astype(str)
    s.loc[six, "label"] = s.loc[six, "legacy_folder"].astype(str).str.extract(r"(SET_[IVX]+)", expand=False) \
        .radd("6kPa_").fillna(s.loc[six, "set"].astype(str))
    s["desc"] = s["role"] + ":" + s["label"] + " [" + s["condition"].astype(str) + "]"
    g = s.groupby("bf_md5")
    return {h: dict(roles=";".join(sorted(set(x["role"]))), where="; ".join(sorted(set(x["desc"]))),
                    conditions=";".join(sorted(set(x["condition"].astype(str)))), sample_dir=x["sample_dir"].iloc[0])
            for h, x in g}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                    help="Shape2Force dataset folder (ML samples of the split files + images_cells)")
    ap.add_argument("--images-cells", default=None, help="default: <dataset-root>/images_cells")
    ap.add_argument("--legacy-root", default=None,
                    help="optional: old tree, e.g. C:/Users/Kaveh/Documents/cellforce/dataset/single_cell")
    ap.add_argument("--near", type=float, default=0.95)
    a = ap.parse_args()
    a.images_cells = a.images_cells or os.path.join(a.dataset_root, "images_cells")
    if not os.path.isdir(a.images_cells):
        raise SystemExit(f"images_cells not found: {a.images_cells}")
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = Cache()
    splits = load_splits()

    print("1/5 reading images_cells ...", flush=True)
    P = walk_images_cells(a.images_cells)
    print(f"    {len(P)} position folders, {P['bf'].notna().sum()} with BF_001.tif")
    print("2/5 reading the ML dataset (split files) ...", flush=True)
    Lg, missing = walk_ml(splits, a.dataset_root)
    print(f"    {len(Lg)} of {splits['bf_md5'].nunique()} ML images found under {a.dataset_root}")
    if missing:
        print(f"    WARNING: {missing} ML images not found locally; they are matched by MD5 only (from the split files)")
    if a.legacy_root:
        lg = walk_legacy(a.legacy_root)
        lg["folder"] = "legacy:" + lg["folder"]
        print(f"    + {len(lg)} folders with a bright-field image in the legacy tree")
        Lg = pd.concat([Lg, lg], ignore_index=True)

    print("3/5 MD5 and thumbnails ...", flush=True)
    for i, p in enumerate(Lg["bf"]):
        if i % 200 == 0:
            print(f"    reference images {i}/{len(Lg)}", flush=True)
    Lg["md5"] = [h if isinstance(h, str) else cache.get_md5(p) for h, p in zip(Lg.get("md5", [None] * len(Lg)), Lg["bf"])]
    Lthumb = np.stack([cache.get_thumb(p) for p in Lg["bf"]]) if len(Lg) else np.zeros((0, THUMB * THUMB))
    P["bf_md5"] = [cache.get_md5(p) if p else None for p in P["bf"]]
    Pthumb = np.stack([cache.get_thumb(p) if p else np.zeros(THUMB * THUMB, np.float32) for p in P["bf"]])
    cache.save()

    W = ml_where(splits)
    lg_only = Lg[Lg["folder"].str.startswith("legacy:")]
    legacy_by_md5 = lg_only.groupby("md5")["folder"].apply(lambda v: "; ".join(sorted(v))).to_dict()
    gray_by_md5 = Lg.dropna(subset=["gray"]).groupby("md5")["gray"].first().to_dict()

    print("4/5 matching ...", flush=True)
    recs = []
    for i, r in P.iterrows():
        rec = dict(match="none", near_r=np.nan, ml_roles="", ml_where="", ml_conditions="", legacy_where="",
                   matched_md5=None)
        h = r["bf_md5"]
        if h is None:
            rec["match"] = "no BF"
        elif h in W or h in legacy_by_md5:
            rec["match"], rec["matched_md5"] = "exact", h
        elif len(Lthumb):
            rr = Lthumb @ Pthumb[i] / Pthumb[i].size
            k = int(np.argmax(rr))
            rec["near_r"] = float(rr[k])
            if rr[k] >= a.near:
                rec["match"], rec["matched_md5"] = "near", Lg["md5"].iloc[k]
        mh = rec["matched_md5"]
        if mh:
            if mh in W:
                rec.update(ml_roles=W[mh]["roles"], ml_where=W[mh]["where"], ml_conditions=W[mh]["conditions"])
            rec["legacy_where"] = legacy_by_md5.get(mh, "")
            # traction map: images_cells colour map vs ML / legacy grey map
            g = None
            if mh in W:
                d = os.path.join(a.dataset_root, W[mh]["sample_dir"])
                if os.path.isdir(d):
                    gg = [f for f in os.listdir(d) if f.endswith("_gray.jpg")]
                    g = os.path.join(d, gg[0]) if gg else None
            g = g or gray_by_md5.get(mh)
            rec["gt_map_r"] = corr(jet_to_gray(r["traction"]), gray_map(g)) if (r["traction"] and g) else np.nan
        recs.append(rec)
    P = pd.concat([P, pd.DataFrame(recs, index=P.index)], axis=1)

    def status(x):
        if x["match"] == "no BF":
            return "no BF image"
        if "train" in x["ml_roles"]:
            return "ML training"
        if "validation" in x["ml_roles"]:
            return "ML validation"
        if "test" in x["ml_roles"]:
            return "ML independent test"
        if x["match"] in ("exact", "near"):
            return "legacy tree only"
        return "NEW" if a.legacy_root else "NEW (not in ML dataset)"
    P["status"] = P.apply(status, axis=1)
    P["condition_mismatch_ML"] = P.apply(
        lambda x: bool(x["ml_conditions"]) and x["condition"] not in x["ml_conditions"].split(";"), axis=1)

    # duplicates inside images_cells (same BF in another condition/set)
    key = P["bf_md5"].fillna("")
    R = Pthumb @ Pthumb.T / Pthumb.shape[1]
    dups, dup_other_cond = [], []
    for i in range(len(P)):
        same = set(np.where((key == key.iloc[i]) & (key != ""))[0]) | set(np.where(R[i] >= 0.98)[0])
        same.discard(i)
        locs = sorted(f"{P['condition_folder'].iloc[j]}/{P['set_folder'].iloc[j]}/{P['position'].iloc[j]}" for j in same)
        dups.append("; ".join(locs))
        dup_other_cond.append(bool(any(P["condition"].iloc[j] != P["condition"].iloc[i] for j in same)))
    P["same_image_elsewhere_in_images_cells"] = dups
    P["same_image_other_condition_in_images_cells"] = dup_other_cond

    q = pd.DataFrame([parse_quant(p) for p in P["quant"]], index=P.index)
    P = pd.concat([P, q], axis=1)

    print("5/5 writing results ...", flush=True)
    cols = ["condition_folder", "condition", "set_folder", "position", "pos_num", "b_copy", "status", "match", "near_r",
            "ml_where", "ml_conditions", "condition_mismatch_ML", "legacy_where", "gt_map_r",
            "same_image_elsewhere_in_images_cells", "same_image_other_condition_in_images_cells", "bf_md5", "rel"] + \
           [c for c in q.columns]
    P[cols].to_csv(os.path.join(OUT_DIR, "positions.csv"), index=False)
    S = P.pivot_table(index=["condition_folder", "set_folder"], columns="status", values="position", aggfunc="count",
                      fill_value=0)
    S["positions"] = S.sum(1)
    S["condition_mismatch_ML"] = P.groupby(["condition_folder", "set_folder"])["condition_mismatch_ML"].sum()
    S["same_image_other_condition"] = P.groupby(["condition_folder", "set_folder"])[
        "same_image_other_condition_in_images_cells"].sum()
    S.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    MM = P[P["condition_mismatch_ML"] | P["same_image_other_condition_in_images_cells"]]
    MM[cols].to_csv(os.path.join(OUT_DIR, "condition_mismatch.csv"), index=False)
    NEW = P[P["status"].str.startswith("NEW")]
    NEW[cols].to_csv(os.path.join(OUT_DIR, "new_samples.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    lines = [
        f"images_cells: {a.images_cells}", f"ML dataset  : {a.dataset_root} ({len(Lg) - len(lg_only)} images found, "
        f"{missing} matched by MD5 only)", f"legacy tree : {a.legacy_root or 'not searched'}",
        f"positions: {len(P)}  (unique bright-field images: {P['bf_md5'].nunique()})", "",
        "Status by condition and set:", S.to_string(), "",
        "Status by condition (all sets):",
        P.pivot_table(index="condition_folder", columns="status", values="position", aggfunc="count",
                      fill_value=0).to_string(), "",
        f"Positions whose image is in the ML dataset under ANOTHER condition: {int(P['condition_mismatch_ML'].sum())}",
    ]
    if P["condition_mismatch_ML"].any():
        lines.append(P[P["condition_mismatch_ML"]].groupby(["condition_folder", "set_folder", "ml_conditions"])
                     .size().rename("n").to_string())
    lines += ["", f"Positions whose image also appears under another condition inside images_cells: "
                  f"{int(P['same_image_other_condition_in_images_cells'].sum())}"]
    if P["same_image_other_condition_in_images_cells"].any():
        lines.append(P[P["same_image_other_condition_in_images_cells"]].groupby(["condition_folder", "set_folder"])
                     .size().rename("n").to_string())
    m = P["gt_map_r"].notna()
    if m.any():
        lines += ["", f"Traction map agreement for matched images (images_cells colour map vs ML grey map): "
                      f"median r = {P.loc[m, 'gt_map_r'].median():.3f}, "
                      f"{int((P.loc[m, 'gt_map_r'] < 0.9).sum())} of {int(m.sum())} below 0.9"]
    lines += ["", f"NEW positions (not in the ML dataset{' or the legacy tree' if a.legacy_root else ''}): {len(NEW)}",
              NEW.groupby(["condition_folder", "set_folder"]).size().rename("n").to_string() if len(NEW) else "  none",
              "", f"Near-match threshold r >= {a.near}; the highest r of an unmatched image is "
                  f"{P.loc[P['match'] == 'none', 'near_r'].max():.3f}" if (P["match"] == "none").any() else ""]
    txt = "\n".join(lines)
    open(os.path.join(OUT_DIR, "report.txt"), "w").write(txt)
    print(txt)
    print(f"\nresults: {OUT_DIR}")


if __name__ == "__main__":
    main()
