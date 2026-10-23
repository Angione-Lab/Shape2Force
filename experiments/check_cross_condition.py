"""
Are any images of the ML dataset the same cell under two different conditions?

    python -m experiments.check_cross_condition

Every bright-field image listed in experiments/splits/*.csv (read from the dataset root) is reduced to a
128 x 128 thumbnail and compared with all images of OTHER conditions (Pearson r; identical or re-saved images
give r > 0.99, different cells r < 0.9). Pairs with r >= --near are listed.

Output: experiments/results/dataset_audit/cross_condition_pairs.csv and, per set, the highest similarity to
any image of another condition (cross_condition_max.csv).
"""
import argparse
import os

import numpy as np
import pandas as pd

from experiments.check_overlap import load_splits, thumb
from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR

OUT = os.path.join(RESULTS_DIR, "dataset_audit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--near", type=float, default=0.95)
    a = ap.parse_args()
    s = load_splits()
    s = s[s["cell_type"] != "LS174T spheroid"].drop_duplicates("sample_dir")
    rows, th = [], []
    for _, r in s.iterrows():
        d = os.path.join(a.dataset_root, r["sample_dir"])
        bf = next((os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith("BF_001.tif")), None) \
            if os.path.isdir(d) else None
        t = thumb(bf) if bf else None
        if t is None:
            continue
        rows.append(r)
        th.append(t)
    s = pd.DataFrame(rows).reset_index(drop=True)
    T = np.stack(th)
    print(f"{len(s)} single-cell images read")
    R = T @ T.T / T.shape[1]
    cond = s["condition"].values
    other = cond[:, None] != cond[None, :]
    Rm = np.where(other, R, -1)
    s["max_r_other_condition"] = Rm.max(1)
    s["closest_other"] = [s["sample_dir"].iloc[j] + " [" + cond[j] + "]" for j in Rm.argmax(1)]
    os.makedirs(OUT, exist_ok=True)
    pairs = [(s["sample_dir"].iloc[i], cond[i], s["sample_dir"].iloc[j], cond[j], float(R[i, j]))
             for i, j in zip(*np.where(np.triu(Rm >= a.near, 1)))]
    P = pd.DataFrame(pairs, columns=["sample_a", "condition_a", "sample_b", "condition_b", "r"])
    P.to_csv(os.path.join(OUT, "cross_condition_pairs.csv"), index=False)
    M = s.groupby(["split_file", "set", "condition"])["max_r_other_condition"].agg(["count", "max"]).reset_index()
    M.to_csv(os.path.join(OUT, "cross_condition_max.csv"), index=False)
    pd.set_option("display.width", 200)
    print(M.to_string(index=False))
    print(f"\npairs of the same image under two conditions (r >= {a.near}): {len(P)}")
    if len(P):
        print(P.to_string(index=False))


if __name__ == "__main__":
    main()
