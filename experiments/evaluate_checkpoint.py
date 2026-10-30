"""Evaluate any S2F generator checkpoint on the validation set and the 145 independent test images, and compare it with
the released model (from results/eval/per_image.csv) using paired Wilcoxon tests.

  python -m experiments.evaluate_checkpoint --ckpt "C:/.../fine_tune_settings_mean/rme/best_checkpoint.pth" --name rme_stage2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR  # noqa: E402

TEST = ["test_fibro_1kPa_setIII", "test_fibro_DMSO_setI", "test_fibro_DMSO_setII", "test_fibro_Y27632_setI",
        "test_fibro_Y27632_setII", "test_fibro_TGFb_setI", "test_fibro_fibronectin_setI"]


def pool(df, split):
    if split == "val":
        d = df[df.manifest == "alltune_val"]
    else:
        d = df[df.manifest.isin(TEST)]
        d = d[~(d.sample_dir.str.contains("Y27632_setII") & d.sample_dir.str.contains("Position_08"))]
    return d.drop_duplicates(["manifest", "sample_dir"]).set_index(["manifest", "sample_dir"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    from scipy.stats import wilcoxon
    from experiments.evaluate import CONFIG_PATH, evaluate_one
    from experiments.match_checkpoint import load_generator
    from utils.metrics import TorchMetricsWrapper
    from utils.substrate_settings import compute_settings_normalization

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    g, in_ch, ep = load_generator(a.ckpt, dev)
    spec = dict(model=g, kind="singlecell", in_channels=in_ch, tanh=True)
    mw, norm = TorchMetricsWrapper(device=dev), compute_settings_normalization(config_path=CONFIG_PATH)
    rows = []
    for man in TEST + ["alltune_val"]:
        rows += evaluate_one(a.name, spec, man, a.dataset_root, dev, mw, norm,
                             "force:fibroblasts_PDMS" if in_ch == 3 else "none", 2)
    new = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"eval_{a.name}.csv"); new.to_csv(out, index=False)
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "eval", "per_image.csv"))
    ref = ref[(ref.model == "released_singlecell") & (ref.substrate_mode == "force:fibroblasts_PDMS")]
    print(f"{a.name} (epoch {ep}, {in_ch} input channels) vs released; paired Wilcoxon over images")
    for split in ("val", "test"):
        A, B = pool(new, split), pool(ref, split)
        idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
        print(f"\n{split} (n = {len(idx)})")
        for m in ("pixel_r", "rme", "mse", "ms_ssim"):
            print(f"  {m:8s} {a.name} {A[m].mean():.4f}  released {B[m].mean():.4f}  p = {wilcoxon(A[m], B[m]).pvalue:.2g}")
        pa, pb = A.mean_pred.sum() / A.mean_gt.sum(), B.mean_pred.sum() / B.mean_gt.sum()
        print(f"  P/GT     {a.name} {pa:.3f}  released {pb:.3f}  "
              f"p = {wilcoxon(A.mean_pred / A.mean_gt, B.mean_pred / B.mean_gt).pvalue:.2g}")
    print(f"\nper-image results: {out}")


if __name__ == "__main__":
    main()
