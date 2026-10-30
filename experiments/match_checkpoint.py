"""Find which checkpoint produced the old predictions shared with Lautaro.

Runs every candidate checkpoint on the images listed in experiments/results/old_prediction_targets.csv (pixel r, MSE and
MS-SSIM copied from Lautaro's sheets, which come from the old shared predictions) and reports how closely each candidate
reproduces them. 1 kPa SET III is a positive control: its old values match the released model.

  python -m experiments.match_checkpoint --ckpts "D:/path/a.pth,D:/path/**/*.pth"
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR  # noqa: E402


def load_generator(path, dev):
    import torch
    from experiments.ablation_models import build_generator
    from models.s2f_model import create_s2f_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "config" in ck and "generator_state_dict" in ck and "in_channels" in ck["config"]:
        g = build_generator(**ck["config"]); g.load_state_dict(ck["generator_state_dict"], strict=True)
        return g.to(dev).eval(), ck["config"]["in_channels"], ck.get("epoch")
    sd = None
    for k in ("generator_state_dict", "model_state_dict", "state_dict", "generator"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            sd = ck[k]; break
    sd = sd or ck
    sd = {k.replace("module.", "").replace("denoise_module", "initial_conv"): v for k, v in sd.items()}
    convs = [v for k, v in sd.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 4]
    in_ch = int(convs[0].shape[1]) if convs else 1
    g, _ = create_s2f_model(in_channels=in_ch, model_type="s2f")
    g.load_state_dict(sd, strict=True)
    return g.to(dev).eval(), in_ch, ck.get("epoch") if isinstance(ck, dict) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", required=True, help="comma-separated paths or globs (** allowed)")
    ap.add_argument("--targets", default=os.path.join(RESULTS_DIR, "old_prediction_targets.csv"))
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "match_checkpoint.csv"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    from experiments.evaluate import CONFIG_PATH
    from experiments.manifest_data import make_loader
    from models.s2f_model import create_settings_channels
    from utils.metrics import TorchMetricsWrapper
    from utils.substrate_settings import compute_settings_normalization

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = sorted({p for pat in a.ckpts.split(",") if pat for p in glob.glob(pat.strip(), recursive=True)})
    print(f"{len(paths)} candidate checkpoints", flush=True)
    T = pd.read_csv(a.targets)
    mw, norm = TorchMetricsWrapper(device=dev), compute_settings_normalization(config_path=CONFIG_PATH)
    tmpdir, rows = tempfile.mkdtemp(), []
    loaders = {}
    for man, d in T.groupby("manifest"):
        tmp = os.path.join(tmpdir, f"{man}.csv")
        from experiments.layout import manifest_path, read_manifest
        from experiments.figure3_panels import _has_pair
        keep = [r for r in read_manifest(manifest_path(man)) if r["sample_dir"] in set(d.sample_dir)
                and _has_pair(os.path.join(a.dataset_root, r["sample_dir"]))]
        print(f"{man}: {len(keep)}/{len(d)} target images found", flush=True)
        if not keep:
            continue
        pd.DataFrame(keep).to_csv(tmp, index=False)
        loaders[man] = tmp
    for p in paths:
        try:
            g, in_ch, ep = load_generator(p, dev)
        except Exception as e:  # not an S2F generator checkpoint
            print(f"  skip {p}: {type(e).__name__}: {str(e)[:120]}", flush=True); continue
        settings = ["fibroblasts_PDMS", "nominal"] if in_ch == 3 else ["none"]
        for st in settings:
            res = []
            for man, tmp in loaders.items():
                loader = make_loader(tmp, a.dataset_root, batch_size=2, substrate_mode="nominal" if in_ch == 3 else "none")
                meta_rows, k = loader.dataset.rows, 0
                with torch.no_grad():
                    for b in loader:
                        x, y = b[0].to(dev), b[1].to(dev)
                        if in_ch == 3:
                            meta = b[4] if st == "nominal" else dict(b[4], substrate=[st] * x.size(0))
                            x = torch.cat([x, create_settings_channels(meta, norm, dev, x.shape, config_path=CONFIG_PATH)], 1)
                        q = ((g(x) + 1) / 2).clamp(0, 1)
                        for i in range(y.size(0)):
                            r = meta_rows[k]; k += 1
                            t, qi = y[i:i + 1], q[i:i + 1]
                            res.append(dict(sample_dir=r["sample_dir"], r=float(np.corrcoef(t.flatten().cpu(), qi.flatten().cpu())[0, 1]),
                                            mse=mw.compute_mse(t, qi), msssim=mw.compute_ms_ssim(t, qi)))
            m = T.merge(pd.DataFrame(res), on="sample_dir")
            for grp, d in m.groupby("group"):
                match = ((d.r - d.old_r).abs() < 1e-3) & ((d.mse - d.old_mse).abs() < 1e-4)
                rows.append(dict(checkpoint=p, epoch=ep, in_channels=in_ch, setting=st, group=grp, n=len(d),
                                 mean_abs_dr=(d.r - d.old_r).abs().mean(), mean_abs_dmse=(d.mse - d.old_mse).abs().mean(),
                                 n_exact=int(match.sum())))
            print(f"  {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} [{st}]: " +
                  ", ".join(f"{r['group']} {r['n_exact']}/{r['n']} (|dr| {r['mean_abs_dr']:.3f})" for r in rows[-len(m.group.unique()):]), flush=True)
    out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
    if len(out):
        best = out.groupby(["checkpoint", "setting"]).agg(n_exact=("n_exact", "sum"), mean_abs_dr=("mean_abs_dr", "mean")).sort_values(["n_exact", "mean_abs_dr"], ascending=[False, True])
        print(best.head(10).to_string())


if __name__ == "__main__":
    main()
