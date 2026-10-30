"""Traction-stress readouts on 1, 6 and 10 kPa held-out images, 6 kPa reference vs true substrate setting.

Question: is the low predicted magnitude at 1 kPa (true setting) real, or an artefact of averaging over the whole
field (e.g. the prediction is spread over a larger area at a lower level)?

For each image and model/setting it records:
  field_*     mean over the whole image
  gtreg_*     mean inside the force region detected on the GT map (web-app detector), same pixels for GT and P
  own_*       mean inside each map's own detected force region (GT in its region, P in its region)
  p99_*       99th percentile (peak stress)
  area_*      fraction of the image covered by the detected force region (spread)
  a50_*       fraction of pixels above 50 Pa
  sum_*       total stress over the image (sum of Pa over pixels), i.e. field mean x n_pixels
and writes one PNG per image (BF | GT | released ref | retrained ref | retrained true) on a common colour scale per row.

  python -m experiments.stiffness_readouts \
      [--runs "experiments/runs/s2f_staged_seed*/best_checkpoint.pth"] [--out experiments/results/stiffness_readouts]
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

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, RUNS_DIR, manifest_path, read_manifest  # noqa: E402

GROUPS = {  # manifest, set filter -> group label
    ("test_fibro_1kPa_setIII", None): "1 kPa test (SET III)",
    ("alltune_val", "1kpa_new/test"): "1 kPa val",
    ("alltune_val", "fibro_split/test"): "6 kPa val (SET VI)",
    ("alltune_val", "10kpa_new/test"): "10 kPa val",
}
REFERENCE = "fibroblasts_PDMS"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(RUNS_DIR, "s2f_staged_seed*", "best_checkpoint.pth"))
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "stiffness_readouts"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max images per group (smoke test)")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from experiments.evaluate import CONFIG_PATH, released_models, run_models
    from experiments.figure3_panels import _has_pair, _segmentation
    from experiments.manifest_data import make_loader
    from models.s2f_model import create_settings_channels
    from utils.substrate_settings import compute_settings_normalization

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {"released": released_models(dev)["released_singlecell"]["model"]}
    runs = run_models(dev, a.runs)
    models.update({k: v["model"] for k, v in runs.items()})
    print("models:", list(models), flush=True)
    norm, est = compute_settings_normalization(config_path=CONFIG_PATH), _segmentation()
    os.makedirs(a.out, exist_ok=True)
    rows, tmpdir = [], tempfile.mkdtemp()

    def readouts(tn, qn, yt):
        mg = est(tn).astype(bool); mg = mg if mg.any() else np.ones_like(tn, bool)
        mp = est(qn).astype(bool); mp = mp if mp.any() else np.ones_like(qn, bool)
        T, Q = tn * yt, qn * yt
        return dict(field_gt=T.mean(), field_pred=Q.mean(), gtreg_gt=T[mg].mean(), gtreg_pred=Q[mg].mean(),
                    own_gt=T[mg].mean(), own_pred=Q[mp].mean(), p99_gt=np.percentile(T, 99), p99_pred=np.percentile(Q, 99),
                    area_gt=mg.mean(), area_pred=mp.mean(), a50_gt=(T > 50).mean(), a50_pred=(Q > 50).mean(),
                    sum_gt=T.sum(), sum_pred=Q.sum(), pixel_r=float(np.corrcoef(tn.ravel(), qn.ravel())[0, 1]))

    for (man, setf), label in GROUPS.items():
        mrows = read_manifest(manifest_path(man))
        keep = [r for r in mrows if (setf is None or r.get("set") == setf) and _has_pair(os.path.join(a.dataset_root, r["sample_dir"]))]
        if a.limit:
            keep = keep[:a.limit]
        print(f"{label}: {len(keep)} images", flush=True)
        if not keep:
            continue
        tmp = os.path.join(tmpdir, "m.csv"); pd.DataFrame(keep).to_csv(tmp, index=False)
        loader = make_loader(tmp, a.dataset_root, batch_size=2, substrate_mode="nominal")
        meta_rows, k = loader.dataset.rows, 0
        pdir = os.path.join(a.out, "panels", label.replace(" ", "_").replace("(", "").replace(")", ""))
        os.makedirs(pdir, exist_ok=True)
        with torch.no_grad():
            for b in loader:
                x, y = b[0].to(dev), b[1].to(dev)
                meta_true = b[4]
                meta_ref = dict(meta_true, substrate=[REFERENCE] * x.size(0))
                preds = {}
                for nm, g in models.items():
                    for setting, meta in (("reference", meta_ref), ("true", meta_true)):
                        xs = torch.cat([x, create_settings_channels(meta, norm, dev, x.shape, config_path=CONFIG_PATH)], 1)
                        preds[(nm, setting)] = ((g(xs) + 1) / 2).clamp(0, 1).cpu().numpy()
                for i in range(y.size(0)):
                    r = meta_rows[k]; k += 1
                    yt = float(r.get("ythreshold_Pa") or 500)
                    tn = y[i, 0].cpu().numpy()
                    maps = {}
                    for (nm, setting), p in preds.items():
                        qn = p[i, 0]; maps[(nm, setting)] = qn
                        rows.append(dict(group=label, sample_dir=r["sample_dir"], model=nm, setting=setting, **readouts(tn, qn, yt)))
                    if a.no_png:
                        continue
                    seed0 = sorted(k_ for k_ in models if k_ != "released")[:1]
                    show = [("released", "reference")] + [(s, st) for s in seed0 for st in ("reference", "true")]
                    vmax = max(np.percentile(tn * yt, 99.5), *[np.percentile(maps[s] * yt, 99.5) for s in show], 1.0)
                    fig, axs = plt.subplots(1, 2 + len(show), figsize=(3 * (2 + len(show)), 3.4), gridspec_kw=dict(wspace=0.04))
                    axs[0].imshow(x[i, 0].cpu().numpy(), cmap="gray"); axs[0].set_title("Bright-field", fontsize=8)
                    rd = readouts(tn, tn, yt)
                    im = axs[1].imshow(tn * yt, cmap="jet", vmin=0, vmax=vmax)
                    axs[1].set_title(f"GT\nfield {rd['field_gt']:.0f} | region {rd['gtreg_gt']:.0f} Pa | area {rd['area_gt']:.2f}", fontsize=7)
                    for ax, s in zip(axs[2:], show):
                        d = readouts(tn, maps[s], yt)
                        ax.imshow(maps[s] * yt, cmap="jet", vmin=0, vmax=vmax)
                        ax.set_title(f"{s[0]} ({s[1]})\nfield {d['field_pred']:.0f} | GT-region {d['gtreg_pred']:.0f} | own {d['own_pred']:.0f} Pa"
                                     f"\narea {d['area_pred']:.2f} | r {d['pixel_r']:.2f}", fontsize=7)
                    for ax in axs:
                        ax.set_xticks([]); ax.set_yticks([])
                    cb = fig.colorbar(im, ax=axs[1:], fraction=0.015, pad=0.01); cb.set_label("Traction stress (Pa)", fontsize=7)
                    fig.suptitle(f"{label} | {r['sample_dir']} | true setting {r.get('substrate_nominal', '')} (colour scale 0-{vmax:.0f} Pa)",
                                 fontsize=8, x=0.01, ha="left")
                    fig.savefig(os.path.join(pdir, os.path.basename(r["sample_dir"]) + ".png"), dpi=100, bbox_inches="tight")
                    plt.close(fig)
        print(f"  done {label}", flush=True)

    df = pd.DataFrame(rows)
    # average the two seeds of each retrained run family per image
    df["family"] = df.model.str.replace(r"_seed\d+$", "", regex=True)
    num = [c for c in df.columns if c not in ("group", "sample_dir", "model", "setting", "family")]
    per = df.groupby(["group", "sample_dir", "family", "setting"])[num].mean().reset_index()
    per.to_csv(os.path.join(a.out, "per_image_readouts.csv"), index=False)
    out = []
    for (g, fam, st), d in per.groupby(["group", "family", "setting"]):
        o = dict(group=g, model=fam, setting=st, n=len(d), pixel_r=d.pixel_r.mean())
        for ro in ("field", "gtreg", "own", "p99", "a50", "area"):
            o[f"GT_{ro}"] = d[f"{ro}_gt"].mean(); o[f"P_{ro}"] = d[f"{ro}_pred"].mean()
            o[f"P/GT_{ro}"] = d[f"{ro}_pred"].mean() / max(d[f"{ro}_gt"].mean(), 1e-12)
        out.append(o)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, "summary_readouts.csv"), index=False)
    cols = ["group", "model", "setting", "n", "pixel_r"] + [f"P/GT_{ro}" for ro in ("field", "gtreg", "own", "p99", "area")]
    print(s[cols].round(3).to_string(index=False))
    # 10 kPa vs 1 kPa fold change of each readout (1 kPa = test SET III + validation), GT and P
    per["kpa"] = per.group.map(lambda g: "1" if g.startswith("1 kPa") else ("10" if g.startswith("10 kPa") else "6"))
    fold = []
    for (fam, st), d in per.groupby(["family", "setting"]):
        o = dict(model=fam, setting=st)
        for ro in ("field", "gtreg", "own", "p99"):
            for w in ("gt", "pred"):
                lo, hi = d.loc[d.kpa == "1", f"{ro}_{w}"], d.loc[d.kpa == "10", f"{ro}_{w}"]
                o[f"{'GT' if w == 'gt' else 'P'}_fold_{ro}"] = hi.mean() / max(lo.mean(), 1e-12) if len(lo) and len(hi) else np.nan
        fold.append(o)
    f = pd.DataFrame(fold)
    f.to_csv(os.path.join(a.out, "fold_10_vs_1kPa.csv"), index=False)
    print(f.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
