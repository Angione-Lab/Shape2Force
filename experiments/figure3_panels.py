"""Figure 3 panels B, C, E, F, H for two cohort definitions, using one pipeline.

Both versions use the released checkpoints, the same metrics and the same traction-stress readout.
They differ only in which images are assigned to which condition and replicate:

  mine     : the labels of the ML dataset, images not used for training
             1 kPa  = alltune_val 1kpa_new/test (SET I&II) + test 1kPa_SET_III
             10 kPa = alltune_val 10kpa_new/test (SET I)
             DMSO   = Control_SET_I + DMSO_SET_II; Y-27632 = SET I + SET II; TGF-beta = SET I
             The duplicated Position_08 (DMSO SET II = Y-27632 SET II) is left out of both
             unless --keep-dup is given.
  lautaro  : the condition and set folders of dataset/images_cells (Lautaro's Fig. 3 cohorts),
             matched to the ML images by bright-field MD5 (results/images_cells_audit/positions.csv).

Usage (MODEL = released | image_only):
  python -m experiments.figure3_panels all --model released
  python -m experiments.figure3_panels all --model image_only --stage1 <path>/last_checkpoint_final.pth
  -> results/figure3/per_image_fig3_<MODEL>.csv, figure3_{mine,lautaro}_<MODEL>.png/.pdf,
     figure3_stats_<MODEL>.csv, figure3_cells_{mine,lautaro}_<MODEL>.csv
The spheroid panel H always uses the released spheroid GN model (results/eval/per_image.csv).

Traction-stress readout for C/F: mean traction stress (Pa) inside the force region, where the region is
the union of the regions detected on the GT and the predicted map (same detector as the web app,
S2FApp/utils/segmentation.estimate_cell_mask). The same region is used for GT and prediction.
"""
import argparse
import importlib.util
import os
import sys
import tempfile

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, manifest_path, read_manifest  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "figure3")
POSITIONS = os.path.join(RESULTS_DIR, "images_cells_audit", "positions.csv")
EVAL = os.path.join(RESULTS_DIR, "eval", "per_image.csv")
MANIFESTS = ["alltune_train", "alltune_val", "test_fibro_1kPa_setIII", "test_fibro_DMSO_setI",
             "test_fibro_DMSO_setII", "test_fibro_Y27632_setI", "test_fibro_Y27632_setII", "test_fibro_TGFb_setI"]
DUP = ("singlecell_external/fibro_DMSO_setII/Position_08", "singlecell_external/fibro_Y27632_setII/Position_08")


def _segmentation():
    p = os.path.join(S2F_ROOT, "S2FApp", "utils", "segmentation.py")
    spec = importlib.util.spec_from_file_location("s2fapp_segmentation", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.estimate_cell_mask


def _has_pair(d):
    if not os.path.isdir(d):
        return False
    fs = os.listdir(d)
    return any(f.endswith("BF_001.tif") for f in fs) and any(f.endswith("_gray.jpg") for f in fs)


def infer(dataset_root, device=None, batch=2, model="released", stage1=None, substrate="legacy", manifests=None, tag=None,
          out=None):
    """substrate: 'nominal' = settings of the modulus each set was imaged on; 'legacy' = original codes."""
    import torch
    from experiments.evaluate import CONFIG_PATH, released_models, stage1_model
    from experiments.losses import PerImageWFMRMELoss
    from experiments.manifest_data import make_loader
    from models.s2f_model import create_settings_channels
    from utils.metrics import TorchMetricsWrapper
    from utils.substrate_settings import compute_settings_normalization

    est, rme = _segmentation(), PerImageWFMRMELoss()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model == "image_only":
        if not stage1 or not os.path.exists(stage1):
            raise SystemExit("--stage1 <path to last_checkpoint_final.pth> is needed for --model image_only")
        g, use_settings = stage1_model(stage1, dev)["model"], False
    else:
        g, use_settings = released_models(dev)["released_singlecell"]["model"], True
    mw, norm = TorchMetricsWrapper(device=dev), compute_settings_normalization(config_path=CONFIG_PATH)
    rows, tmpdir = [], tempfile.mkdtemp()
    for man in (manifests or MANIFESTS):
        mrows = read_manifest(manifest_path(man))
        keep = [r for r in mrows if _has_pair(os.path.join(dataset_root, r["sample_dir"]))]
        print(f"{man}: {len(keep)}/{len(mrows)} samples found", flush=True)
        if not keep:
            continue
        tmp = os.path.join(tmpdir, f"{man}.csv")
        pd.DataFrame(keep).to_csv(tmp, index=False)
        loader = make_loader(tmp, dataset_root, batch_size=batch, substrate_mode=substrate if use_settings else "none")
        meta_rows, k = loader.dataset.rows, 0
        with torch.no_grad():
            for b in loader:
                x, y = b[0].to(dev), b[1].to(dev)
                if use_settings:
                    x = torch.cat([x, create_settings_channels(b[4], norm, dev, x.shape, config_path=CONFIG_PATH)], 1)
                p = ((g(x) + 1) / 2).clamp(0, 1)
                for i in range(y.size(0)):
                    r = meta_rows[k]; k += 1
                    t, q = y[i:i + 1], p[i:i + 1]
                    yt = float(r.get("ythreshold_Pa") or 500)
                    tn, qn = t[0, 0].cpu().numpy(), q[0, 0].cpu().numpy()
                    mg, mp = est(tn).astype(bool), est(qn).astype(bool)   # uint8 0/1 -> boolean mask
                    mg = mg if mg.any() else np.ones_like(tn, bool)
                    mp = mp if mp.any() else np.ones_like(qn, bool)
                    mu = mg | mp
                    rows.append({
                        "manifest": man, "sample_dir": r["sample_dir"], "set": r.get("set", ""),
                        "condition": r.get("condition", ""), "bf_md5": r.get("bf_md5", ""),
                        "mse": mw.compute_mse(t, q), "ms_ssim": mw.compute_ms_ssim(t, q), "rme": float(rme(q, t)),
                        "pixel_r": float(np.corrcoef(tn.ravel(), qn.ravel())[0, 1]),
                        "field_mean_gt_Pa": float(tn.mean()) * yt, "field_mean_pred_Pa": float(qn.mean()) * yt,
                        "region_mean_gt_Pa": float(tn[mu].mean()) * yt, "region_mean_pred_Pa": float(qn[mu].mean()) * yt,
                        "own_region_mean_gt_Pa": float(tn[mg].mean()) * yt,
                        "own_region_mean_pred_Pa": float(qn[mp].mean()) * yt,
                        "region_frac": float(mu.mean())})
        print(f"  done {man} ({len(rows)} images so far)", flush=True)
    out = out or OUT
    os.makedirs(out, exist_ok=True)
    f = os.path.join(out, f"per_image_fig3_{tag or model}.csv")
    pd.DataFrame(rows).to_csv(f, index=False)
    print("wrote", f)


# ----------------------------------------------------------------------------------------- cohorts
STIFF = ["1 kPa", "10 kPa"]
DRUGS = ["DMSO", "Y-27632", "TGF-β"]


def cohorts_mine(pi, keep_dup=False):
    c = pi[pi.manifest != "alltune_train"].copy()
    lab = {("alltune_val", "1kpa_new/test"): ("1 kPa", "SET I&II"),
           ("test_fibro_1kPa_setIII", "1kPa_SET_III"): ("1 kPa", "SET III"),
           ("alltune_val", "10kpa_new/test"): ("10 kPa", "SET I"),
           ("test_fibro_DMSO_setI", "Control_SET_I"): ("DMSO", "Control SET I"),
           ("test_fibro_DMSO_setII", "DMSO_SET_II"): ("DMSO", "SET II"),
           ("test_fibro_Y27632_setI", "Y27632_SET_I"): ("Y-27632", "SET I"),
           ("test_fibro_Y27632_setII", "Y27632_SET_II"): ("Y-27632", "SET II"),
           ("test_fibro_TGFb_setI", "TGFB_SET_I"): ("TGF-β", "SET I")}
    c["group"] = [lab.get((m, s), (None, None))[0] for m, s in zip(c.manifest, c.set)]
    c["replicate"] = [lab.get((m, s), (None, None))[1] for m, s in zip(c.manifest, c.set)]
    c = c.dropna(subset=["group"])
    if not keep_dup:
        c = c[~c.sample_dir.isin(DUP)]
    c["origin"] = "genuine"
    return c


LAU_GROUP = {"01kPa": "1 kPa", "10_kPa": "10 kPa", "06kPa_DMSO": "DMSO", "06kPa_Y27632": "Y-27632",
             "06kPa_TGF-b": "TGF-β"}


def _lau_rep(s):
    s = s.replace("Sum_Force_Values_", "").replace("SUM_FORCE_", "").replace("Sum_Force_", "")
    return s.replace("_", " ").replace("SET ", "SET ")


def cohorts_lautaro(pi):
    P = pd.read_csv(POSITIONS)
    P = P[P.condition_folder.isin(LAU_GROUP)].copy()
    P["group"] = P.condition_folder.map(LAU_GROUP)
    P["replicate"] = P.set_folder.map(_lau_rep)
    one = pi.drop_duplicates("bf_md5")
    m = P.merge(one, on="bf_md5", how="left", suffixes=("_lau", ""))
    m["origin"] = np.where(m.condition_mismatch_ML.astype(str) == "True", "relabelled 6 kPa", "genuine")
    return m


# ------------------------------------------------------------------------------------------- stats
def _stats(c, version, groups, ref):
    from scipy import stats
    out = []
    for gname in groups:
        d = c[c.group == gname].dropna(subset=["region_mean_gt_Pa"])
        if len(d) > 1:
            w = stats.wilcoxon(d.region_mean_gt_Pa, d.region_mean_pred_Pa)
            out.append(dict(version=version, test="GT vs P (Wilcoxon, cells)", group=gname, n=len(d),
                            N=d.replicate.nunique(), gt_mean=d.region_mean_gt_Pa.mean(),
                            pred_mean=d.region_mean_pred_Pa.mean(), p=w.pvalue))
    for gname in groups:
        if gname == ref:
            continue
        a, b = c[c.group == ref], c[c.group == gname]
        for col, lab in [("region_mean_gt_Pa", "GT"), ("region_mean_pred_Pa", "P")]:
            u = stats.mannwhitneyu(a[col].dropna(), b[col].dropna())
            out.append(dict(version=version, test=f"{gname} vs {ref}, {lab} (Mann-Whitney, cells)", group=gname,
                            n=len(b), N=b.replicate.nunique(), gt_mean=a[col].mean(), pred_mean=b[col].mean(),
                            p=u.pvalue))
        # replicate-level t-test when both groups have >= 2 replicates
        ra = a.groupby("replicate")["region_mean_gt_Pa"].mean()
        rb = b.groupby("replicate")["region_mean_gt_Pa"].mean()
        if len(ra) >= 2 and len(rb) >= 2:
            t = stats.ttest_ind(ra, rb)
            out.append(dict(version=version, test=f"{gname} vs {ref}, GT (t-test, replicate means)", group=gname,
                            n=len(b), N=len(rb), gt_mean=ra.mean(), pred_mean=rb.mean(), p=t.pvalue))
    return out


def _stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ------------------------------------------------------------------------------------------- plots
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
                         "savefig.dpi": 300})


COL = {"1 kPa": "#6baed6", "10 kPa": "#08519c", "DMSO": "#969696", "Y-27632": "#41ab5d", "TGF-β": "#d94801",
       "spheroid": "#756bb1"}
REL = "#e7298a"


def _metric_bars(axs, c, groups, show_origin):
    rng = np.random.default_rng(0)
    for ax, (col, lab) in zip(axs, [("mse", "MSE"), ("pixel_r", "Pixel correlation (r)"), ("ms_ssim", "MS-SSIM")]):
        for i, gname in enumerate(groups):
            d = c[c.group == gname].dropna(subset=[col])
            ax.bar(i, d[col].mean(), yerr=d[col].std(), width=0.65, color=COL[gname], alpha=0.35,
                   edgecolor=COL[gname], linewidth=0.8, error_kw=dict(lw=0.7, capsize=2))
            jit = rng.uniform(-0.2, 0.2, len(d))
            rel = (d.origin != "genuine").values if show_origin else np.zeros(len(d), bool)
            ax.scatter(i + jit[~rel], d[col].values[~rel], s=3, color=COL[gname], lw=0, zorder=3)
            if rel.any():
                ax.scatter(i + jit[rel], d[col].values[rel], s=5, facecolor="none", edgecolor=REL, lw=0.5, zorder=3)
        ax.set_xticks(range(len(groups)), groups, rotation=35 if len(groups) > 2 else 0,
                      ha="right" if len(groups) > 2 else "center", rotation_mode="anchor")
        ax.set_ylabel(lab)
        ax.margins(x=0.08)
        if col != "mse":
            ax.set_ylim(0, 1.05)


def _superplot(ax, c, groups, st, ref):
    rng = np.random.default_rng(1)
    xt, xl = [], []
    for i, gname in enumerate(groups):
        d = c[c.group == gname]
        reps = sorted(d.replicate.unique())
        mk = ["o", "s", "^", "D"]
        for j, (col, lab) in enumerate([("region_mean_gt_Pa", "GT"), ("region_mean_pred_Pa", "P")]):
            x = i * 2.6 + j
            v = d[col].dropna().values
            if len(v) > 2:
                vp = ax.violinplot(v, [x], widths=0.8, showextrema=False)
                for b in vp["bodies"]:
                    b.set_facecolor(COL[gname]); b.set_alpha(0.18 if j == 0 else 0.10); b.set_edgecolor("none")
            rel = (d.origin != "genuine").values
            jit = rng.uniform(-0.18, 0.18, len(d))
            ax.scatter(x + jit[~rel], d[col].values[~rel], s=3, color=COL[gname], lw=0, alpha=0.8, zorder=3)
            if rel.any():
                ax.scatter(x + jit[rel], d[col].values[rel], s=5, facecolor="none", edgecolor=REL, lw=0.5, zorder=3)
            for k, rp in enumerate(reps):
                rm = d[d.replicate == rp][col].mean()
                relrep = (d[d.replicate == rp].origin != "genuine").all()
                ax.scatter(x, rm, s=26, marker=mk[k % 4], facecolor=REL if relrep else COL[gname],
                           edgecolor="k", lw=0.5, zorder=4)
            xt.append(x); xl.append(lab)
        w = [s for s in st if s["group"] == gname and s["test"].startswith("GT vs P")]
        if w:
            y = np.nanmax(d[["region_mean_gt_Pa", "region_mean_pred_Pa"]].values) * 1.04
            ax.plot([i * 2.6, i * 2.6 + 1], [y, y], color="k", lw=0.5)
            ax.text(i * 2.6 + 0.5, y, _stars(w[0]["p"]), ha="center", va="bottom", fontsize=6)
        ax.text(i * 2.6 + 0.5, -0.17, f"{gname}\nn={len(d)}, N={len(reps)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6)
    ax.set_xticks(xt, xl)
    ax.set_ylabel("Mean traction stress in force region (Pa)")
    ax.set_ylim(bottom=0)
    ymax = ax.get_ylim()[1]
    k = 0
    for i, gname in enumerate(groups):
        if gname == ref:
            continue
        for lab, off in [("GT", 0)]:
            s = [x for x in st if x["group"] == gname and x["test"].startswith(f"{gname} vs {ref}, {lab} (Mann")]
            if not s:
                continue
            i0 = groups.index(ref)
            y = ymax * (1.02 + 0.07 * k)
            ax.plot([i0 * 2.6 + off, i * 2.6 + off], [y, y], color="0.3", lw=0.5)
            ax.text((i0 * 2.6 + i * 2.6) / 2 + off, y, f"GT {_stars(s[0]['p'])}", ha="center", va="bottom", fontsize=6)
            k += 1
    ax.set_ylim(0, ymax * (1.05 + 0.07 * k))


def plot(keep_dup=False, model="released"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style()
    pi = pd.read_csv(os.path.join(OUT, f"per_image_fig3_{model}.csv"))
    ev = pd.read_csv(EVAL)
    sph = ev[(ev.model == "released_spheroid_GN") & (ev.manifest == "spheroid_val")].assign(group="spheroid",
                                                                                            origin="genuine")
    allst = []
    for version, c in [("mine", cohorts_mine(pi, keep_dup)), ("lautaro", cohorts_lautaro(pi))]:
        miss = c.region_mean_gt_Pa.isna().sum()
        c.to_csv(os.path.join(OUT, f"figure3_cells_{version}_{model}.csv"), index=False)
        st = _stats(c[c.group.isin(STIFF)], version, STIFF, "1 kPa") + _stats(c[c.group.isin(DRUGS)], version, DRUGS,
                                                                             "DMSO")
        allst += st
        fig = plt.figure(figsize=(7.2, 7.6))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.75, wspace=0.28,
                              left=0.08, right=0.98, top=0.93, bottom=0.08)
        rows = [("B", "C", STIFF, "1 kPa"), ("E", "F", DRUGS, "DMSO")]
        for r, (lb, lc, groups, ref) in enumerate(rows):
            sub = gs[r, 0].subgridspec(1, 3, wspace=0.55)
            axs = [fig.add_subplot(sub[0, k]) for k in range(3)]
            _metric_bars(axs, c[c.group.isin(groups)], groups, version == "lautaro")
            axs[0].text(-0.55, 1.12, lb, transform=axs[0].transAxes, fontsize=9, fontweight="bold")
            axc = fig.add_subplot(gs[r, 1])
            _superplot(axc, c[c.group.isin(groups)], groups, st, ref)
            axc.text(-0.3, 1.12, lc, transform=axc.transAxes, fontsize=9, fontweight="bold")
        sub = gs[2, 0].subgridspec(1, 3, wspace=0.55)
        axs = [fig.add_subplot(sub[0, k]) for k in range(3)]
        _metric_bars(axs, sph, ["spheroid"], False)
        for a in axs:
            a.set_xticks([0], [f"LS174T\nn={len(sph)}"])
        axs[0].text(-0.55, 1.12, "H", transform=axs[0].transAxes, fontsize=9, fontweight="bold")
        axl = fig.add_subplot(gs[2, 1]); axl.axis("off")
        hd = [Line2D([], [], marker="o", ls="", ms=6, mfc="0.6", mec="k", mew=0.5, label="replicate (set) mean"),
              Line2D([], [], marker="o", ls="", ms=2.5, mfc="0.4", mec="none", label="cell")]
        if version == "lautaro":
            hd += [Line2D([], [], marker="o", ls="", ms=3, mfc="none", mec=REL, label="6 kPa untreated cell in this group"),
                   Line2D([], [], marker="o", ls="", ms=6, mfc=REL, mec="k", mew=0.5,
                          label="replicate made of 6 kPa untreated cells")]
        axl.legend(handles=hd, loc="upper left", frameon=False)
        title = {"mine": "Fig. 3 with the ML-dataset cohorts (genuine cells, not used for training)",
                 "lautaro": "Fig. 3 with the images_cells cohorts (as in the revised figure)"}[version]
        mname = {"released": "released single-cell model (BF + substrate)", "image_only": "image-only model (BF)"}[model]
        fig.suptitle(f"{title}; {mname}", fontsize=8, x=0.02, ha="left")
        note = ("GT vs P: Wilcoxon signed-rank on cells; between conditions: Mann-Whitney on cells (GT). "
                "Bars: mean ± SD. Spheroids (H): released GN model, 28 validation spheroids, identical in both versions.")
        if miss:
            note += f" {miss} cells without a prediction (images not found)."
        import textwrap
        fig.text(0.02, 0.012, "\n".join(textwrap.wrap(note, 190)), fontsize=5.5, color="0.3", va="bottom")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"figure3_{version}_{model}.{ext}"))
        plt.close(fig)
    pd.DataFrame(allst).to_csv(os.path.join(OUT, f"figure3_stats_{model}.csv"), index=False)
    print("wrote figures to", OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["infer", "plot", "all"])
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", choices=["released", "image_only"], default="released",
                    help="released single-cell checkpoint (ckp/single_cell/ckp_singlecell_GN.pth) or the image-only model")
    ap.add_argument("--stage1", default=None, help="image-only checkpoint (last_checkpoint_final.pth)")
    ap.add_argument("--keep-dup", action="store_true", help="keep Position_08 in both DMSO SET II and Y-27632 SET II")
    a = ap.parse_args()
    if a.step in ("infer", "all"):
        infer(a.dataset_root, a.device, model=a.model, stage1=a.stage1)
    if a.step in ("plot", "all"):
        plot(a.keep_dup, a.model)


if __name__ == "__main__":
    main()
