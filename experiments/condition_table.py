"""Per-condition table (counts and GT traction per source) + box plots, ML dataset vs Lautaro's cohorts.

    python experiments/condition_table.py [--out experiments/results/condition_table]
needs: experiments/splits/*.csv, experiments/results/predictions/predictions_metrics.csv,
       experiments/results/images_cells_audit/positions.csv, Archive/OneDrive_1_20-10-2026 (Prism files)
writes: condition_table.xlsx (+ .csv), condition_boxplots.png/.pdf
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONEDRIVE = os.path.join(ROOT, "Archive", "OneDrive_1_20-10-2026")
CONDS = ["1 kPa", "6 kPa collagen, untreated", "10 kPa", "6 kPa fibronectin", "6 kPa DMSO", "6 kPa Y-27632", "6 kPa TGF-β"]
SHORT = {"1 kPa": "1 kPa", "6 kPa collagen, untreated": "6 kPa\nuntreated", "10 kPa": "10 kPa",
         "6 kPa fibronectin": "Fibronectin", "6 kPa DMSO": "DMSO", "6 kPa Y-27632": "Y-27632", "6 kPa TGF-β": "TGF-β"}
MANUSCRIPT_N = {"1 kPa": 50, "10 kPa": 87, "6 kPa collagen, untreated": 153, "6 kPa fibronectin": 40,
                "6 kPa DMSO": 30, "6 kPa Y-27632": 54, "6 kPa TGF-β": 82}
DUP = {"singlecell_external/fibro_DMSO_setII/Position_08", "singlecell_external/fibro_Y27632_setII/Position_08"}
COMPARISONS = [("1 kPa", "10 kPa"), ("6 kPa collagen, untreated", "6 kPa fibronectin"),
               ("6 kPa DMSO", "6 kPa Y-27632"), ("6 kPa DMSO", "6 kPa TGF-β")]


def ml_condition(manifest, set_, condition):
    if condition.startswith("1 kPa"):
        return "1 kPa"
    if condition.startswith("10 kPa"):
        return "10 kPa"
    return {"6 kPa collagen": "6 kPa collagen, untreated", "fibronectin": "6 kPa fibronectin", "DMSO": "6 kPa DMSO",
            "Y-27632": "6 kPa Y-27632", "TGF-beta": "6 kPa TGF-β"}.get(condition)


def read_pzfx(path):
    ns = lambda t: t.split("}")[-1]
    out = {}
    for tab in ET.parse(path).getroot().iter():
        if ns(tab.tag) not in ("Table", "HugeTable"):
            continue
        t = [c for c in tab if ns(c.tag) == "Title"]
        cols = []
        for col in tab:
            if ns(col.tag) != "YColumn":
                continue
            ct = [c for c in col if ns(c.tag) == "Title"]
            for sub in col:
                if ns(sub.tag) == "Subcolumn":
                    vals = []
                    for d in sub:
                        try:
                            vals.append(float((d.text or "").strip()))
                        except ValueError:
                            pass
                    cols.append(("".join(ct[0].itertext()) if ct else "", vals))
        out["".join(t[0].itertext()) if t else tab.get("ID")] = cols
    return out


def lautaro_figure_values():
    """GT values exactly as plotted in Lautaro's REVISION Prism files (Figs 2 and 3)."""
    vals = {c: [] for c in CONDS}
    st = read_pzfx(os.path.join(ONEDRIVE, "Stiffness_extra graphs", "Superplots_Fibros_Figure_Stiffness_GTvsAI_REVISION.pzfx"))["Nested"]
    for t, v in st:
        if t.startswith("1 kPa GT"):
            vals["1 kPa"] += v
        elif t.startswith("10 kPa GT"):
            vals["10 kPa"] += v
    dr = read_pzfx(os.path.join(ONEDRIVE, "Drugs", "Superplots_Fibros_Figure_Drugs_GT-ONLY_REVISION.pzfx"))["Nested"]
    for t, v in dr:
        vals[{"DMSO": "6 kPa DMSO", "Y27632": "6 kPa Y-27632", "TGF-B": "6 kPa TGF-β"}[t]] += v
    fn = read_pzfx(os.path.join(ONEDRIVE, "Results for Figures", "Figure_All_Conditions", "Results_Fibronectin",
                                "Results_Fibronectin.pzfx"))["Force GT vs AI"]
    vals["6 kPa fibronectin"] = [x for t, v in fn if t == "GT" for x in v]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "condition_table"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ML dataset counts by role
    man = []
    for f in sorted(os.listdir(os.path.join(ROOT, "experiments", "splits"))):
        if not f.endswith(".csv") or f.startswith(("zenodo", "spheroid", "test_u2os")):
            continue
        d = pd.read_csv(os.path.join(ROOT, "experiments", "splits", f)).assign(manifest=f[:-4])
        man.append(d)
    man = pd.concat(man)
    man["role"] = np.where(man.manifest == "alltune_train", "training",
                           np.where(man.manifest == "alltune_val", "validation", "test"))
    man["cond"] = [ml_condition(m, s, c) for m, s, c in zip(man.manifest, man.set, man.condition)]

    # GT values for my held-out images (from experiments.predict_panels; GT columns are model-independent)
    pm = pd.read_csv(os.path.join(ROOT, "experiments", "results", "predictions", "predictions_metrics.csv"))
    pm = pm[pm.model == "released"].merge(man[["manifest", "sample_dir", "cond", "role", "bf_md5"]],
                                          on=["manifest", "sample_dir"], how="inner")
    mine = pm[~pm.sample_dir.isin(DUP)]           # Position_08 left out of both drug groups

    # Lautaro's images_cells folders and per-cell values
    P = pd.read_csv(os.path.join(ROOT, "experiments", "results", "images_cells_audit", "positions.csv"))
    lmap = {"01kPa": "1 kPa", "10_kPa": "10 kPa", "06kPa_Untreated": "6 kPa collagen, untreated",
            "06kPa_DMSO": "6 kPa DMSO", "06kPa_Y27632": "6 kPa Y-27632", "06kPa_TGF-b": "6 kPa TGF-β"}
    P["cond"] = P.condition_folder.map(lmap)
    P["genuine"] = P.condition_mismatch_ML.astype(str) != "True"
    P["is_dup"] = P.position.eq("Position_08") & P.set_folder.isin(["Sum_Force_Values_DMSO_SET_II", "Sum_Force_Values_Y27632_SET_II"])
    v = "q_GT_Force_Lauti_per_Area"
    figvals = lautaro_figure_values()
    # 6 kPa untreated: Fig. 2C plots map metrics only (n = 153 in 6kPa_Results.pzfx); his per-cell GT values for the
    # same 153 cells come from the Manual_Quantification files of images_cells/06kPa_Untreated
    figvals["6 kPa collagen, untreated"] = list(P[P.cond == "6 kPa collagen, untreated"][v].dropna().values)
    lau_md5 = P.dropna(subset=[v]).drop_duplicates("bf_md5").set_index("bf_md5")[v]

    rows = []
    for c in CONDS:
        m = man[man.cond == c]
        tr, va, te = (m.role == "training").sum(), (m.role == "validation").sum(), (m.role == "test").sum()
        lp = P[P.cond == c]
        my = mine[mine.cond == c]
        lau_on_mine = my.bf_md5.map(lau_md5).dropna()
        fv = np.array(figvals[c]) if figvals[c] else None
        gen = lp[lp.genuine & ~lp.is_dup]
        rows.append({
            "condition": c,
            "1 Lautaro, manuscript n": MANUSCRIPT_N[c],
            "2 shared images_cells: positions": len(lp) if len(lp) else "not in folder",
            "2 shared images_cells: unique images": lp.bf_md5.nunique() if len(lp) else "",
            "2 shared images_cells: imaged in this condition": int(lp.genuine.sum()) if len(lp) else "",
            "3 n in Lautaro's Fig. 2/3 Prism files": len(fv) if fv is not None and len(fv) else "",
            "ML training": tr, "ML validation": va, "ML test": te, "ML total": tr + va + te,
            "4 my n (validation + test)": va + te,
            "4 my n used (Position_08 excluded)": len(my),
            "5 my mean GT, force region (Pa)": my.gt_region_mean_Pa.mean(),
            "5 SD": my.gt_region_mean_Pa.std(),
            "5 my mean GT, whole field (Pa)": my.gt_field_mean_Pa.mean(),
            "6 Lautaro mean GT, his Fig. n (force/area)": fv.mean() if fv is not None else np.nan,
            "6 SD": fv.std(ddof=1) if fv is not None else np.nan,
            "6 Lautaro: n imaged in this condition (Position_08 excluded)": len(gen) if len(lp) else "",
            "6 Lautaro mean, cells imaged in this condition (force/area)": gen[v].mean() if len(lp) else np.nan,
            "6 Lautaro values on my n: n matched": len(lau_on_mine),
            "6 Lautaro values on my n: mean (force/area)": lau_on_mine.mean() if len(lau_on_mine) else np.nan,
        })
    T = pd.DataFrame(rows)
    tot = {k: (T[k].sum() if pd.api.types.is_numeric_dtype(T[k]) and k.startswith(("ML", "4 my n (")) else "")
           for k in T.columns}
    tot["condition"] = "All"
    T = pd.concat([T, pd.DataFrame([tot])], ignore_index=True)

    # statistics for the box plots
    groups = {
        "Mine: validation + test images, GT in force region (Pa)": {c: mine[mine.cond == c].gt_region_mean_Pa.values for c in CONDS},
        "Lautaro: values as in his Figs 2-3 cohorts (force per cell area)": {c: np.array(figvals[c]) for c in CONDS},
        "Lautaro: his values, cells imaged in that condition only": {
            c: (P[(P.cond == c) & P.genuine & ~P.is_dup][v].dropna().values
                if c != "6 kPa fibronectin" else np.array(figvals[c])) for c in CONDS},
    }
    st = []
    for gname, g in groups.items():
        ps = []
        for a_, b_ in COMPARISONS:
            x, y = g[a_], g[b_]
            p = stats.mannwhitneyu(x, y).pvalue if len(x) and len(y) else np.nan
            ps.append(p)
            st.append(dict(source=gname, comparison=f"{b_} vs {a_}", n=f"{len(y)} vs {len(x)}",
                           mean_ratio=(y.mean() / x.mean()) if len(x) and len(y) else np.nan, p_mann_whitney=p))
        ps = np.array(ps); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
        for i, j in enumerate(order):
            run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
        for k, p in enumerate(adj):
            st[-len(adj) + k]["p_holm"] = p
    ST = pd.DataFrame(st)

    readme = pd.DataFrame({"column": [
        "1", "2", "3", "ML training/validation/test", "4", "5", "6", "Position_08", "Statistics"], "meaning": [
        "Cell numbers stated by Lautaro in the manuscript (Fig. 2/3 legends, Model Validation).",
        "Lautaro's shared images_cells folder: number of position folders, unique images (MD5), and how many of them "
        "were imaged in that condition (the rest are copies of 6 kPa untreated cells). Fibronectin is not in that folder.",
        "Number of GT values in Lautaro's REVISION Prism files (Superplots ... REVISION.pzfx; fibronectin: Results_Fibronectin.pzfx). 6 kPa untreated: Fig. 2C (n = 153 in 6kPa_Results.pzfx); per-cell values from images_cells/06kPa_Untreated.",
        "Image pairs in the ML split files (single cells, HFF).",
        "My held-out images = validation + test; 'used' excludes Position_08 from both DMSO and Y-27632.",
        "Mean GT traction stress of my held-out images: in the force region detected on the GT map, and over the whole field (Pa).",
        "Lautaro's GT force per cell area: as plotted in his figures; for cells imaged in that condition only; and for my "
        "held-out images where he has a value (matched by MD5). Units: his quantification (force per area).",
        "Same image in DMSO SET II and Y-27632 SET II; label pending; left out of both drug groups.",
        "Two-sided Mann-Whitney on cells, Holm-corrected over the 4 pre-specified comparisons per source."]})
    xl = os.path.join(a.out, "condition_table.xlsx")
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False)
        T.round(2).to_excel(w, sheet_name="conditions", index=False)
        ST.to_excel(w, sheet_name="comparisons", index=False)
    T.to_csv(os.path.join(a.out, "condition_table.csv"), index=False)
    ST.to_csv(os.path.join(a.out, "condition_comparisons.csv"), index=False)

    # box plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    col = {"1 kPa": "#6baed6", "6 kPa collagen, untreated": "#9e9e9e", "10 kPa": "#08519c", "6 kPa fibronectin": "#8c6bb1",
           "6 kPa DMSO": "#bdbdbd", "6 kPa Y-27632": "#41ab5d", "6 kPa TGF-β": "#d94801"}
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.4), gridspec_kw=dict(hspace=0.55))
    rng = np.random.default_rng(0)
    for ax, (gname, g) in zip(axs, groups.items()):
        data = [g[c] for c in CONDS]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k", lw=1))
        for patch, c in zip(bp["boxes"], CONDS):
            patch.set_facecolor(col[c]); patch.set_alpha(0.35)
        for i, d in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=4, color=col[CONDS[i]], lw=0, zorder=3)
        ax.set_xticks(range(1, len(CONDS) + 1), [f"{SHORT[c]}\nn={len(d)}" for c, d in zip(CONDS, data)])
        ax.set_ylabel("GT traction stress (Pa)" if gname.startswith("Mine") else "GT force per cell area")
        ax.set_title(gname, loc="left", fontsize=8)
        top = np.nanmax([np.nanmax(d) for d in data if len(d)])
        sub = ST[ST.source == gname].reset_index(drop=True)
        for k, (a_, b_) in enumerate(COMPARISONS):
            i, j = CONDS.index(a_) + 1, CONDS.index(b_) + 1
            y = top * (1.05 + 0.09 * (k % 2))
            p = sub.loc[k, "p_holm"]
            lab = "n.s." if p >= 0.05 else ("****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*")
            ax.plot([i, j], [y, y], color="k", lw=0.5)
            ax.text((i + j) / 2, y, f"{lab} (×{sub.loc[k, 'mean_ratio']:.2f})", ha="center", va="bottom", fontsize=6)
        ax.set_ylim(0, top * 1.3)
    fig.text(0.01, 0.005, "Boxes: median and IQR; dots: cells. Brackets: two-sided Mann-Whitney on cells, Holm-corrected "
             "(4 comparisons per panel); ×ratio = mean of the second group / first. Panel 1: my validation + test images; "
             "Position_08 excluded from DMSO and Y-27632.", fontsize=5.5, color="0.3")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"condition_boxplots.{ext}"), dpi=250, bbox_inches="tight")
    print("wrote", a.out)
    print(T.to_string())
    print(ST.round(4).to_string())


if __name__ == "__main__":
    main()
