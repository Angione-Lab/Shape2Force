"""
Audit the legacy dataset tree against the repository dataset and (optionally) complete the
repository dataset with every image used in the paper, plus split manifests.

Step 1 - audit only (nothing is copied):
    python -m experiments.prepare_dataset --legacy-root C:/Users/Kaveh/Documents/cellforce/dataset

Step 2 - copy missing images into S2F/dataset and write the manifests:
    python -m experiments.prepare_dataset --legacy-root C:/Users/Kaveh/Documents/cellforce/dataset --write

What the audit answers (written to experiments/results/dataset_audit/):
  counts.csv                 number of samples per legacy and repository set
  published_vs_legacy.csv    is S2F/dataset/singlecell (236/46) identical to legacy fibro_split? spheroids?
  alltune_composition.csv    where every image of all_tune/train (280) and all_tune/test (81) comes from
  missing_from_published.csv all_tune images that are NOT in S2F/dataset (i.e. not in the Zenodo v1 deposit)
  overlap_matrix.csv         shared bright-field images between every pair of sets (leakage check)
  substrate_codes.csv        substrate code the original code derived from each all_tune folder name
  report.txt                 human-readable summary
Images are matched by the MD5 of the bright-field file, so renamed/copied folders are still recognised.

With --write:
  dataset/singlecell_extra/<set>/<folder>/      all_tune images missing from dataset/singlecell
  dataset/singlecell_external/<set>/<folder>/   independent test sets
  experiments/splits/*.csv                       manifests (commit these to git):
      alltune_train.csv, alltune_val.csv        split of the released single-cell model (280 / 81)
      zenodo_v1_train.csv, zenodo_v1_test.csv   published 6 kPa split (236 / 46)
      spheroid_train.csv, spheroid_val.csv      spheroid split (129 / 28)
      test_<set>.csv                            one per independent test set
"""
import argparse
import collections
import csv
import os
import shutil

from experiments.layout import (DEFAULT_DATASET_ROOT, LEGACY_SETS, PUBLISHED_SETS, RESULTS_DIR,
                                legacy_code_to_config_key, legacy_substrate_code, manifest_path,
                                md5, nominal_config_key, sample_folders, write_manifest)


def index_tree(root, rel):
    """List samples of one set: [{folder, bf, hm, dir, md5}]."""
    out = []
    for folder, bf, hm in sample_folders(os.path.join(root, rel)):
        out.append({"folder": folder, "bf": bf, "hm": hm, "dir": os.path.dirname(bf), "md5": md5(bf)})
    return out


def discover_unlisted(root, listed_rel, max_depth=4):
    """Directories under root that contain sample folders but are not in LEGACY_SETS (so nothing is missed)."""
    found = []
    listed = {os.path.normpath(os.path.join(root, r)) for r in listed_rel}
    base_depth = os.path.normpath(root).count(os.sep)
    for dp, dn, _ in os.walk(root):
        if os.path.normpath(dp).count(os.sep) - base_depth >= max_depth:
            dn[:] = []
            continue
        if os.path.normpath(dp) in listed:
            dn[:] = []
            continue
        n = sum(1 for _ in sample_folders(dp))
        if n:
            found.append((os.path.relpath(dp, root), n))
            dn[:] = []
    return found


def row_for(sample_dir, split, set_name, meta, legacy_folder, bf_md5):
    code = legacy_substrate_code(legacy_folder) if legacy_folder else ""
    return {"sample_dir": sample_dir.replace(os.sep, "/"), "split": split, "set": set_name,
            "condition": meta.get("condition", ""), "cell_type": meta.get("cell_type", ""),
            "coating": meta.get("coating", ""), "young_Pa": meta.get("young_Pa", ""),
            "tfm_pixelsize": meta.get("tfm_pixelsize", ""), "ythreshold_Pa": meta.get("ythreshold_Pa", ""),
            "legacy_folder": legacy_folder, "legacy_substrate_code": code,
            "substrate_legacy": legacy_code_to_config_key(code) if meta.get("cell_type") != "LS174T spheroid" else "",
            "substrate_nominal": (nominal_config_key(meta.get("young_Pa"), meta.get("tfm_pixelsize"), meta.get("cell_type"))
                                  if meta.get("cell_type") != "LS174T spheroid" else "") or "",
            "bf_md5": bf_md5}


def copy_sample(src_dir, dst_dir, expected_md5):
    if os.path.isdir(dst_dir):
        bf = [f for f in os.listdir(dst_dir) if f.endswith("BF_001.tif")]
        if bf and md5(os.path.join(dst_dir, bf[0])) == expected_md5:
            return "exists"
        raise RuntimeError(f"{dst_dir} exists with different content; refusing to overwrite")
    os.makedirs(dst_dir)
    for f in os.listdir(src_dir):
        if os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    return "copied"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-root", required=True, help="old dataset folder (contains single_cell/ and spheroid/)")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="repository dataset (default: S2F/dataset)")
    ap.add_argument("--write", action="store_true", help="copy missing images and write split manifests")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "dataset_audit"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = []

    def say(s=""):
        print(s); rep.append(s)

    # ---------------- index ----------------
    legacy = {k: index_tree(a.legacy_root, v[0]) for k, v in LEGACY_SETS.items()}
    pub = {k: index_tree(a.dataset_root, v) for k, v in PUBLISHED_SETS.items()}
    say(f"legacy root : {a.legacy_root}\ndataset root: {a.dataset_root}\n")
    with open(os.path.join(a.out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tree", "set", "path", "n_samples", "n_unique_bf"])
        for k, v in legacy.items():
            w.writerow(["legacy", k, LEGACY_SETS[k][0], len(v), len({s['md5'] for s in v})])
            say(f"  legacy    {k:20s} n={len(v):4d}")
        for k, v in pub.items():
            w.writerow(["repository", k, PUBLISHED_SETS[k], len(v), len({s['md5'] for s in v})])
            say(f"  repository {k:19s} n={len(v):4d}")
    unlisted = discover_unlisted(a.legacy_root, [v[0] for v in LEGACY_SETS.values()])
    if unlisted:
        say("\nFolders with samples in the legacy tree that are NOT listed in experiments/layout.py:")
        for rel, n in unlisted:
            say(f"  {rel}  ({n} samples)")
    if not legacy["all_tune/train"]:
        say("\nERROR: single_cell/all_tune/train not found under --legacy-root."); open(os.path.join(a.out, "report.txt"), "w").write("\n".join(rep)); return

    # ---------------- published vs legacy ----------------
    say("\nRepository dataset vs legacy:")
    rows = []
    for p, l in [("singlecell/train", "fibro_split/train"), ("singlecell/test", "fibro_split/test"),
                 ("spheroid/train", "spheroid/train"), ("spheroid/test", "spheroid/test")]:
        P, Lg = {s["md5"] for s in pub[p]}, {s["md5"] for s in legacy[l]}
        r = {"repository_set": p, "legacy_set": l, "n_repository": len(P), "n_legacy": len(Lg),
             "shared": len(P & Lg), "only_repository": len(P - Lg), "only_legacy": len(Lg - P)}
        rows.append(r); say(f"  {p:18s} vs {l:18s}: shared {r['shared']}, only repo {r['only_repository']}, only legacy {r['only_legacy']}")
    with open(os.path.join(a.out, "published_vs_legacy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---------------- all_tune composition ----------------
    by_md5_pub = {s["md5"]: (k, s) for k, v in pub.items() for s in v}
    src_keys = [k for k, v in LEGACY_SETS.items() if v[1] in ("published", "source", "external")]
    by_md5_leg = collections.defaultdict(list)
    for k in src_keys:
        for s in legacy[k]:
            by_md5_leg[s["md5"]].append((k, s))
    comp, missing, codes = [], [], collections.Counter()
    for split_key, split in [("all_tune/train", "train"), ("all_tune/test", "val")]:
        for s in legacy[split_key]:
            in_pub = by_md5_pub.get(s["md5"])
            srcs = [k for k, _ in by_md5_leg.get(s["md5"], [])]
            src = next((k for k in srcs if LEGACY_SETS[k][1] != "external"), srcs[0] if srcs else "UNMATCHED")
            code = legacy_substrate_code(s["folder"])
            codes[(split, code)] += 1
            r = {"alltune_split": split, "alltune_folder": s["folder"], "source_legacy_set": src,
                 "all_matching_sets": ";".join(srcs), "in_repository_dataset": in_pub[0] if in_pub else "",
                 "legacy_substrate_code": code, "bf_md5": s["md5"]}
            comp.append(r)
            if not in_pub:
                missing.append(r)
    for name, data in [("alltune_composition.csv", comp), ("missing_from_published.csv", missing)]:
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(data)
    with open(os.path.join(a.out, "substrate_codes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["alltune_split", "legacy_substrate_code", "n", "config_key_fed_to_model"])
        for (sp, c), n in sorted(codes.items()):
            w.writerow([sp, c, n, legacy_code_to_config_key(c)])
    say("\nall_tune composition (split x source set):")
    cc = collections.Counter((r["alltune_split"], r["source_legacy_set"]) for r in comp)
    for (sp, src), n in sorted(cc.items()):
        say(f"  {sp:5s}  {src:20s} {n:4d}")
    say(f"\nall_tune images NOT in the repository dataset (Zenodo v1): {len(missing)}")
    for (sp, src), n in sorted(collections.Counter((r['alltune_split'], r['source_legacy_set']) for r in missing).items()):
        say(f"  {sp:5s}  {src:20s} {n:4d}")
    say("\nSubstrate code the original code fed for all_tune folders (folder_name[:4]):")
    for (sp, c), n in sorted(codes.items()):
        say(f"  {sp:5s}  '{c}' -> {legacy_code_to_config_key(c):18s} {n:4d}")

    # ---------------- overlaps / leakage ----------------
    names = list(legacy)
    sets = {k: {s["md5"] for s in legacy[k]} for k in names}
    with open(os.path.join(a.out, "overlap_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set"] + names)
        for x in names:
            w.writerow([x] + [len(sets[x] & sets[y]) for y in names])
    tv = sets["all_tune/train"] | sets["all_tune/test"]
    say("\nLeakage check - independent test sets vs all_tune train+val (must be 0):")
    for k, v in LEGACY_SETS.items():
        if v[1] == "external":
            say(f"  {k:20s} {len(sets[k] & tv):3d}   (of {len(sets[k])})")
    say("Validation-set content - legacy 'test' folders inside all_tune/test (val):")
    for k in ["fibro_split/test", "1kpa_new/test", "10kpa_new/test"]:
        say(f"  {k:20s} {len(sets[k] & sets['all_tune/test']):3d} of {len(sets[k])} in val; {len(sets[k] & sets['all_tune/train']):3d} in train")

    # ---------------- write ----------------
    if a.write:
        say("\nCopying and writing manifests ...")
        n_copied = collections.Counter()
        alltune_rows = {"train": [], "val": []}
        for r, s in zip(comp, legacy["all_tune/train"] + legacy["all_tune/test"]):
            split, src = r["alltune_split"], r["source_legacy_set"]
            meta = LEGACY_SETS[src][3] if src in LEGACY_SETS else {}
            if r["in_repository_dataset"]:
                pk, ps = by_md5_pub[s["md5"]]
                sample_dir = os.path.join(PUBLISHED_SETS[pk], ps["folder"])
            else:
                clean = LEGACY_SETS[src][2] if src in LEGACY_SETS and LEGACY_SETS[src][2] else "unmatched"
                sample_dir = os.path.join("singlecell_extra", clean, s["folder"])
                n_copied[copy_sample(s["dir"], os.path.join(a.dataset_root, sample_dir), s["md5"])] += 1
            alltune_rows[split].append(row_for(sample_dir, split, src, meta, s["folder"], s["md5"]))
        write_manifest(manifest_path("alltune_train"), alltune_rows["train"])
        write_manifest(manifest_path("alltune_val"), alltune_rows["val"])
        for pk, man, lk in [("singlecell/train", "zenodo_v1_train", "fibro_split/train"),
                            ("singlecell/test", "zenodo_v1_test", "fibro_split/test"),
                            ("spheroid/train", "spheroid_train", "spheroid/train"),
                            ("spheroid/test", "spheroid_val", "spheroid/test")]:
            meta = LEGACY_SETS[lk][3]
            write_manifest(manifest_path(man), [row_for(os.path.join(PUBLISHED_SETS[pk], s["folder"]), man.split("_")[-1],
                                                        lk, meta, s["folder"], s["md5"]) for s in pub[pk]])
        for k, (rel, role, clean, meta) in LEGACY_SETS.items():
            if role != "external" or not legacy[k]:
                continue
            rows_ = []
            for s in legacy[k]:
                sample_dir = os.path.join("singlecell_external", clean, s["folder"])
                n_copied[copy_sample(s["dir"], os.path.join(a.dataset_root, sample_dir), s["md5"])] += 1
                rows_.append(row_for(sample_dir, "test", k, meta, s["folder"], s["md5"]))
            write_manifest(manifest_path(f"test_{clean}"), rows_)
        say(f"  sample folders: {dict(n_copied)}")
        say(f"  manifests written to experiments/splits/ - commit them to git")
    else:
        say("\n(audit only - rerun with --write to copy the missing images and write experiments/splits/*.csv)")
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(rep))
    print(f"\nReport: {os.path.join(a.out, 'report.txt')}")


if __name__ == "__main__":
    main()
