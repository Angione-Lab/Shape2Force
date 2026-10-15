"""
Dataset layout, per-set metadata and manifest helpers for the S2F experiments.

Two dataset trees are involved:

* the **repository dataset** ``S2F/dataset/`` (git-ignored; same content as the Zenodo deposit):
      singlecell/{train,test}/<sample>/      236 / 46 fibroblasts on 6 kPa collagen gels
      spheroid/{train,test}/<sample>/        129 / 28 LS174T spheroids
  ``experiments/prepare_dataset.py --write`` adds the remaining sets used in the paper:
      singlecell_extra/<set>/<sample>/       training/validation images of the released model
                                             that are not in singlecell/ (1 and 10 kPa)
      singlecell_external/<set>/<sample>/    independent test sets (drugs, fibronectin, ...)

* the **legacy tree** used to develop the model (``cellforce/dataset`` on the workstation), whose
  ``single_cell/all_tune/{train,test}`` folder (280 / 81 pairs) is the split of the released
  single-cell checkpoint.

Splits are defined by manifest CSV files in ``experiments/splits/`` (tracked in git), one row per
sample, so that every experiment is reproducible from the repository dataset alone.
"""
import csv
import hashlib
import os

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Override with environment variables if needed (e.g. dataset on another disk)
DEFAULT_DATASET_ROOT = os.environ.get("S2F_DATASET_ROOT", os.path.join(S2F_ROOT, "dataset"))
SPLITS_DIR = os.environ.get("S2F_SPLITS_DIR", os.path.join(S2F_ROOT, "experiments", "splits"))
RUNS_DIR = os.environ.get("S2F_RUNS_DIR", os.path.join(S2F_ROOT, "experiments", "runs"))
RESULTS_DIR = os.environ.get("S2F_RESULTS_DIR", os.path.join(S2F_ROOT, "experiments", "results"))

BF_SUFFIX = "BF_001.tif"
HM_SUFFIX = "_gray.jpg"
MASK_NAMES = ("cell_mask.tif", "cell_mask.png", "mask.tif", "mask.png")  # optional, for the mask-only baseline

# TFM reconstruction settings (from the MATLAB settings files of each set).
# ythreshold_Pa = upper limit of the colour scale used to export the heatmaps -> grey 1.0 == ythreshold_Pa.
# tfm_pixelsize = value of Settings.PixelSize used in the TFM reconstruction.
FIBRO_6KPA = dict(cell_type="HFF", coating="collagen I", young_Pa=6000, tfm_pixelsize=3.0769, ythreshold_Pa=500)

# Legacy sets: key -> (relative path under legacy root, role, clean name, metadata)
# role: 'alltune'   split of the released single-cell model (train 280 / val 81)
#       'published' already in S2F/dataset
#       'source'    pool from which all_tune was assembled (1 / 10 kPa)
#       'external'  independent test set
#       'union'     duplicate collection, audited only
LEGACY_SETS = {
    "all_tune/train": ("single_cell/all_tune/train", "alltune", None, {}),
    "all_tune/test": ("single_cell/all_tune/test", "alltune", None, {}),
    "fibro_split/train": ("single_cell/fibro_split/train", "published", "fibro_6kPa_train", dict(FIBRO_6KPA, condition="6 kPa collagen")),
    "fibro_split/test": ("single_cell/fibro_split/test", "published", "fibro_6kPa_setVI", dict(FIBRO_6KPA, condition="6 kPa collagen")),
    "1kpa_new/train": ("single_cell/1kpa_new/train", "source", "fibro_1kPa_train", dict(FIBRO_6KPA, young_Pa=1000, condition="1 kPa collagen")),
    "1kpa_new/test": ("single_cell/1kpa_new/test", "source", "fibro_1kPa_setI", dict(FIBRO_6KPA, young_Pa=1000, condition="1 kPa collagen")),
    "10kpa_new/train": ("single_cell/10kpa_new/train", "source", "fibro_10kPa_train", dict(FIBRO_6KPA, young_Pa=10000, condition="10 kPa collagen")),
    "10kpa_new/test": ("single_cell/10kpa_new/test", "source", "fibro_10kPa_setI", dict(FIBRO_6KPA, young_Pa=10000, condition="10 kPa collagen")),
    "1kPa_SET_III": ("single_cell/last_new_1kpa_10kpa/Heatmaps_01kPa_SET_III_for_Kaveh", "external", "fibro_1kPa_setIII", dict(FIBRO_6KPA, young_Pa=1000, condition="1 kPa collagen")),
    "DMSO_SET_II": ("single_cell/last_new_1kpa_10kpa/Heatmaps_DMSO_SET_II_for_Kaveh", "external", "fibro_DMSO_setII", dict(FIBRO_6KPA, condition="DMSO")),
    "Y27632_SET_II": ("single_cell/last_new_1kpa_10kpa/Heatmaps_Y27632_SET_II_for_Kaveh", "external", "fibro_Y27632_setII", dict(FIBRO_6KPA, condition="Y-27632")),
    "Control_SET_I": ("single_cell/Drugs/Fibros_Control_SET_I_for_Kaveh", "external", "fibro_DMSO_setI", dict(FIBRO_6KPA, condition="DMSO")),
    "Fibronectin_SET_I": ("single_cell/Drugs/Fibros_Fibronectin_SET_I_for_Kaveh", "external", "fibro_fibronectin_setI", dict(FIBRO_6KPA, coating="fibronectin", condition="fibronectin")),
    "TGFB_SET_I": ("single_cell/Drugs/Fibros_TGFB_SETI_for_Kaveh", "external", "fibro_TGFb_setI", dict(FIBRO_6KPA, condition="TGF-beta")),
    "Y27632_SET_I": ("single_cell/Drugs/Fibros_Y27632_SET_I_for_Kaveh", "external", "fibro_Y27632_setI", dict(FIBRO_6KPA, condition="Y-27632")),
    "U2OS/test": ("single_cell/u2os/test", "external", "u2os_6kPa", dict(cell_type="U2OS", coating="collagen I", young_Pa=6000, tfm_pixelsize=6.1538, ythreshold_Pa=500, condition="U2OS")),
    "Drugs/all": ("single_cell/Drugs/all", "union", None, {}),
    "spheroid/train": ("spheroid/train", "published", "spheroid_train", dict(cell_type="LS174T spheroid", coating="collagen I", ythreshold_Pa=200, condition="spheroid")),
    "spheroid/test": ("spheroid/test", "published", "spheroid_val", dict(cell_type="LS174T spheroid", coating="collagen I", ythreshold_Pa=200, condition="spheroid")),
}

# Published repository dataset (S2F/dataset), relative paths
PUBLISHED_SETS = {
    "singlecell/train": "singlecell/train",
    "singlecell/test": "singlecell/test",
    "spheroid/train": "spheroid/train",
    "spheroid/test": "spheroid/test",
}

MANIFEST_FIELDS = ["sample_dir", "split", "set", "condition", "cell_type", "coating", "young_Pa",
                   "tfm_pixelsize", "ythreshold_Pa", "legacy_folder", "legacy_substrate_code",
                   "substrate_legacy", "substrate_nominal", "bf_md5"]


def legacy_substrate_code(folder_name):
    """Substrate code exactly as the original training code derived it (folder_name[:4]; 'kpan*' -> [:5])."""
    code = folder_name[:4]
    return folder_name[:5] if code.startswith("kpan") else code


def legacy_code_to_config_key(code):
    """S2F config key that encodes the same (pixel size, Young's modulus) as the original code's substrate code."""
    return {"kpa0": "PDMS_1kPa", "kpa1": "PDMS_10kPa", "kpan0": "PDMS_1kPa_3um",
            "kpan1": "PDMS_10kPa_3um", "u2os": "U2OS_PDMS"}.get(code, "fibroblasts_PDMS")


def nominal_config_key(young_Pa, tfm_pixelsize, cell_type="HFF"):
    """S2F config key for the nominal Young's modulus and TFM pixel-size setting of a sample."""
    if cell_type == "U2OS":
        return "U2OS_PDMS"
    if young_Pa in (None, ""):
        return None
    young_Pa, px = float(young_Pa), float(tfm_pixelsize or 3.0769)
    if young_Pa == 6000:
        return "fibroblasts_PDMS"
    suffix = "_3um" if abs(px - 3.0769) < 1e-3 else ""
    return {1000: "PDMS_1kPa", 10000: "PDMS_10kPa"}.get(int(young_Pa), "fibroblasts_PDMS") + suffix


def sample_folders(root):
    """Yield (folder_name, bf_path, hm_path) for every direct sub-folder holding a BF image and a gray heatmap."""
    if not os.path.isdir(root):
        return
    for sub in sorted(os.listdir(root)):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        files = os.listdir(p)
        bf = [f for f in files if f.endswith(BF_SUFFIX)]
        hm = [f for f in files if f.endswith(HM_SUFFIX)]
        if bf and hm:
            yield sub, os.path.join(p, bf[0]), os.path.join(p, hm[0])


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def manifest_path(name):
    return os.path.join(SPLITS_DIR, f"{name}.csv")


def list_manifests():
    if not os.path.isdir(SPLITS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(SPLITS_DIR) if f.endswith(".csv"))
