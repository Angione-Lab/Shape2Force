"""Download checkpoints from Hugging Face if ckp is empty (for Space deployment)."""
import os
from pathlib import Path

ckp = Path("/app/ckp")
ckp_single_cell = ckp / "single_cell"
ckp_spheroid = ckp / "spheroid"
has_any = list(ckp.glob("*.pth")) or list(ckp_single_cell.glob("*.pth")) or list(ckp_spheroid.glob("*.pth"))

if not has_any:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo = os.environ.get("HF_MODEL_REPO", "Angione-Lab/Shape2Force")
        files = list_repo_files(repo)
        pth_files = [f for f in files if f.startswith("ckp/") and f.endswith(".pth")]
        # For spheroid: only download ckp_spheroid_FN.pth (not ckp_spheroid_GN.pth or others)
        def should_download(f):
            if "spheroid" in f and "ckp_spheroid_FN.pth" not in f:
                return False
            return True
        pth_files = [f for f in pth_files if should_download(f)]
        for f in pth_files:
            hf_hub_download(repo_id=repo, filename=f, local_dir="/app")
        print("Downloaded checkpoints from", repo)
    except Exception as e:
        print("Could not download checkpoints:", e)
else:
    print("Checkpoints already present")
