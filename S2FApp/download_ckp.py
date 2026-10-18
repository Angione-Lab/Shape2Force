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
        # All checkpoints: single cell (S2F), spheroid S2F (ckp_spheroid_GN.pth, the model reported in the paper
        # and the app default) and the smaller supervised spheroid model (ckp_spheroid_FN.pth).
        for f in pth_files:
            hf_hub_download(repo_id=repo, filename=f, local_dir="/app")
        print("Downloaded checkpoints from", repo)
    except Exception as e:
        print("Could not download checkpoints:", e)
else:
    print("Checkpoints already present")
