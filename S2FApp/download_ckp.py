"""Download checkpoints from Hugging Face if ckp is empty (for Space deployment)."""
import os
from pathlib import Path

ckp = Path("/app/ckp")
if not list(ckp.glob("*.pth")):
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo = os.environ.get("HF_MODEL_REPO", "kaveh/Shape2Force")
        files = list_repo_files(repo)
        pth_files = [f for f in files if f.startswith("ckp/") and f.endswith(".pth")]
        for f in pth_files:
            hf_hub_download(repo_id=repo, filename=f, local_dir="/app")
        print("Downloaded checkpoints from", repo)
    except Exception as e:
        print("Could not download checkpoints:", e)
else:
    print("Checkpoints already present")
