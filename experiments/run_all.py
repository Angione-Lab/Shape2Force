"""
Run the revision experiments end to end (cross-platform; run from the S2F repository root).

    python -m experiments.run_all --legacy-root C:/Users/Kaveh/Documents/cellforce/dataset --steps audit
    python -m experiments.run_all --legacy-root ... --steps prepare,eval_released
    python -m experiments.run_all --steps train,eval_runs --variants s2f_full,unet,s2f_no_gan
    python -m experiments.run_all --legacy-root ... --stage1 C:/.../outputs/checkpoints/last/last_checkpoint_final.pth --steps eval_released

steps:
  audit          dataset audit only (no copying)                          -> experiments/results/dataset_audit/
  prepare        copy missing images into dataset/ + write experiments/splits/*.csv
  eval_released  released checkpoints (+ substrate sweep, mean-map baseline, optional stage-1)
                                                                          -> experiments/results/eval/
  train          train the listed variants (default: configs.PRIORITY), skipping finished runs.
                 One job per GPU in parallel (--gpus 0,1); the next variant starts as soon as a GPU is free.
                 Logs: experiments/runs/<variant>_seed<seed>/train.log
  eval_runs      evaluate every experiments/runs/*/best_checkpoint.pth
"""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.configs import PRIORITY, get_config
from experiments.layout import RUNS_DIR, S2F_ROOT


def sh(args):
    print("\n$ " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m"] + args, cwd=S2F_ROOT)
    if r.returncode:
        sys.exit(r.returncode)


def finished(variant, seed):
    h = os.path.join(RUNS_DIR, f"{variant}_seed{seed}", "history.json")
    if not os.path.exists(h):
        return False
    return len(json.load(open(h))) >= get_config(variant)["epochs"]


def gpu_queue(queue, gpus, num_workers, extra=()):
    """Run training jobs in parallel, one per GPU (each job sees only its GPU as cuda:0)."""
    running, failed = {}, []
    queue = list(queue)
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                v, seed = queue.pop(0)
                run_dir = os.path.join(RUNS_DIR, f"{v}_seed{seed}")
                os.makedirs(run_dir, exist_ok=True)
                log = open(os.path.join(run_dir, "train.log"), "a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                cmd = [sys.executable, "-m", "experiments.train", "--variant", v, "--seed", seed, "--resume",
                       "--num-workers", num_workers] + list(extra)
                running[gpu] = (subprocess.Popen(cmd, cwd=S2F_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), v, seed, log)
                print(f"[{time.strftime('%H:%M')}] GPU {gpu}: started {v} seed {seed}  (log {run_dir}/train.log)", flush=True)
        time.sleep(10 if extra else 30)
        for gpu, (proc, v, seed, log) in list(running.items()):
            if proc.poll() is not None:
                log.close(); del running[gpu]
                status = "finished" if proc.returncode == 0 else f"FAILED (exit {proc.returncode}, see train.log)"
                if proc.returncode:
                    failed.append((v, seed))
                print(f"[{time.strftime('%H:%M')}] GPU {gpu}: {v} seed {seed} {status}", flush=True)
    if failed:
        print(f"failed jobs: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--stage1", default=None, help="path to last_checkpoint_final.pth")
    ap.add_argument("--steps", default="audit")
    ap.add_argument("--variants", default=",".join(PRIORITY))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--gpus", default="0,1", help="GPU ids; one training job per GPU at a time")
    ap.add_argument("--num-workers", default="2", help="DataLoader workers per training job")
    ap.add_argument("--extra-train-args", default="", help="passed to experiments.train, e.g. '--epochs 1 --max-batches 2 --allow-cpu' for a smoke test")
    a = ap.parse_args()
    steps = a.steps.split(",")
    if ("audit" in steps or "prepare" in steps) and not a.legacy_root:
        sys.exit("--legacy-root is required for audit/prepare")
    if "audit" in steps:
        sh(["experiments.prepare_dataset", "--legacy-root", a.legacy_root])
    if "prepare" in steps:
        sh(["experiments.prepare_dataset", "--legacy-root", a.legacy_root, "--write"])
    if "eval_released" in steps:
        cmd = ["experiments.evaluate", "--released", "--sweep", "--meanmap"]
        if a.stage1:
            cmd += ["--stage1", a.stage1]
        sh(cmd)
    if "train" in steps:
        queue = [(v, s) for v in a.variants.split(",") for s in a.seeds.split(",") if not finished(v, int(s))]
        print(f"{len(queue)} training jobs on GPUs {a.gpus}: {queue}")
        gpu_queue(queue, a.gpus.split(","), a.num_workers, a.extra_train_args.split())
    if "eval_runs" in steps:
        sh(["experiments.evaluate", "--runs"])


if __name__ == "__main__":
    main()
