"""Two-GPU job runner with resume and a time budget (Kaggle sessions stop at 12 h).

  python -m fedaudit.experiments.runner worker --settings S.json --gpu 0
Workers claim jobs with atomic lock files, so any number of GPUs can share one list.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def merge_previous(S: dict) -> int:
    """Copy finished work from earlier versions attached as inputs (notebook output or uploaded zips)."""
    out, root = S["out"], S["input_root"]
    n0 = len(glob.glob(f"{out}/runs/*.json"))
    for d in glob.glob(f"{root}/**/fedaudit_results", recursive=True):
        if os.path.abspath(d) == os.path.abspath(out):
            continue
        for sub in ("runs", "results", "data", "emb"):
            if os.path.isdir(f"{d}/{sub}"):
                shutil.copytree(f"{d}/{sub}", f"{out}/{sub}", dirs_exist_ok=True)
    for z in glob.glob(f"{root}/**/fedaudit_results*.zip", recursive=True):
        if "quick" in os.path.basename(z):
            continue  # quick-test output never mixes with real runs
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                rel = name.split("fedaudit_results/", 1)[-1]
                if rel.split("/")[0] in ("runs", "results", "data", "emb") and not name.endswith("/"):
                    dst = f"{out}/{rel}"
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with zf.open(name) as src, open(dst, "wb") as f:
                            shutil.copyfileobj(src, f)
    # claims from an earlier session are stale
    for c in glob.glob(f"{out}/claims/*"):
        os.remove(c)
    return len(glob.glob(f"{out}/runs/*.json")) - n0


EST_MIN = {"E4brain": 8, "E4ham": 6, "E4leak": 6, "E6": 5, "E5nat": 5}


def worker(settings_path: str, gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)  # sklearn notes on tiny quick-test splits
    S = json.load(open(settings_path))
    from .jobs import job_id, run_job
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(1, S.get("n_workers", 1))))
    jobs = json.load(open(f"{S['out']}/jobs.json"))
    os.makedirs(f"{S['out']}/claims", exist_ok=True)
    os.makedirs(f"{S['out']}/errors", exist_ok=True)
    tag = f"[GPU{gpu}]"
    for j in jobs:
        jid = job_id(j)
        if os.path.exists(f"{S['out']}/runs/{jid}.json"):
            continue
        left = S["deadline"] - time.time()
        if left < EST_MIN.get(j["exp"], 6) * 60 * S.get("time_safety", 1.5):
            print(f"{tag} [time budget] {left / 60:.0f} min left — stopping", flush=True)
            break
        try:
            fd = os.open(f"{S['out']}/claims/{jid}", os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue
        print(f"{tag} === {jid}", flush=True)
        t = time.time()
        try:
            run_job(S, j, device)
            r = json.load(open(f"{S['out']}/runs/{jid}.json"))
            allm = [m for m in r["metrics"] if m["site"] == "ALL"]
            first = allm[0] if allm else {}
            print(f"{tag} === done in {(time.time() - t) / 60:.1f} min: {first.get('view')} acc={first.get('acc', float('nan')):.3f} "
                  f"auc={first.get('auc', float('nan')):.3f}", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"{tag} !!! run failed: {jid}\n{err[-1500:]}", flush=True)
            with open(f"{S['out']}/errors/{jid}.txt", "w") as f:
                f.write(err)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()


def run_all(S: dict, n_gpus: int, cpu_task=None):
    """Write the job list and run one worker per GPU in parallel (blocks until done/budget)."""
    from .jobs import all_jobs, job_id
    out = S["out"]
    avail = set()
    for name, p in (("brain", "results/audit_pairs_brain.csv"), ("ham", "results/audit_pairs_ham.csv"),
                    ("brain_dedup", "data/brain_dedup.csv"), ("ct", "data/ct_pool.csv")):
        if os.path.exists(f"{out}/{p}"):
            avail.add(name)
    jobs = all_jobs(S, avail)
    with open(f"{out}/jobs.json", "w") as f:
        json.dump(jobs, f)
    S["n_workers"] = max(1, n_gpus)
    sp = f"{out}/settings.json"
    with open(sp, "w") as f:
        json.dump(S, f)
    print(progress(S, jobs), flush=True)
    procs = [subprocess.Popen([sys.executable, "-u", "-m", "fedaudit.experiments.runner", "worker", "--settings", sp, "--gpu", str(g)],
                              cwd=S["code_dir"]) for g in range(max(1, n_gpus))]
    if cpu_task is not None:  # e.g. E2 cost scaling runs on the CPU while the GPUs train
        try:
            cpu_task()
        except Exception:
            traceback.print_exc()
    for p in procs:
        p.wait()
    print(progress(S, jobs), flush=True)


def progress(S, jobs) -> str:
    from .jobs import job_id
    done = {os.path.basename(p)[:-5] for p in glob.glob(f"{S['out']}/runs/*.json")}
    by = {}
    for j in jobs:
        d, t = by.get(j["exp"], (0, 0))
        by[j["exp"]] = (d + (job_id(j) in done), t + 1)
    return "Progress: " + ", ".join(f"{k}: {d} / {t}" for k, (d, t) in by.items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--settings")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(a.settings, a.gpu)
