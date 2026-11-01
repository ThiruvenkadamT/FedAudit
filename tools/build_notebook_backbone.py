"""Build notebooks/FedAudit_E4_ResNet18.ipynb: the brain MRI leakage-inflation experiment (E4) with an
ImageNet-pretrained ResNet-18 instead of the compact CNN.

    python tools/build_notebook_backbone.py path/to/fedaudit_results

The notebook embeds the package, the brain MRI manifest (data/brain.csv) and the audit pairs
(results/audit_pairs_brain.csv) from the given results folder, so it uses exactly the same splits and
the same flagged images as the main experiments. Only the four brain MRI datasets need to be attached.
"""
import argparse
import base64
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import ROOT, code, md, pkg_b64  # noqa: E402

INTRO = """
# FedAudit — E4 brain MRI with a pretrained ResNet-18

Repeats the leakage-inflation experiment on the brain MRI federation with an ImageNet-pretrained
ResNet-18 (10 training methods x 3 seeds = 30 runs). The data splits and the images flagged by the
audit are embedded in this notebook and are identical to those of the main experiments.

**Inputs to attach (+ Add Input):** the four brain MRI datasets (figshare/Cheng, SARTAJ, Br35H, Nickparvar).
**Settings:** GPU T4 x2 and Internet on (the pretrained ResNet-18 weights are downloaded once).

**Run 1:** keep `QUICK_TEST = True` and Run all (about 15 min).
**Run 2:** set `QUICK_TEST = False`, Save Version → Save & Run All (Commit). To continue a stopped run,
attach this notebook's previous output as an input and commit again; finished runs are skipped.
Download `fedaudit_e4_resnet18.zip` at the end.
"""

SETTINGS = """
# ============================== SETTINGS ==============================
QUICK_TEST = True          # True: short pipeline check. False: the 30 runs.
TIME_BUDGET_HOURS = 11.0   # stop starting new runs after this (Kaggle stops sessions at 12 h)
ROUNDS = 30
SEEDS = [0, 1, 2]
MODEL = "resnet18"
SOURCE_OVERRIDES = {}      # e.g. {"figshare": "/kaggle/input/my-figshare-copy"} if auto-detection misses one
# ======================================================================
"""

INSTALL = """
import base64, gzip, io, os, sys, tarfile, subprocess, time
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py"], check=False)
CODE_DIR = "/kaggle/working/fedaudit_code"
os.makedirs(CODE_DIR, exist_ok=True)
PKG = "__PKG__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG)), mode="r:gz").extractall(CODE_DIR, filter="data")
sys.path.insert(0, CODE_DIR)
for m in [m for m in sys.modules if m == "fedaudit" or m.startswith("fedaudit.")]:
    del sys.modules[m]
import fedaudit, torch
print("fedaudit code ready |", "CUDA GPUs:", torch.cuda.device_count())
"""

WEIGHTS = """
# Pretrained ResNet-18 weights: downloaded once here, or copied from an attached input (resnet18*.pth)
import glob, shutil
from torchvision.models import ResNet18_Weights
ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
target = os.path.join(ckpt_dir, os.path.basename(ResNet18_Weights.IMAGENET1K_V1.url))
if not os.path.exists(target):
    local = sorted(glob.glob("/kaggle/input/**/resnet18*.pth", recursive=True))
    if local:
        shutil.copy(local[0], target)
    else:
        try:
            ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        except Exception as e:
            raise RuntimeError("Could not download the ResNet-18 weights. Turn on Internet in the notebook settings "
                               "(Settings > Internet; it needs a phone-verified Kaggle account), or attach a dataset "
                               "that contains a resnet18*.pth file.") from e
print("ResNet-18 weights ready:", target, f"{os.path.getsize(target) / 1e6:.0f} MB")
"""

DATA = """
# Step 1: restore the brain MRI manifest and audit pairs of the main experiments, rebuild the image cache
import pandas as pd
from fedaudit.experiments import sources
from fedaudit.experiments.prepare import restore_stores
from fedaudit.experiments.runner import merge_previous

BRAIN_CSV = "__BRAIN__"
PAIRS_CSV = "__PAIRS__"
OLD_ROOTS = __ROOTS__

OUT = "/kaggle/working/fedaudit_e4_resnet18_quick" if QUICK_TEST else "/kaggle/working/fedaudit_e4_resnet18"
for d in ("data", "results", "runs"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
S = dict(out=OUT, input_root="/kaggle/input", code_dir=CODE_DIR, source_overrides=SOURCE_OVERRIDES,
         quick=1 if QUICK_TEST else 0, size=128, rounds=2 if QUICK_TEST else ROUNDS,
         seeds=SEEDS[:1] if QUICK_TEST else SEEDS, model=MODEL, experiments=["E4brain"],
         deadline=START + TIME_BUDGET_HOURS * 3600)

src = sources.detect_sources(S["input_root"], SOURCE_OVERRIDES)
print("sources found:", src)
df = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(BRAIN_CSV))))
for name, old in OLD_ROOTS.items():  # datasets may be mounted under a different folder in this session
    if name in src and old != src[name]:
        m = df["path"].str.startswith(old)
        df.loc[m, "path"] = src[name] + df.loc[m, "path"].str[len(old):]
df.to_csv(f"{OUT}/data/brain.csv", index=False)
open(f"{OUT}/results/audit_pairs_brain.csv", "wb").write(gzip.decompress(base64.b64decode(PAIRS_CSV)))
print(f"Merged {merge_previous(S)} finished runs from previous versions")
restore_stores(S, src)
"""

TRAIN = """
# Step 2: 30 training runs on both GPUs
from fedaudit.experiments.runner import run_all
run_all(S, n_gpus=max(1, torch.cuda.device_count()))
"""

ZIP = """
# Step 3: zip the runs
import glob, zipfile, shutil
name = "fedaudit_e4_resnet18_quick" if QUICK_TEST else "fedaudit_e4_resnet18"
with zipfile.ZipFile(f"/kaggle/working/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(p) and "/store/" not in p and "/claims/" not in p:
            z.write(p, os.path.join("fedaudit_e4_resnet18", os.path.relpath(p, OUT)))
shutil.rmtree(f"{OUT}/store", ignore_errors=True)
print(len(glob.glob(f"{OUT}/runs/*.json")), "runs finished |", "errors:", len(glob.glob(f"{OUT}/errors/*")))
print("Wrote", f"/kaggle/working/{name}.zip")
"""


def b64gz(path):
    return base64.b64encode(gzip.compress(open(path, "rb").read(), 9)).decode()


def main(res):
    roots = {k: v for k, v in json.load(open(f"{res}/dataset_report.json"))["sources"].items()
             if k in ("figshare", "sartaj", "br35h", "nickparvar")}
    data = (DATA.replace("__BRAIN__", b64gz(f"{res}/data/brain.csv"))
                .replace("__PAIRS__", b64gz(f"{res}/results/audit_pairs_brain.csv"))
                .replace("__ROOTS__", json.dumps(roots)))
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[md(INTRO), code(SETTINGS), code(INSTALL.replace("__PKG__", pkg_b64())), code(WEIGHTS), code(data), code(TRAIN),
                     code(ZIP)])
    path = os.path.join(ROOT, "notebooks", "FedAudit_E4_ResNet18.ipynb")
    json.dump(nb, open(path, "w"), indent=1)
    print("wrote", path, f"{os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="fedaudit_results folder of the main experiments")
    main(ap.parse_args().results)
