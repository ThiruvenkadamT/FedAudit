"""Build notebooks/FedAudit_Experiments.ipynb with the fedaudit package embedded, so the notebook
is the only file you import into Kaggle.   python tools/build_notebook.py"""
import base64
import io
import json
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pkg_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for dp, dns, fns in os.walk(os.path.join(ROOT, "fedaudit")):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for f in fns:
                if f.endswith(".py"):
                    full = os.path.join(dp, f)
                    tf.add(full, arcname=os.path.relpath(full, ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(True))


def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n").splitlines(True))


INTRO = """
# FedAudit — all experiments (E1–E6)

Privacy-preserving auditing of cross-site leakage and shortcut reliance in federated medical imaging.

**Inputs to attach (+ Add Input):** the four brain MRI sources (figshare/Cheng, SARTAJ, Br35H, Nickparvar),
HAM10000, and the Brain Tumor Multimodal Image (CT & MRI) set. The notebook finds them by their contents.

**Run 1:** keep `QUICK_TEST = True`, Run all (≈20 min) and check for `!!! run failed`.
**Run 2+:** set `QUICK_TEST = False`, Save Version → Save & Run All (Commit). To continue a stopped run,
attach this notebook's previous output as an input and commit again; finished runs are skipped.

| Step | What | Where it runs |
|---|---|---|
| 1 | Manifests, image caches, embeddings, central ground truth, **E1** audit accuracy, **E3** robustness | GPU 0 + CPU |
| 2 | **E4** leakage inflation, **E5** shortcut audit, **E6** robust training (two GPUs in parallel); **E2** cost scaling on CPU at the same time | 2 × T4 |
| 3 | Tables (ALL_TABLES.xlsx, SUMMARY.txt) and figures (600 dpi PNG + TIFF) | CPU |
| 4 | Zip `fedaudit_results_download.zip` | — |
"""

SETTINGS = """
# ============================== SETTINGS ==============================
QUICK_TEST = True          # True: ~20-minute pipeline check on small samples. False: the real experiments.
TIME_BUDGET_HOURS = 11.0   # stop starting new runs after this (Kaggle stops sessions at 12 h)
ROUNDS = 30                # federated rounds per run (local epochs = 1)
SEEDS = [0, 1, 2]
ROBUST_LAMBDA = 1.0        # Stage D consistency weight (λ)
HE_VERIFY = True           # real homomorphic-encryption verification in the final audit (E1 final, E2)
SOURCE_OVERRIDES = {}      # e.g. {"figshare": "/kaggle/input/my-figshare-copy"} if auto-detection misses one
# ======================================================================
"""

INSTALL = """
import base64, io, os, sys, tarfile, subprocess, time
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openmined.psi", "tenseal", "h5py"], check=False)
CODE_DIR = "/kaggle/working/fedaudit_code"
os.makedirs(CODE_DIR, exist_ok=True)
PKG = "__PKG__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG)), mode="r:gz").extractall(CODE_DIR, filter="data")
sys.path.insert(0, CODE_DIR)
for m in [m for m in sys.modules if m == "fedaudit" or m.startswith("fedaudit.")]:
    del sys.modules[m]  # re-running this cell after importing a new notebook loads the new code
import fedaudit, torch
print("fedaudit code ready |", "CUDA GPUs:", torch.cuda.device_count(), *[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
"""

STEP1 = """
# Step 1: data, Stage A audit experiments (E1, E3)
import json
from fedaudit.experiments.prepare import run_prepare
from fedaudit.experiments.runner import merge_previous

OUT = "/kaggle/working/fedaudit_results_quick" if QUICK_TEST else "/kaggle/working/fedaudit_results"  # quick test never mixes with real runs
S = dict(out=OUT, input_root="/kaggle/input", code_dir=CODE_DIR, source_overrides=SOURCE_OVERRIDES,
         quick=600 if QUICK_TEST else 0, size=128, rounds=2 if QUICK_TEST else ROUNDS,
         seeds=SEEDS[:1] if QUICK_TEST else SEEDS, he_verify=HE_VERIFY, robust_lam=ROBUST_LAMBDA,
         encoders=["resnet50", "pixel"] if QUICK_TEST else ["resnet50", "dino_vits16", "pixel"],
         band_grid=[(16, 16), (32, 8)] if QUICK_TEST else [(8, 16), (16, 12), (16, 16), (16, 20), (24, 12), (32, 12)],
         tau_grid=[0.9, 0.95] if QUICK_TEST else [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
         e2_grid=[(2, 500), (3, 500)] if QUICK_TEST else [(2, 1000), (2, 5000), (2, 20000), (2, 100000),
                                                         (3, 2000), (4, 2000), (5, 2000), (8, 2000), (10, 2000)],
         e3_n=60 if QUICK_TEST else 300, stagec_per_site=32 if QUICK_TEST else 128,
         deadline=START + TIME_BUDGET_HOURS * 3600)
os.makedirs(OUT, exist_ok=True)
print(f"Merged {merge_previous(S)} finished runs from previous versions")
run_prepare(S, skip_e2=True)
"""

STEP2 = """
# Step 2: training runs on both GPUs (E4, E5, E6) + E2 cost scaling on the CPU meanwhile
from fedaudit.experiments.runner import run_all
from fedaudit.experiments.prepare import e2_scaling
run_all(S, n_gpus=max(1, torch.cuda.device_count()), cpu_task=lambda: e2_scaling(S))
"""

STEP3 = """
# Step 3: tables and figures
from fedaudit.experiments.analyze import run_analysis
T = run_analysis(OUT)
print(open(f"{OUT}/analysis/SUMMARY.txt").read()[:6000])
"""

STEP4 = """
# Step 4: show a few figures and zip the results
import glob, shutil
from IPython.display import Image, display
for f in sorted(glob.glob(f"{OUT}/analysis/figures/*.png"))[:6]:
    print(os.path.basename(f)); display(Image(f, width=600))
shutil.rmtree(f"{OUT}/tmp", ignore_errors=True)
name = "fedaudit_results_download_quick" if QUICK_TEST else "fedaudit_results_download"
# image caches (store/) are rebuilt each session, so they are left out of the zip
import zipfile
with zipfile.ZipFile(f"/kaggle/working/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(p) and "/store/" not in p and "/claims/" not in p:
            z.write(p, os.path.join("fedaudit_results", os.path.relpath(p, OUT)))
shutil.rmtree(f"{OUT}/store", ignore_errors=True)
print("Wrote", f"/kaggle/working/{name}.zip", f"{os.path.getsize(f'/kaggle/working/{name}.zip') / 1e6:.1f} MB")
"""


def main():
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[md(INTRO), code(SETTINGS), code(INSTALL.replace("__PKG__", pkg_b64())), code(STEP1), code(STEP2),
                     code(STEP3), code(STEP4)])
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    p = os.path.join(ROOT, "notebooks", "FedAudit_Experiments.ipynb")
    json.dump(nb, open(p, "w"), indent=1)
    print("wrote", p, os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    main()
