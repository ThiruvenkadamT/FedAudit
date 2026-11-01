"""Find the attached public datasets under /kaggle/input and build FedAudit manifests.

Detection is by content, not by slug, because Kaggle mirrors are named differently:
  ham10000     a HAM10000_metadata*.csv file
  ctmri        folders mentioning CT and MRI with Healthy/Tumor classes
  figshare     *.mat files with cjdata, or a folder whose path mentions figshare
  nickparvar   Training/Testing with a 'notumor' class folder
  sartaj       Training/Testing with 'no_tumor' and 'glioma_tumor' class folders
  br35h        'yes' and 'no' class folders
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from ..data import manifest as mf

BRAIN4 = ["glioma", "meningioma", "notumor", "pituitary"]


def _dirs(root, depth=6):
    base = root.rstrip("/").count("/")
    for dp, dns, _ in os.walk(root):
        if dp.count("/") - base >= depth:
            dns[:] = []
        yield dp, [d.lower() for d in dns]


def detect_sources(root: str = "/kaggle/input", overrides: dict | None = None) -> dict:
    found = dict(overrides or {})
    if not os.path.isdir(root):
        return found
    for dp, dns in _dirs(root):
        low = dp.lower()
        if "ham10000" not in found and glob.glob(os.path.join(dp, "HAM10000_metadata*.csv")):
            found["ham10000"] = dp
        if "nickparvar" not in found and "notumor" in dns:
            found["nickparvar"] = os.path.dirname(dp) if os.path.basename(low) in ("training", "testing") else dp
        if "sartaj" not in found and "no_tumor" in dns and "glioma_tumor" in dns:
            found["sartaj"] = os.path.dirname(dp) if os.path.basename(low) in ("training", "testing") else dp
        if "br35h" not in found and "yes" in dns and "no" in dns:
            found["br35h"] = dp
        ct_dirs = [d for d in dns if "ct" in d.replace("-", " ").split() and "mri" not in d]
        mri_dirs = [d for d in dns if "mri" in d.replace("-", " ").split() and not d.replace("-", " ").split().count("ct")]
        if "ctmri" not in found and ct_dirs and mri_dirs:
            found["ctmri"] = dp
        if "figshare" not in found and ("figshare" in low or glob.glob(os.path.join(dp, "*.mat"))):
            if glob.glob(os.path.join(dp, "**", "*.mat"), recursive=True) or "figshare" in low:
                found["figshare"] = dp
    # normalize: nickparvar/sartaj roots are the folder holding Training/Testing
    for k in ("nickparvar", "sartaj"):
        if k in found:
            p = found[k]
            while os.path.basename(p).lower() in ("training", "testing", "train", "test"):
                p = os.path.dirname(p)
            found[k] = p
    return found


def _sample(d: pd.DataFrame, by, k: int, seed: int) -> pd.DataFrame:
    """Up to k random rows per group (pandas-version independent)."""
    return d.sample(frac=1.0, random_state=seed).groupby(by, sort=False).head(k).reset_index(drop=True)


def brain_mri_federation(src: dict, quick: int = 0, seed: int = 0) -> pd.DataFrame:
    """Each public source = one hospital. Br35H keeps its yes/no labels ('tumor' rows are audited
    but not used for 4-class training). Published train/test splits are kept; sources without one
    (figshare, Br35H) get a stratified image-level 80/20 split, as in common practice."""
    parts = []
    for name in ("figshare", "sartaj", "br35h", "nickparvar"):
        if name not in src:
            continue
        d = mf.figshare_cheng(src[name], name, limit=quick, seed=seed) if name == "figshare" else mf.folder_source(src[name], name)
        if quick:
            d = _sample(d, "label_name", quick // 4, seed)
        parts.append(d)
    if not parts:
        raise FileNotFoundError("No brain MRI source found (figshare, SARTAJ, Br35H, Nickparvar)")
    df = pd.concat(parts, ignore_index=True)
    df = mf.finalize(df, BRAIN4 + ["tumor"])
    df["image_id"] = [f"{s}_{i:06d}" for i, s in enumerate(df["site"])]
    df = mf.assign_splits(df, seed=seed, frac=(0.7, 0.1, 0.2), keep_existing=True)
    return df


def ham_federation(src: dict, by_lesion: bool, quick: int = 0, seed: int = 0, input_root: str = "/kaggle/input") -> pd.DataFrame:
    d = mf.ham10000(src["ham10000"], search_roots=[input_root], seed=seed)
    mode = d.attrs.get("sites", "natural")
    if mode == "simulated":
        print("[prepare] HAM10000: no source column in this copy's metadata -> 4 simulated sites drawn at lesion "
              "level (real source sizes). Attach the Harvard Dataverse metadata CSV to use the natural sources.", flush=True)
    if quick:
        d = _sample(d, "site", quick // 4, seed)
    df = mf.finalize(d, sorted(d["label_name"].unique()))
    df["image_id"] = [f"ham_{os.path.splitext(os.path.basename(p))[0]}" for p in df["path"]]
    df = mf.assign_splits(df, seed=seed, by_group=by_lesion, keep_existing=False)
    df.attrs["sites"] = mode
    return df


def ctmri_pool(src: dict, quick: int = 0, seed: int = 0) -> pd.DataFrame:
    d = mf.ct_mri_multimodal(src["ctmri"])
    if quick:
        d = _sample(d, ["modality", "label_name"], quick // 4, seed)
    df = mf.finalize(d, ["healthy", "tumor"])
    df["image_id"] = [f"ct_{i:06d}" for i in range(len(df))]
    return df


CT_PARTITIONS = {"iid": dict(alpha=None), "dir1.0": dict(alpha=1.0), "dir0.5": dict(alpha=0.5),
                 "dir0.1": dict(alpha=0.1), "modskew": dict(alpha=None, modality_skew=True)}


def ctmri_federation(pool: pd.DataFrame, partition: str, n_sites: int = 5, seed: int = 0) -> pd.DataFrame:
    df = mf.dirichlet_sites(pool, n_sites, seed=seed, **CT_PARTITIONS[partition])
    return mf.assign_splits(df, seed=seed, keep_existing=False)
