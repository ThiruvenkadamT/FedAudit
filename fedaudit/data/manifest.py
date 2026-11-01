"""Manifests: one CSV row per image, the common currency of every FedAudit stage.

Columns
-------
image_id  unique string id
path      absolute or relative path to the image file
label     integer class index
label_name
site      hospital / source name (one federated client)
split     train | val | test
group     patient id or lesion id when known ('' otherwise) — used as ground truth
modality  'CT', 'MRI', 'DERM' or ''

Builders below turn the public datasets' folder layouts into manifests. They glob
rather than assume exact paths, because Kaggle mirrors of the same dataset differ.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
COLUMNS = ["image_id", "path", "label", "label_name", "site", "split", "group", "modality"]


def _images_under(root: str):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return sorted(out)


def _norm_label(name: str) -> str | None:
    n = name.lower()
    if "glioma" in n:
        return "glioma"
    if "mening" in n:
        return "meningioma"
    if "pituitary" in n:
        return "pituitary"
    if n in ("no", "notumor", "no_tumor", "no tumor", "normal", "healthy") or "no_tumor" in n or "notumor" in n or "healthy" in n:
        return "notumor"
    if n in ("yes", "tumor", "tumour"):
        return "tumor"
    return None


def _label_from_path(p: str) -> str | None:
    for part in reversed(p.replace("\\", "/").split("/")[:-1]):
        lab = _norm_label(part)
        if lab is not None:
            return lab
    return None


def _split_from_path(p: str) -> str | None:
    s = p.replace("\\", "/").lower()
    if re.search(r"/(testing|test)/", s):
        return "test"
    if re.search(r"/(training|train)/", s):
        return "train"
    if re.search(r"/(val|valid|validation)/", s):
        return "val"
    return None


def folder_source(root: str, site: str, modality: str = "MRI", group_fn=None) -> pd.DataFrame:
    """Generic class-per-folder source (SARTAJ, Br35H, Nickparvar, figshare-jpg mirrors)."""
    rows = []
    for p in _images_under(root):
        lab = _label_from_path(p)
        if lab is None:
            continue
        rows.append(dict(path=p, label_name=lab, site=site, split=_split_from_path(p) or "",
                         group=group_fn(p) if group_fn else "", modality=modality))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(f"No labelled images found under {root}")
    return df


def _read_mat(m: str, cache: str):
    import h5py
    from PIL import Image
    names = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    try:
        with h5py.File(m, "r") as f:
            d = f["cjdata"]
            lab = int(np.array(d["label"]).squeeze())
            pid = "".join(chr(int(c)) for c in np.array(d["PID"]).squeeze())
            out = os.path.join(cache, os.path.basename(m).replace(".mat", ".png"))
            if not os.path.exists(out):
                # h5py returns MATLAB (column-major) arrays transposed: .T restores the slice orientation
                img = np.array(d["image"]).astype(np.float32).T
                lo, hi = np.percentile(img, [0.5, 99.5])
                u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                Image.fromarray(u8).save(out)
    except Exception:
        return None
    return dict(path=out, label_name=names[lab], group=f"fs_{pid}")


def figshare_cheng(root: str, site: str = "figshare", limit: int = 0, seed: int = 0) -> pd.DataFrame:
    """Cheng et al. figshare set. Works on the original .mat release (needs h5py) or a jpg mirror.

    The .mat files carry cjdata.PID (patient id) and cjdata.label (1 menin, 2 glioma, 3 pituitary).
    Slices are exported once to PNG in a cache folder (in parallel; reading 3,064 .mat files from
    /kaggle/input one by one takes ~20 min). limit>0 reads only a random subset (quick test).
    """
    mats = sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True))
    if not mats:
        return folder_source(root, site)
    from concurrent.futures import ThreadPoolExecutor
    if limit and len(mats) > limit:
        mats = list(np.random.default_rng(seed).choice(mats, limit, replace=False))
    cache = os.path.join(os.environ.get("FEDAUDIT_CACHE", "/kaggle/working/cache"), "figshare_png_v2")
    os.makedirs(cache, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        recs = [r for r in ex.map(lambda m: _read_mat(m, cache), mats) if r]
    df = pd.DataFrame(recs)
    df["site"], df["split"], df["modality"] = site, "", "MRI"
    return df


HAM_SOURCES = {"vidir_modern": "vidir_current", "rosendahl": "rosendahl",
               "vidir_molemax": "molemax", "vienna_dias": "vidir_legacy"}
# image counts per source in the original release, used only for the fallback below
HAM_SOURCE_SIZES = {"rosendahl": 2259, "vidir_legacy": 439, "vidir_current": 3363, "molemax": 3954}


def _ham_source_column(search_roots) -> pd.DataFrame | None:
    """A HAM10000 metadata CSV that has the 'dataset' (source) column, e.g. the Harvard Dataverse
    release uploaded as a Kaggle dataset. The Kaggle mirror's metadata lacks it."""
    for r in search_roots:
        for c in glob.glob(os.path.join(r, "**", "*.csv"), recursive=True):
            if "metadata" not in os.path.basename(c).lower():
                continue
            try:
                head = pd.read_csv(c, nrows=2)
            except Exception:
                continue
            if {"image_id", "dataset"} <= set(head.columns):
                return pd.read_csv(c, usecols=["image_id", "dataset"])
    return None


def ham10000(root: str, search_roots=(), seed: int = 0) -> pd.DataFrame:
    """HAM10000 with lesion_id as group. Sites = the four natural sources when a metadata file
    with the 'dataset' column is available; otherwise four simulated sites drawn at lesion level
    (all images of a lesion stay at one site) with the original sources' sizes. df.attrs['sites']
    says which was used."""
    meta = glob.glob(os.path.join(root, "**", "HAM10000_metadata*.csv"), recursive=True)
    if not meta:
        raise FileNotFoundError("HAM10000_metadata.csv not found")
    md = pd.read_csv(meta[0])
    paths = {}
    for p in _images_under(root):  # the Kaggle mirror holds each image twice (two folder casings)
        paths.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    md = md[md["image_id"].isin(paths)].copy()
    mode = "natural"
    if "dataset" not in md.columns:
        src = _ham_source_column([root, *search_roots])
        if src is not None:
            md = md.merge(src, on="image_id", how="left")
    if "dataset" in md.columns and md["dataset"].notna().all():
        site = md["dataset"].map(lambda s: HAM_SOURCES.get(s, s)).values
    else:
        mode = "simulated"
        rng = np.random.default_rng(seed)
        les = md["lesion_id"].unique()
        names = list(HAM_SOURCE_SIZES)
        w = np.array([HAM_SOURCE_SIZES[n] for n in names], float)
        pick = dict(zip(les, rng.choice(len(names), len(les), p=w / w.sum())))
        site = [f"sim_{names[pick[l]]}" for l in md["lesion_id"]]
    df = pd.DataFrame(dict(
        path=md["image_id"].map(paths).values,
        label_name=md["dx"].values,
        site=site,
        split="",
        group=("lesion_" + md["lesion_id"].astype(str)).values,
        modality="DERM",
    ))
    df.attrs["sites"] = mode
    return df


def ct_mri_multimodal(root: str) -> pd.DataFrame:
    """'Brain Tumor Multimodal Image (CT & MRI)' Kaggle set: Brain Tumor CT/ MRI × Healthy/Tumor folders."""
    rows = []
    for p in _images_under(root):
        s = p.replace("\\", "/").lower()
        mod = "CT" if "/ct" in s or "ct scan" in s or "_ct" in s else ("MRI" if "mri" in s else "")
        if "healthy" in s or "/no" in s or "normal" in s:
            lab = "healthy"
        elif "tumor" in s or "tumour" in s:
            lab = "tumor"
        else:
            continue
        rows.append(dict(path=p, label_name=lab, site="", split="", group="", modality=mod))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(f"No CT/MRI images under {root}")
    return df


def finalize(df: pd.DataFrame, label_order: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if label_order is None:
        label_order = sorted(df["label_name"].unique())
    df = df[df["label_name"].isin(label_order)]
    df["label"] = df["label_name"].map({n: i for i, n in enumerate(label_order)}).astype(int)
    if "image_id" not in df or df["image_id"].isna().any():
        df["image_id"] = [f"{s}_{i:06d}" for i, s in enumerate(df["site"])]
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df["group"] = df["group"].fillna("").astype(str)
    return df[COLUMNS].reset_index(drop=True)


def assign_splits(df: pd.DataFrame, seed: int = 0, frac=(0.7, 0.1, 0.2), by_group: bool = False,
                  keep_existing: bool = True) -> pd.DataFrame:
    """Per-site stratified split. by_group=True keeps all images of a patient/lesion in one split
    (the 'clean' split); by_group=False is the image-level split that can leak."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    out = []
    for site, d in df.groupby("site"):
        d = d.copy()
        has = keep_existing and d["split"].isin(["train", "test", "val"]).all()
        if has:
            # carve a validation set out of the published train split
            tr = d.index[d["split"] == "train"].to_numpy()
            if (d["split"] == "val").sum() == 0 and len(tr):
                n_val = int(round(len(tr) * frac[1] / (frac[0] + frac[1])))
                d.loc[rng.choice(tr, n_val, replace=False), "split"] = "val"
            out.append(d)
            continue
        if by_group and (d["group"] != "").all():
            keys = list(d["group"].unique())
            rng.shuffle(keys)
            n = len(keys)
            a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
            sp = {k: ("train" if i < a else "val" if i < b else "test") for i, k in enumerate(keys)}
            d["split"] = d["group"].map(sp)
        else:
            d["split"] = ""
            for _, dl in d.groupby("label"):
                idx = dl.index.to_numpy().copy()
                rng.shuffle(idx)
                n = len(idx)
                a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
                d.loc[idx[:a], "split"] = "train"
                d.loc[idx[a:b], "split"] = "val"
                d.loc[idx[b:], "split"] = "test"
        out.append(d)
    return pd.concat(out).sort_index()


def dirichlet_sites(df: pd.DataFrame, n_sites: int, alpha: float | None, seed: int = 0,
                    modality_skew: bool = False) -> pd.DataFrame:
    """Simulated hospitals for a single-source dataset (CT/MRI set). alpha=None → IID.
    modality_skew → half the sites CT-heavy (80/20), half MRI-heavy."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    site = np.empty(len(df), dtype=object)
    if modality_skew:
        for mod in df["modality"].unique():
            idx = np.where(df["modality"].values == mod)[0]
            rng.shuffle(idx)
            heavy = [s for s in range(n_sites) if (s % 2 == 0) == (mod == "CT")]
            p = np.array([0.8 / len(heavy) if s in heavy else 0.2 / (n_sites - len(heavy)) for s in range(n_sites)])
            site[idx] = rng.choice(n_sites, len(idx), p=p)
    else:
        for lab in df["label"].unique():
            idx = np.where(df["label"].values == lab)[0]
            rng.shuffle(idx)
            if alpha is None:
                parts = np.array_split(idx, n_sites)
            else:
                p = rng.dirichlet([alpha] * n_sites)
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                parts = np.split(idx, cuts)
            for s, part in enumerate(parts):
                site[part] = s
    df["site"] = [f"site{int(s)}" for s in site]
    return df


@dataclass
class Federation:
    df: pd.DataFrame

    @property
    def sites(self):
        return sorted(self.df["site"].unique())

    def part(self, site: str | None = None, split: str | None = None) -> pd.DataFrame:
        d = self.df
        if site is not None:
            d = d[d["site"] == site]
        if split is not None:
            d = d[d["split"] == split]
        return d
