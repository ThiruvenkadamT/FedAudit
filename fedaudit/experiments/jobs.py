"""Training jobs (E4, E5, E6). Each job is a small dict; run_job() trains, evaluates and writes
one JSON (+ a compressed per-image file) into out/runs/. Jobs are independent so two GPUs can
work through the list and an interrupted Kaggle session can resume."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from ..audit.embed import embed_paths
from ..audit.stage_a import AuditParams, run_stage_a
from ..data import imaging as im
from ..data import manifest as mf
from ..fl.algorithms import FLConfig, run_federated
from ..fl.data import TensorSet
from ..fl.evaluate import evaluate
from ..shortcut.faithfulness import deletion_auc
from ..shortcut.stage_c import collect_site_stats
from . import sources
from .store import Store, fit

E4_ALGOS = ["fedavg", "fedprox", "scaffold", "fednova", "feddyn", "fedadam", "moon", "fedbn", "local", "central"]
LEAK_ALGOS = ["fedavg", "scaffold", "fedbn", "local"]
LEAK_PARTS = ["natural", "iid", "dir0.5"]
LEAK_RATES = [0.0, 0.01, 0.02, 0.05, 0.10]
CT_ALGOS = ["fedavg", "scaffold", "fedbn"]
CT_VARIANTS = ["none", "headcrop", "blanktrain", "featmask", "robust"]


def job_id(j: dict) -> str:
    return "_".join(f"{k}-{j[k]}" for k in sorted(j))


def all_jobs(S: dict, available: set) -> list[dict]:
    seeds = S.get("seeds", [0, 1, 2])
    J = []
    if "brain" in available:
        J += [dict(exp="E4brain", algo=a, seed=s) for s in seeds for a in E4_ALGOS]
    if "ham" in available:
        J += [dict(exp="E4ham", algo=a, seed=s, split=sp) for s in seeds for sp in ("image", "lesion") for a in E4_ALGOS]
    if "brain_dedup" in available:
        J += [dict(exp="E4leak", algo=a, part=p, rate=r, seed=s) for s in seeds for r in LEAK_RATES for p in LEAK_PARTS for a in LEAK_ALGOS]
    if "ct" in available:
        J += [dict(exp="E6", algo=a, part=p, variant=v, seed=s) for s in seeds for p in sources.CT_PARTITIONS for v in CT_VARIANTS for a in CT_ALGOS]
        J += [dict(exp="E5nat", algo="fedavg", part="iid", variant=v, seed=s) for s in seeds for v in ("none", "headcrop", "blanktrain", "robust")]
    if S.get("experiments"):
        J = [j for j in J if j["exp"] in S["experiments"]]
    if S.get("quick"):
        # one representative of each kind is enough to test the pipeline
        keep, seen = [], set()
        for j in J:
            key = (j["exp"], j.get("variant"), j.get("split"), j["algo"] if j["exp"] == "E4brain" else None)
            if key not in seen and j["seed"] == seeds[0]:
                seen.add(key)
                keep.append(j)
        J = keep
    return J


# ------------------------------------------------------------------ helpers
def _ts(store: Store, df: pd.DataFrame, size: int, device: str, fn=None, variants: dict | None = None) -> TensorSet:
    """fn(image_id, img) → img at store resolution; resized to `size`."""
    X, V = [], {k: [] for k in (variants or {})}
    for iid in df["image_id"]:
        img = store.get(iid)
        X.append(fit(fn(iid, img) if fn else img, size))
        for k, f in (variants or {}).items():
            V[k].append(fit(f(iid, img), size))

    def stack(L):
        a = np.stack(L) if L else np.zeros((0, size, size), np.uint8)
        a = a[:, None] if a.ndim == 3 else a.transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    return TensorSet(stack(X), torch.as_tensor(df["label"].values, dtype=torch.long, device=device), df["image_id"].values,
                     {k: stack(L) for k, L in V.items()} if variants else None)


def _cfg(S, algo, seed, in_ch, n_classes, device, **kw) -> FLConfig:
    return FLConfig(algo=algo, model=S.get("model", "cnn"), in_ch=in_ch, n_classes=n_classes, rounds=S["rounds"],
                    local_epochs=S.get("local_epochs", 1), batch=S.get("batch", 32), lr=S.get("lr", 0.01), seed=seed,
                    device=device, **kw)


def leaked_ids(pairs: pd.DataFrame, split: dict, site: dict | None = None, cross_only: bool = False) -> set:
    bad = set()
    for a, b in zip(pairs["id_a"], pairs["id_b"]):
        if a not in split or b not in split:
            continue
        if cross_only and site is not None and site[a] == site[b]:
            continue
        if split[a] in ("test", "val") and split[b] == "train":
            bad.add(a)
        if split[b] in ("test", "val") and split[a] == "train":
            bad.add(b)
    return bad


def _views_by_site(sets_fn, df, sites):
    return {s: sets_fn(df[df["site"] == s]) for s in sites}


def _save(S, j, metrics: pd.DataFrame, extra: dict, probs: dict | None = None):
    os.makedirs(f"{S['out']}/runs", exist_ok=True)
    jid = job_id(j)
    if probs:
        np.savez_compressed(f"{S['out']}/runs/{jid}.npz",
                            **{f"{v}|{s}|{k}": arr for (v, s), (ids, y, p) in probs.items()
                               for k, arr in (("ids", ids.astype(str)), ("y", y), ("p", p.astype(np.float16)))})
    with open(f"{S['out']}/runs/{jid}.json", "w") as f:
        json.dump(dict(job=j, metrics=metrics.to_dict("records"), **extra), f, default=float)


# ------------------------------------------------------------------ E4 natural: brain MRI sources, HAM10000
def run_e4_natural(S, j, device):
    out = S["out"]
    if j["exp"] == "E4brain":
        df = pd.read_csv(f"{out}/data/brain.csv").fillna({"group": ""})
        pairs = pd.read_csv(f"{out}/results/audit_pairs_brain.csv")
        df = df[df["label_name"].isin(sources.BRAIN4)].copy()
        df["label"] = df["label_name"].map({n: i for i, n in enumerate(sources.BRAIN4)})
        store, in_ch = Store(f"{out}/store/brain"), 1
    else:
        df = pd.read_csv(f"{out}/data/ham_{j['split']}.csv").fillna({"group": ""})
        pairs = pd.read_csv(f"{out}/results/audit_pairs_ham.csv")
        store, in_ch = Store(f"{out}/store/ham"), 3
    n_classes = int(df["label"].max()) + 1
    sites = sorted(df["site"].unique())
    split = dict(zip(df["image_id"], df["split"]))
    site = dict(zip(df["image_id"], df["site"]))
    bad = leaked_ids(pairs, split)
    bad_x = leaked_ids(pairs, split, site, cross_only=True)
    size = S["size"]
    mk = lambda d: _ts(store, d, size, device)
    train = {s: mk(df[(df.site == s) & (df.split == "train")]) for s in sites}
    test = df[df.split == "test"]
    views = {"unaudited": _views_by_site(mk, test, sites),
             "audited": _views_by_site(mk, test[~test.image_id.isin(bad)], sites),
             "audited_cross": _views_by_site(mk, test[~test.image_id.isin(bad_x)], sites),
             "leaked_only": _views_by_site(mk, test[test.image_id.isin(bad)], sites)}
    cfg = _cfg(S, j["algo"], j["seed"], in_ch, n_classes, device)
    t = time.time()
    res = run_federated(cfg, train)
    secs = time.time() - t
    m, probs = evaluate(res["personal"], views, cfg, keep_probs=True)
    probs = {k: v for k, v in probs.items() if k[0] == "unaudited"}
    _save(S, j, m, dict(seconds=secs, n_leaked_test=len(bad & set(test.image_id)),
                        n_leaked_test_cross=len(bad_x & set(test.image_id)), n_test=len(test)), probs)


# ------------------------------------------------------------------ E4 controlled leakage
def run_e4_leak(S, j, device):
    out = S["out"]
    pool = pd.read_csv(f"{out}/data/brain_dedup.csv").fillna({"group": ""})
    pool["label"] = pool["label_name"].map({n: i for i, n in enumerate(sources.BRAIN4)})
    store = Store(f"{out}/store/brain")
    seed = j["seed"]
    if j["part"] == "natural":
        fed = pool.copy()
        fed["split"] = ""
    else:
        fed = mf.dirichlet_sites(pool, 4, alpha=None if j["part"] == "iid" else 0.5, seed=seed)
    fed = mf.assign_splits(fed, seed=seed, keep_existing=False)
    sites = sorted(fed["site"].unique())
    rng = random.Random(1000 * seed + int(j["rate"] * 1000))
    copies, copy_img = [], {}
    for s in sites:
        te = fed[(fed.site == s) & (fed.split == "test")]
        k = int(round(j["rate"] * len(te)))
        for _, r in te.sample(k, random_state=rng.randint(0, 10 ** 6)).iterrows() if k else []:
            kind = "exact" if rng.random() < 0.5 else rng.choice(["resize", "jpeg", "crop", "overlay"])
            cid = f"copy_{len(copies)}_{kind}"
            copy_img[cid] = im.NEAR_COPY[kind](im.load_gray(r.path), rng)  # copy made from the original file
            copies.append(dict(r, image_id=cid, site=rng.choice([x for x in sites if x != s]), split="train",
                               path="", orig=r.image_id, kind=kind))
    fed["orig"], fed["kind"] = "", "orig"
    full = pd.concat([fed, pd.DataFrame(copies)], ignore_index=True) if copies else fed
    class _S:  # store view that also serves the injected copies
        size = store.size

        def get(self, iid):
            return fit(copy_img[iid], store.size) if iid in copy_img else store.get(iid)

    st = _S()
    # Stage A on the injected federation (simulated PSI: identical output, measured cost in E2)
    cfg_a = json.load(open(f"{out}/results/audit_config_brain.json"))
    z = np.load(f"{out}/emb/brain_{cfg_a['encoder']}.npz", allow_pickle=True)
    emb_map = dict(zip(z["ids"], z["emb"]))
    tmp = f"{out}/tmp/leak_{job_id(j)}"
    os.makedirs(tmp, exist_ok=True)
    cp_paths = []
    for cid in [c["image_id"] for c in copies]:
        p = f"{tmp}/{cid}.png"
        from PIL import Image
        Image.fromarray(copy_img[cid]).save(p)
        cp_paths.append(p)
    if cp_paths:
        ce = embed_paths(cp_paths, cfg_a["encoder"], crop=True)
        emb_map.update(dict(zip([c["image_id"] for c in copies], ce)))
    full_a = full.copy()
    full_a.loc[full_a.image_id.isin(copy_img), "path"] = cp_paths
    E = np.stack([emb_map[i] for i in full_a.image_id])
    A = run_stage_a(full_a, E, AuditParams(n_bands=cfg_a["n_bands"], band_width=cfg_a["band_width"], tau=cfg_a["tau"],
                                           psi="simulate", exact=True, within_site=True))
    split = dict(zip(full.image_id, full.split))
    bad = leaked_ids(A.pairs, split)
    truth_leaked = set(c["orig"] for c in copies)
    recall = len(bad & truth_leaked) / len(truth_leaked) if truth_leaked else np.nan
    size = S["size"]
    mk = lambda d: _ts(st, d, size, device)
    train = {s: mk(full[(full.site == s) & (full.split == "train")]) for s in sites}
    test = full[full.split == "test"]
    views = {"unaudited": _views_by_site(mk, test, sites),
             "audited": _views_by_site(mk, test[~test.image_id.isin(bad)], sites),
             "truth_clean": _views_by_site(mk, test[~test.image_id.isin(truth_leaked)], sites),
             "leaked_only": _views_by_site(mk, test[test.image_id.isin(truth_leaked)], sites)}
    cfg = _cfg(S, j["algo"], seed, 1, 4, device)
    t = time.time()
    res = run_federated(cfg, train)
    m = evaluate(res["personal"], views, cfg)
    _save(S, j, m, dict(seconds=time.time() - t, n_copies=len(copies), n_test=len(test), audit_recall=recall,
                        audit_flagged=len(bad & set(test.image_id)),
                        audit_precision=(len(bad & truth_leaked) / len(bad & set(test.image_id))) if bad & set(test.image_id) else np.nan))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ E5/E6 shortcut audit and robust training on CT/MRI
PLANT_SITES = ("site0", "site1")


class CTData:
    """CT/MRI images at 256 px with precomputed head/text masks, plus planted markers."""

    def __init__(self, out, fed: pd.DataFrame, planted: bool, seed: int, corr: float = 0.9):
        self.st = Store(f"{out}/store/ct")
        self.fed = fed
        self.rng = random.Random(seed)
        self.planted = {}
        if planted:
            for r in fed.itertuples():
                if r.site in PLANT_SITES:
                    lab = r.label if self.rng.random() < corr else 1 - r.label
                    self.planted[r.image_id] = lab

    def base(self, iid):
        img = self.st.get(iid)
        h, t = self.st.masks(iid)
        if iid in self.planted:
            mk = im.plant_marker(img, self.planted[iid], 2)
            t = t | (mk != img)
            t = np.asarray(__import__("cv2").dilate(t.astype(np.uint8), np.ones((5, 5), np.uint8)), bool)
            img = mk
        return img, h, t


def _headcrop(img, h):
    x0, y0, x1, y1 = im.head_bbox(h)
    return img[y0:y1, x0:x1]


def run_ct(S, j, device):
    out = S["out"]
    seed = j["seed"]
    pool = pd.read_csv(f"{out}/data/ct_pool.csv").fillna({"group": ""})
    fed = sources.ctmri_federation(pool, j["part"], 5, seed)
    D = CTData(out, fed, planted=(j["exp"] == "E6"), seed=seed)
    sites = sorted(fed["site"].unique())
    size = S["size"]
    v = j["variant"]
    cache = {}

    def B(iid):
        if iid not in cache:
            cache[iid] = D.base(iid)
        return cache[iid]

    post = (lambda img, iid: _headcrop(img, B(iid)[1])) if v == "headcrop" else (lambda img, iid: img)
    f_std = lambda iid, _: post(B(iid)[0], iid)
    f_clean = lambda iid, _: post(np.where(B(iid)[2], 0, B(iid)[0]).astype(np.uint8), iid)
    # swap bank per site: (text mask, image) of training images by class
    banks = {}
    for s in sites:
        bk = {0: [], 1: []}
        for r in fed[(fed.site == s) & (fed.split == "train")].itertuples():
            img, h, t = B(r.image_id)
            if t.mean() > 0.002 and len(bk[r.label]) < 100:
                bk[r.label].append((t, img))
        banks[s] = bk
    srng = random.Random(seed + 7)
    meta = {r.image_id: (r.site, r.label) for r in fed.itertuples()}

    def f_swap(iid, _):
        img, h, t = B(iid)
        r_site, r_label = meta[iid]
        other = banks[r_site][1 - r_label] or [b for s2 in sites for b in banks[s2][1 - r_label]]
        o = np.where(t, 0, img).astype(np.uint8)
        if other:
            tt, src = srng.choice(other)
            o[tt] = np.maximum(o[tt], src[tt])
        else:
            o = im.plant_marker(o, 1 - r_label, 2)
        return post(o, iid)

    trng = random.Random(seed + 11)
    f_train = f_clean if v == "blanktrain" else f_std
    variants = None
    if v == "robust":
        variants = {"blank": f_clean, "rt1": lambda iid, _: im.paste_random_text(B(iid)[0], trng),
                    "rt2": lambda iid, _: im.paste_random_text(B(iid)[0], trng)}
    train = {s: _ts(D.st, fed[(fed.site == s) & (fed.split == "train")], size, device, f_train, variants) for s in sites}
    test = fed[fed.split == "test"]
    views = {}
    for name, f in (("standard", f_std), ("clean", f_clean), ("swap", f_swap)):
        views[name] = {s: _ts(D.st, test[test.site == s], size, device, f) for s in sites}
    for mod in ("CT", "MRI"):
        for name, f in (("standard", f_std), ("clean", f_clean), ("swap", f_swap)):
            views[f"{name}_{mod}"] = {s: _ts(D.st, test[(test.site == s) & (test.modality == mod)], size, device, f) for s in sites}
    nuis = None
    kw = {}
    if v == "featmask":
        kw["featmask"] = 0.1
        nuis = {s: np.array([float(B(i)[2].mean() > 0.002) for i in fed[(fed.site == s) & (fed.split == "train")].image_id]) for s in sites}
    if v == "robust":
        kw["robust"] = dict(lam=S.get("robust_lam", 1.0), p=0.5)
    cfg = _cfg(S, j["algo"], seed, 1, 2, device, **kw)
    t = time.time()
    res = run_federated(cfg, train, nuisance=nuis)
    secs = time.time() - t
    m = evaluate(res["personal"], views, cfg)
    # faithfulness (deletion test) on the standard test set, per site model
    dl = [deletion_auc(res["personal"][s], views["standard"][s], device, n_images=32, seed=seed) for s in sites if len(views["standard"][s])]
    deletion = {k: float(np.mean([d[k] for d in dl])) for k in dl[0]} if dl else {}
    # Stage C per-image statistics (stay 'local'; DP/secure aggregation applied in analysis for each ε)
    site_data = {}
    for s in sites:
        rows = test[test.site == s]
        imgs = [B(i)[0] for i in rows.image_id]
        bank = _BankAdapter(banks[s])
        site_data[s] = (imgs, rows["label"].tolist(), bank)
    post_c = (lambda a, y, rng: im.head_crop(a)) if v == "headcrop" else None
    stats = collect_site_stats(res["personal"], site_data, 2, device, size, per_site=S.get("stagec_per_site", 128),
                               seed=seed, post=post_c)
    sdir = f"{out}/runs/stagec"
    os.makedirs(sdir, exist_ok=True)
    pd.concat([d.assign(site=s) for s, d in stats.items()]).to_csv(f"{sdir}/{job_id(j)}.csv.gz", index=False)
    _save(S, j, m, dict(seconds=secs, deletion=deletion,
                        text_rate={s: float(np.mean([B(i)[2].mean() > 0.002 for i in fed[fed.site == s].image_id])) for s in sites}))


class _BankAdapter:
    """Adapter so Stage C's swap statistic uses the same class-typical text bank as the swap test."""

    def __init__(self, bk):
        self.bk = bk

    def paste(self, img, y, rng, n_classes):
        other = self.bk.get(1 - y) or []
        out = im.blank_text(img)
        if not other:
            return im.plant_marker(out, 1 - y, 2)
        t, src = rng.choice(other)
        out[t] = np.maximum(out[t], src[t])
        return out


def run_job(S: dict, j: dict, device: str):
    torch.backends.cudnn.benchmark = True
    if j["exp"] in ("E4brain", "E4ham"):
        return run_e4_natural(S, j, device)
    if j["exp"] == "E4leak":
        return run_e4_leak(S, j, device)
    if j["exp"] in ("E6", "E5nat"):
        return run_ct(S, j, device)
    raise ValueError(j)
