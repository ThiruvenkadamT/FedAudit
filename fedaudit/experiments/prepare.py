"""Step 1 (one GPU, mostly CPU): manifests, image caches, embeddings, ground truth and the
audit experiments E1–E3, plus E2 cost scaling. Writes everything the training jobs need."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

from ..audit import ground_truth as gt
from ..audit.embed import embed_manifest, embed_paths
from ..audit.lsh import collision_prob
from ..audit.psi_protocol import CostLog, mutual_psi
from ..audit.stage_a import AuditParams, CandidateOverflow, run_stage_a
from ..audit.verify import HEVerifier
from ..data.imaging import NEAR_COPY
from . import sources
from .store import build_store


def greedy_keep(n: int, pairs) -> np.ndarray:
    """Drop the later image of each duplicate pair whose earlier image is kept (no transitive
    chaining: a run of similar neighboring slices is not collapsed into one image)."""
    nb = {}
    for a, b in pairs:
        a, b = min(a, b), max(a, b)
        nb.setdefault(a, []).append(b)
    keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            for j in nb.get(i, ()):
                keep[j] = False
    return keep


def union_find_keep(n: int, pairs) -> np.ndarray:
    """Keep one image per connected group of duplicates; returns boolean keep mask."""
    parent = list(range(n))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([f(i) == i for i in range(n)])


def _log(msg):
    print(f"[prepare {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_data(S: dict) -> dict:
    out = S["out"]
    os.makedirs(f"{out}/data", exist_ok=True)
    src = sources.detect_sources(S["input_root"], S.get("source_overrides"))
    _log(f"sources found: {json.dumps(src, indent=1)}")
    report = dict(sources=src)
    q = S.get("quick", 0)
    if any(k in src for k in ("figshare", "sartaj", "br35h", "nickparvar")):
        b = sources.brain_mri_federation(src, quick=q, seed=0)
        b.to_csv(f"{out}/data/brain.csv", index=False)
        build_store(b, f"{out}/store/brain", S["size"], rgb=False)
        report["brain"] = b.groupby(["site", "split"]).size().unstack(fill_value=0).to_dict()
        _log(f"brain MRI federation: {len(b)} images\n{b.groupby(['site', 'split']).size().unstack(fill_value=0)}")
    if "ham10000" in src:
        h_img = sources.ham_federation(src, by_lesion=False, quick=q, seed=0, input_root=S["input_root"])
        h_les = sources.ham_federation(src, by_lesion=True, quick=q, seed=0, input_root=S["input_root"])
        report["ham_sites"] = h_img.attrs.get("sites", "natural")
        h_img.to_csv(f"{out}/data/ham_image.csv", index=False)
        h_les.to_csv(f"{out}/data/ham_lesion.csv", index=False)
        build_store(h_img, f"{out}/store/ham", S["size"], rgb=True)
        report["ham"] = h_img.groupby("site").size().to_dict()
        _log(f"HAM10000: {len(h_img)} images, sites {report['ham']}")
    if "ctmri" in src:
        ct = sources.ctmri_pool(src, quick=q, seed=0)
        report["ct_before_dedup"] = {f"{a}/{b}": int(v) for (a, b), v in ct.groupby(["modality", "label_name"]).size().items()}
        C = gt.central_truth(ct, cache=f"{out}/data/ct_truth.pkl")
        # copies only: exact pixels or NCC ≥ 0.99 (0.98 also catches neighboring slices of one scan)
        V = gt.ncc_vectors(ct["path"].tolist())
        keep = greedy_keep(len(ct), C["exact"] | gt.pairs_above(V, 0.99))
        ct = ct[keep].reset_index(drop=True)
        report["ct_after_dedup"] = {f"{a}/{b}": int(v) for (a, b), v in ct.groupby(["modality", "label_name"]).size().items()}
        ct.to_csv(f"{out}/data/ct_pool.csv", index=False)
        build_store(ct, f"{out}/store/ct", 256, rgb=False, masks=True)
        _log(f"CT/MRI pool: before {report['ct_before_dedup']} after dedup {report['ct_after_dedup']}")
    with open(f"{out}/dataset_report.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    return src


# ------------------------------------------------------------------ E1: accuracy of the private audit
def _truth_cross(df, C):
    site = df["site"].values
    return {k: gt.restrict(v, site, True) for k, v in C.items()}


def _within_site_pairs(df: pd.DataFrame, emb: np.ndarray, min_tau: float, max_pairs: int = 3_000_000):
    """Each site's own local check, computed once per encoder on centered embeddings:
    arrays (i, j, cos) for cos > min_tau. Memory-bounded: if a site would return more than
    max_pairs, its floor is raised (such low thresholds are useless anyway)."""
    from ..audit.stage_a import center_embeddings
    site = df["site"].values
    emb = center_embeddings(emb, emb.mean(0))
    I, J, Cs = [], [], []
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch, dev = None, "cpu"
    for s in np.unique(site):
        ii = np.where(site == s)[0]
        E = emb[ii]
        Et = torch.from_numpy(E).to(dev) if torch is not None else None
        floor = min_tau
        while True:
            bi, bj, bc, total = [], [], [], 0
            for k0 in range(0, len(ii), 2048):
                Sm = (Et[k0:k0 + 2048] @ Et.T).cpu().numpy() if Et is not None else E[k0:k0 + 2048] @ E.T
                r, c = np.where(Sm > floor)
                m = (r + k0) < c
                r, c = r[m], c[m]
                total += len(r)
                if total > max_pairs:
                    break
                bi.append(ii[r + k0]); bj.append(ii[c]); bc.append(Sm[r, c])
            if total <= max_pairs:
                break
            floor = round(floor + 0.02, 4)
            _log(f"within-site pairs at site {s} exceed {max_pairs:,}: raising floor to {floor}")
        if bi:
            I.append(np.concatenate(bi)); J.append(np.concatenate(bj)); Cs.append(np.concatenate(bc))
        del Et
    if not I:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32)
    return np.concatenate(I), np.concatenate(J), np.concatenate(Cs)


def e1_sweep(S: dict, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Scores every configuration twice: cross-site pairs only (the privacy-relevant part, done
    with PSI) and all pairs (cross-site + each site's own local check), each against the central
    ground truth; group_recall = share of same-patient/same-lesion pairs found (all pairs).
    Image hashes and the within-site check are computed once and reused across the grid."""
    from ..audit.stage_a import _sha_all
    out = S["out"]
    C = gt.central_truth(df, cache=f"{out}/data/{name}_truth.pkl")
    site = df["site"].values
    T = _truth_cross(df, C)
    truth, truth_all = T["truth"], C["truth"]
    grp = gt.group_pairs(df)
    sha = _sha_all(df["path"])
    exact_within = {p for p in gt.exact_pairs_from_hashes(sha) if site[p[0]] == site[p[1]]}
    rows = []
    for base in ("exact", "ncc", "dhash"):
        sc_all = gt.score(C[base], truth_all)
        rows.append(dict(dataset=name, method=f"central_{base}", **gt.score(T[base], truth),
                         **{f"all_{k}": v for k, v in sc_all.items()}))
    for enc in S["encoders"]:
        try:
            emb = embed_manifest(df, enc, crop=True, cache=f"{out}/emb/{name}_{enc}.npz")
        except Exception as e:  # e.g. no internet for pretrained weights
            _log(f"encoder {enc} unavailable: {e}")
            continue
        within = _within_site_pairs(df, emb, min(S["tau_grid"]))
        wi, wj, wc = within
        for nb, bw in S["band_grid"]:
            # one Stage A run per band layout keeps every candidate with its cosine; each τ is a filter
            P = AuditParams(n_bands=nb, band_width=bw, tau=None, psi="simulate", within_site=False)
            try:
                r = run_stage_a(df, emb, P, sha=sha)
            except CandidateOverflow as e:
                _log(f"E1 {name} {enc} {nb}x{bw}: skipped, too many candidates ({e})")
                rows.append(dict(dataset=name, method="fedaudit", encoder=enc, n_bands=nb, band_width=bw,
                                 tau=np.nan, candidates=np.nan, note="candidate overflow"))
                continue
            cp = r.pairs
            for tau in S["tau_grid"]:
                sel = cp[(cp["kind"] == "exact") | (cp["cos"] > tau)]
                pred = gt.pairs_from_audit(sel)
                m = wc > tau
                pred_all = pred | exact_within | set(zip(wi[m].tolist(), wj[m].tolist()))
                sc, sc_all = gt.score(pred, truth), gt.score(pred_all, truth_all)
                rows.append(dict(dataset=name, method="fedaudit", encoder=enc, n_bands=nb, band_width=bw, tau=tau,
                                 candidates=r.n_candidates, group_recall=(len(pred_all & grp) / len(grp)) if grp else np.nan,
                                 **sc, **{f"all_{k}": v for k, v in sc_all.items()}))
            del cp, r
        _log(f"E1 {name} {enc}: done")
        pd.DataFrame(rows).to_csv(f"{out}/results/E1_sweep_{name}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/results/E1_sweep_{name}.csv", index=False)
    return res


def choose_config(sweep: pd.DataFrame, min_precision: float = 0.95, min_cross_truth: int = 30) -> dict:
    """Highest recall with precision ≥ min_precision, ties → fewer candidates (cheaper).
    Uses cross-site scores, or all-pair scores when there are too few true cross-site pairs
    to choose from (e.g. HAM10000 with lesion-level simulated sites)."""
    f = sweep[(sweep["method"] == "fedaudit") & sweep["tau"].notna()].copy()
    pre = "" if f["n_true"].max() >= min_cross_truth or "all_recall" not in f else "all_"
    ok = f[f[f"{pre}precision"] >= min_precision]
    if len(ok):
        # within 0.01 recall of the best, take the cheapest (fewest candidates to verify under encryption)
        best = ok[f"{pre}recall"].max()
        near = ok[ok[f"{pre}recall"] >= best - 0.01]
        pick = near.sort_values(["candidates", f"{pre}f1"], ascending=[True, False]).iloc[0]
    else:  # no configuration reaches the precision target: highest F1
        pick = f.sort_values([f"{pre}f1", "candidates"], ascending=[False, True]).iloc[0]
    return dict(encoder=pick["encoder"], n_bands=int(pick["n_bands"]), band_width=int(pick["band_width"]), tau=float(pick["tau"]))


def e1_final(S: dict, name: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Chosen configuration with the real protocol (ECDH PSI + HE verification): checks that
    results equal the simulated sweep and records the real cost. Saves the audit pairs."""
    out = S["out"]
    emb = embed_manifest(df, cfg["encoder"], crop=True, cache=f"{out}/emb/{name}_{cfg['encoder']}.npz")
    P = AuditParams(n_bands=cfg["n_bands"], band_width=cfg["band_width"], tau=cfg["tau"], psi="real",
                    verify="he" if S.get("he_verify", True) else "plain", within_site=True)
    t = time.time()
    r = run_stage_a(df, emb, P, verbose=True)
    r.pairs.to_csv(f"{out}/results/audit_pairs_{name}.csv", index=False)
    pd.DataFrame(r.per_pair_cost).to_csv(f"{out}/results/E2_sitepair_cost_{name}.csv", index=False)
    C = gt.central_truth(df, cache=f"{out}/data/{name}_truth.pkl")
    T = _truth_cross(df, C)
    pred_all = gt.pairs_from_audit(r.pairs)
    pred = gt.restrict(pred_all, df["site"].values, True)
    sc = gt.score(pred, T["truth"])
    sc_all = gt.score(pred_all, C["truth"])
    grp = gt.group_pairs(df)
    pc = pd.DataFrame(r.per_pair_cost)
    est_extra_s = float(pc.get("he_est_seconds", pd.Series(dtype=float)).fillna(0).sum())
    est_extra_b = float(pc.get("he_est_bytes", pd.Series(dtype=float)).fillna(0).sum())
    res = pd.DataFrame([dict(dataset=name, **cfg, protocol="real", seconds=time.time() - t, mb=r.cost.bytes / 1e6,
                             he_sampled_pairs=int(pc["he_sampled"].notna().sum()) if "he_sampled" in pc else 0,
                             est_total_seconds_full_he=time.time() - t + est_extra_s,
                             est_total_mb_full_he=(r.cost.bytes + est_extra_b) / 1e6,
                             candidates=r.n_candidates, n_pairs_total=len(r.pairs),
                             n_cross=int((r.pairs.site_a != r.pairs.site_b).sum()), **sc,
                             **{f"all_{k}": v for k, v in sc_all.items()},
                             group_recall=(len(pred_all & grp) / len(grp)) if grp else np.nan)])
    res.to_csv(f"{out}/results/E1_final_{name}.csv", index=False)
    with open(f"{out}/results/audit_config_{name}.json", "w") as f:
        json.dump(cfg, f)
    _log(f"E1 final {name}: {res.iloc[0].to_dict()}")
    return res


# ------------------------------------------------------------------ E3: robustness to near-copy transformations
def e3_robustness(S: dict, name: str, df: pd.DataFrame, cfg: dict, n: int = 300, seed: int = 0) -> pd.DataFrame:
    out = S["out"]
    rng = random.Random(seed)
    tmp = f"{out}/tmp/e3_{name}"
    os.makedirs(tmp, exist_ok=True)
    base = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    rows = []
    for kind, f in NEAR_COPY.items():
        recs = []
        for i, r in base.iterrows():
            img = np.asarray(Image.open(r["path"]).convert("RGB" if name == "ham" else "L"))
            cp = f(img, rng)
            p = f"{tmp}/{kind}_{i}.png"
            Image.fromarray(cp).save(p)
            recs.append(dict(r, path=p, site="copysite", image_id=f"cp_{kind}_{i}"))
        both = pd.concat([base.assign(site="origsite"), pd.DataFrame(recs)], ignore_index=True)
        emb = embed_paths(both["path"], cfg["encoder"], crop=True)
        P = AuditParams(n_bands=cfg["n_bands"], band_width=cfg["band_width"], tau=cfg["tau"], psi="simulate", within_site=False)
        r = run_stage_a(both, emb, P)
        truth = {(i, i + len(base)) for i in range(len(base))}
        pred = gt.pairs_from_audit(r.pairs)
        rows.append(dict(dataset=name, transform=kind, method="fedaudit", recall=len(pred & truth) / len(truth)))
        C = gt.central_truth(both)
        for b in ("exact", "ncc", "dhash"):
            rows.append(dict(dataset=name, transform=kind, method=f"central_{b}", recall=len(C[b] & truth) / len(truth)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/results/E3_robustness_{name}.csv", index=False)
    _log(f"E3 {name}:\n{res.pivot(index='transform', columns='method', values='recall').round(3)}")
    return res


# ------------------------------------------------------------------ E2: cost scaling
def e2_scaling(S: dict, dim: int = 2048, dup_rate: float = 0.02) -> pd.DataFrame:
    """Real ECDH PSI + HE verification on synthetic embeddings (cost does not depend on content)."""
    out = S["out"]
    rng = np.random.default_rng(0)
    rows = []

    def fed(n_sites, n_per):
        E, site = [], []
        base = rng.standard_normal((n_per, dim)).astype(np.float32)
        for s in range(n_sites):
            x = rng.standard_normal((n_per, dim)).astype(np.float32)
            k = int(dup_rate * n_per)
            x[:k] = base[:k] + 0.05 * rng.standard_normal((k, dim))  # planted near-duplicates
            E.append(x / np.linalg.norm(x, axis=1, keepdims=True))
            site += [f"s{s}"] * n_per
        df = pd.DataFrame(dict(image_id=[f"x{i}" for i in range(len(site))], site=site, split="train", label=0, path=""))
        return df, np.concatenate(E)

    he = HEVerifier() if S.get("he_verify", True) else None
    prev = f"{out}/results/E2_scaling.csv"
    if os.path.exists(prev):
        rows = pd.read_csv(prev).to_dict("records")
    for n_sites, n_per in S["e2_grid"]:
        if any(r["n_sites"] == n_sites and r["images_per_site"] == n_per for r in rows):
            continue
        if time.time() > S.get("deadline", float("inf")) - 1800:
            _log("E2: time budget reached, remaining sizes run in the next session")
            break
        df, E = fed(n_sites, n_per)
        P = AuditParams(n_bands=16, band_width=16, tau=0.9, psi="real", verify="he" if he else "plain",
                        exact=False, within_site=False)
        t = time.time()
        try:
            r = run_stage_a(df, E, P, he=he)
        except CandidateOverflow as e:
            _log(f"E2 sites={n_sites} n={n_per}: skipped ({e})")
            continue
        pc = pd.DataFrame(r.per_pair_cost)
        est_s = float(pc.get("he_est_seconds", pd.Series(dtype=float)).fillna(0).sum())
        est_b = float(pc.get("he_est_bytes", pd.Series(dtype=float)).fillna(0).sum())
        rows.append(dict(n_sites=n_sites, images_per_site=n_per, seconds=time.time() - t, mb=r.cost.bytes / 1e6,
                         est_seconds_full_he=time.time() - t + est_s, est_mb_full_he=(r.cost.bytes + est_b) / 1e6,
                         candidates=r.n_candidates, matches=len(r.pairs),
                         psi_seconds=sum(d[1] for d in r.cost.detail if d[0].startswith("bands")),
                         psi_mb=sum(d[2] for d in r.cost.detail if d[0].startswith("bands")) / 1e6,
                         he_mb=sum(d[2] for d in r.cost.detail if d[0].startswith("he_verify")) / 1e6,
                         he_seconds=sum(d[1] for d in r.cost.detail if d[0] == "he_verify"),
                         mb_per_site=r.cost.bytes / 1e6 / n_sites))
        _log(f"E2 sites={n_sites} n={n_per}: {rows[-1]}")
        pd.DataFrame(rows).to_csv(f"{out}/results/E2_scaling.csv", index=False)
    return pd.DataFrame(rows)


def brain_dedup_pool(S: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Pool for controlled leakage: one representative per duplicate group (audit ∪ central truth),
    4-class images only."""
    out = S["out"]
    pairs = pd.read_csv(f"{out}/results/audit_pairs_brain.csv")
    C = gt.central_truth(df, cache=f"{out}/data/brain_truth.pkl")
    keep = greedy_keep(len(df), set(zip(pairs.a, pairs.b)) | C["truth"])
    pool = df[keep & df["label_name"].isin(sources.BRAIN4).values].reset_index(drop=True)
    pool.to_csv(f"{out}/data/brain_dedup.csv", index=False)
    _log(f"brain dedup pool: {len(pool)} of {len(df)} images kept")
    return pool


def restore_stores(S: dict, src: dict):
    """A continued session starts with an empty working folder: rebuild the decoded-image caches
    (and figshare's PNG export) from the saved manifests, keeping the exact same splits."""
    out = S["out"]
    for name, csv, size, rgb, masks in (("brain", "brain.csv", S["size"], False, False),
                                         ("ham", "ham_image.csv", S["size"], True, False),
                                         ("ct", "ct_pool.csv", 256, False, True)):
        p = f"{out}/data/{csv}"
        if not os.path.exists(p) or os.path.exists(f"{out}/store/{name}/ids.npy"):
            continue
        df = pd.read_csv(p).fillna({"group": ""})
        if name == "brain" and "figshare" in src and (df["site"] == "figshare").any():
            missing = [x for x in df.loc[df.site == "figshare", "path"] if not os.path.exists(x)]
            if missing:
                from ..data import manifest as mf
                mf.figshare_cheng(src["figshare"], "figshare")  # re-exports the PNGs to the same paths
        gone = [x for x in df["path"] if not os.path.exists(x)]
        if gone:
            raise FileNotFoundError(f"{len(gone)} images of {csv} not found (e.g. {gone[0]}); are all datasets attached?")
        _log(f"restoring image cache '{name}' ({len(df)} images)")
        build_store(df, f"{out}/store/{name}", size, rgb=rgb, masks=masks)


def run_prepare(S: dict, skip_e2: bool = False):
    out = S["out"]
    for d in ("results", "emb", "data", "tmp"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    done = f"{out}/results/PREPARE_DONE.json"
    if os.path.exists(done):
        prev = json.load(open(done))
        now = sources.detect_sources(S["input_root"], S.get("source_overrides"))
        if prev.get("sources") and set(prev["sources"]) >= set(now):
            _log("prepare already done (found PREPARE_DONE.json) — skipping; restoring image caches")
            restore_stores(S, now)
            return
        _log("an earlier prepare saw fewer datasets than are attached now — preparing again")
        os.remove(done)
    src = prepare_data(S)
    cfgs = {}
    for name, csv in (("brain", "brain.csv"), ("ham", "ham_image.csv")):
        p = f"{out}/data/{csv}"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p).fillna({"group": ""})
        sw = e1_sweep(S, name, df)
        cfgs[name] = choose_config(sw)
        _log(f"chosen audit config for {name}: {cfgs[name]}")
        e1_final(S, name, df, cfgs[name])
        e3_robustness(S, name, df, cfgs[name], n=S.get("e3_n", 300))
        if name == "brain":
            brain_dedup_pool(S, df)
    if not skip_e2:
        e2_scaling(S)
    # theory curve for the paper (collision probability vs cosine)
    cos = np.linspace(0.5, 1, 51)
    pd.DataFrame({f"{nb}x{bw}": [collision_prob(c, nb, bw) for c in cos] for nb, bw in S["band_grid"]}, index=cos) \
        .to_csv(f"{out}/results/E1_lsh_theory.csv")
    with open(done, "w") as f:
        json.dump(dict(configs=cfgs, sources=src, time=time.ctime()), f, indent=1)
