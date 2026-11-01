"""Stage A: privacy-preserving cross-site near-duplicate audit (orchestration).

For every pair of sites:
  1. exact copies: PSI on SHA-256 of decoded pixel arrays
  2. candidates: PSI on SimHash band tokens of centered head-crop embeddings
  3. verification: secure cosine threshold (HE) or its plaintext equivalent
Within-site duplicates are found by each site on its own data (no privacy issue).

Output: one row per matched pair with both sides' split, which is all the Action step needs.
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

from .lsh import band_tokens, hyperplanes, simhash_bits
from .psi_protocol import CostLog, mutual_psi, pixel_sha256, simulated_psi
from .verify import HEVerifier, plain_verify


@dataclass
class AuditParams:
    n_bands: int = 16
    band_width: int = 16
    tau: float | None = 0.92  # None: keep every candidate with its cosine (parameter sweeps)
    min_bands: int = 1
    center: bool = True
    verify: str = "plain"  # 'plain' | 'he'
    exact: bool = True
    within_site: bool = True
    psi: str = "real"  # 'real' (OpenMined ECDH PSI) | 'simulate' (identical output via set intersection, for sweeps)
    seed: int = 2026
    max_candidates: int | None = 3_000_000
    he_max: int = 3000  # above this many candidates per site pair, HE runs on a sample (cost extrapolated)


@dataclass
class AuditResult:
    pairs: pd.DataFrame
    cost: CostLog
    n_candidates: int = 0
    per_pair_cost: list = field(default_factory=list)


def _sha_all(paths):
    return [pixel_sha256(np.asarray(Image.open(p))) for p in paths]


def _rowdot(E: np.ndarray, ia, ib, chunk: int = 100_000) -> np.ndarray:
    """cos for many index pairs without gathering all rows at once."""
    out = np.empty(len(ia), np.float32)
    for k in range(0, len(ia), chunk):
        out[k:k + chunk] = np.einsum("ij,ij->i", E[ia[k:k + chunk]], E[ib[k:k + chunk]])
    return out


class CandidateOverflow(RuntimeError):
    """The band configuration puts too many image pairs in shared buckets to verify."""


def _candidates_from_tokens(tok_a, tok_b, hit_a: set, hit_b: set, flat_a, flat_b, min_bands: int,
                            max_candidates: int | None = None):
    """Both sides know the intersecting tokens (they are tokens of candidate images only);
    pairing images by shared token gives candidate pairs and their number of shared bands."""
    by_tok_b = defaultdict(list)
    for k in hit_b:
        img, _ = flat_b[k]
        by_tok_b[tok_b[k]].append(img)
    if max_candidates:
        n_a = defaultdict(int)
        for k in hit_a:
            n_a[tok_a[k]] += 1
        est = sum(c * len(by_tok_b.get(t, ())) for t, c in n_a.items())  # upper bound before de-duplication
        if est > max_candidates:
            raise CandidateOverflow(f"{est:,} band collisions > {max_candidates:,}")
    count = defaultdict(int)
    for k in hit_a:
        img_a, _ = flat_a[k]
        for img_b in by_tok_b.get(tok_a[k], ()):
            count[(img_a, img_b)] += 1
    return [p for p, c in count.items() if c >= min_bands]


def center_embeddings(emb: np.ndarray, center: np.ndarray) -> np.ndarray:
    x = (emb - center).astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def run_stage_a(df: pd.DataFrame, emb: np.ndarray, P: AuditParams, he: HEVerifier | None = None,
                sha: list | None = None, verbose: bool = False) -> AuditResult:
    df = df.reset_index(drop=True)
    log = CostLog()
    sites = sorted(df["site"].unique())
    idx = {s: np.where(df["site"].values == s)[0] for s in sites}
    if P.exact and sha is None:
        sha = _sha_all(df["path"])

    if P.center:
        # secure aggregation of per-site (sum, count): only the federation mean is revealed.
        # Every site centers and re-normalizes its own embeddings with it; hashing, verification
        # and the within-site check then all use the centered cosine.
        emb = center_embeddings(emb, sum(emb[idx[s]].sum(0) for s in sites) / len(df))
    planes = hyperplanes(emb.shape[1], P.n_bands * P.band_width, P.seed)
    bits = simhash_bits(emb, planes, None)
    toks = band_tokens(bits, P.n_bands, P.band_width)

    if P.verify == "he" and he is None:
        he = HEVerifier()

    rows = []
    n_cand = 0
    per_pair = []
    for sa, sb in itertools.combinations(sites, 2):
        ia, ib = idx[sa], idx[sb]
        t0, b0 = log.seconds, log.bytes
        found = {}
        if P.exact:
            ha, hb = [sha[i] for i in ia], [sha[i] for i in ib]
            xa, xb = (mutual_psi if P.psi == 'real' else simulated_psi)(ha, hb, log, tag=f"exact:{sa}-{sb}")
            pos_b = defaultdict(list)
            for k in xb:
                pos_b[hb[k]].append(k)
            for k in xa:
                for kb in pos_b.get(ha[k], ()):
                    found[(k, kb)] = "exact"
        flat_a = [(k, b) for k in range(len(ia)) for b in range(P.n_bands)]
        flat_b = [(k, b) for k in range(len(ib)) for b in range(P.n_bands)]
        tok_a = [toks[ia[k]][b] for k, b in flat_a]
        tok_b = [toks[ib[k]][b] for k, b in flat_b]
        hit_a, hit_b = (mutual_psi if P.psi == 'real' else simulated_psi)(tok_a, tok_b, log, tag=f"bands:{sa}-{sb}")
        cand = [p for p in _candidates_from_tokens(tok_a, tok_b, hit_a, hit_b, flat_a, flat_b, P.min_bands,
                                                   P.max_candidates) if p not in found]
        n_cand += len(cand)
        Ea, Eb = emb[ia], emb[ib]
        he_info = {}
        if P.verify == "he" and cand and len(cand) > P.he_max:
            # Encrypted check on a random sample (cost per pair, agreement with the plaintext rule);
            # the decisions use the identical plaintext rule; the full encrypted cost is extrapolated.
            ok = plain_verify(Ea, Eb, cand, P.tau)
            rs = np.random.default_rng(P.seed).choice(len(cand), P.he_max, replace=False)
            sub = CostLog()
            ok_he = he.verify(Ea, Eb, [cand[i] for i in rs], P.tau, sub)
            scale = len(cand) / P.he_max
            he_info = dict(he_sampled=P.he_max, he_agreement=float((ok_he == ok[rs]).mean()),
                           he_est_seconds=sub.seconds * scale, he_est_bytes=sub.bytes * scale)
            log.add("he_verify_sample", sub.seconds, sub.bytes)
        elif P.verify == "he" and cand:
            ok = he.verify(Ea, Eb, cand, P.tau, log)
        else:
            ok = plain_verify(Ea, Eb, cand, -2.0 if P.tau is None else P.tau)
        for p, o in zip(cand, ok):
            if o:
                found[p] = "near"
        for (ka, kb), kind in found.items():
            a, b = ia[ka], ib[kb]
            rows.append((a, b, kind))
        per_pair.append(dict(site_a=sa, site_b=sb, n_a=len(ia), n_b=len(ib), candidates=len(cand),
                             matches=len(found), seconds=log.seconds - t0, bytes=log.bytes - b0, **he_info))
        if verbose:
            print(f"{sa}-{sb}: cand={len(cand)} matched={len(found)} {log.seconds - t0:.1f}s {(log.bytes - b0) / 1e6:.2f}MB")

    if P.within_site:
        for s in sites:
            ii = idx[s]
            E = emb[ii]
            if P.exact:
                pos = defaultdict(list)
                for k, i in enumerate(ii):
                    pos[sha[i]].append(k)
                exact_pairs = {(g[x], g[y]) for g in pos.values() if len(g) > 1
                               for x in range(len(g)) for y in range(x + 1, len(g))}
            else:
                exact_pairs = set()
            for k0 in range(0, len(ii), 2048):
                S = E[k0:k0 + 2048] @ E.T
                r, c = np.where(S > P.tau)
                for x, y in zip(r + k0, c):
                    if x < y:
                        rows.append((ii[x], ii[y], "exact" if (x, y) in exact_pairs else "near"))
            for x, y in exact_pairs:
                if float(E[x] @ E[y]) <= P.tau:
                    rows.append((ii[x], ii[y], "exact"))

    pairs = pd.DataFrame(rows, columns=["a", "b", "kind"]).astype({"a": np.int64, "b": np.int64}).drop_duplicates(["a", "b"])
    pairs["id_a"] = df["image_id"].values[pairs["a"]]
    pairs["id_b"] = df["image_id"].values[pairs["b"]]
    for col in ("site", "split", "label"):
        pairs[f"{col}_a"] = df[col].values[pairs["a"]]
        pairs[f"{col}_b"] = df[col].values[pairs["b"]]
    pairs["cos"] = _rowdot(emb, pairs["a"].values, pairs["b"].values) if len(pairs) else []
    return AuditResult(pairs.reset_index(drop=True), log, n_cand, per_pair)


def leaked_test_ids(pairs: pd.DataFrame, cross_site_only: bool = False) -> set[str]:
    """Test (or val) images that have a near/exact copy in any training split — the Action step.
    Each site removes its own flagged images and reports only how many it removed."""
    p = pairs if not cross_site_only else pairs[pairs["site_a"] != pairs["site_b"]]
    bad = set()
    for _, r in p.iterrows():
        if r["split_a"] in ("test", "val") and r["split_b"] == "train":
            bad.add(r["id_a"])
        if r["split_b"] in ("test", "val") and r["split_a"] == "train":
            bad.add(r["id_b"])
    return bad


def audited(df: pd.DataFrame, pairs: pd.DataFrame, cross_site_only: bool = False) -> pd.DataFrame:
    bad = leaked_test_ids(pairs, cross_site_only)
    return df[~df["image_id"].isin(bad)].copy()
