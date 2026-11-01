"""Run Stage A on the brain MRI federation with the full protocol (ECDH PSI and CKKS verification)
and score it against the central reference.

    python scripts/audit_protocol_brain.py path/to/fedaudit_results

Uses the saved pixel embeddings (emb/brain_pixel.npz), the manifest (data/brain.csv), the central
reference (data/brain_truth.pkl) and the selected configuration (results/audit_config_brain.json).
SHA-256 inputs for the exact-copy step are taken from the images when they are available; otherwise
each image receives a stand-in hash of its exact-duplicate group in the reference, which gives PSI
the same equality structure.

Writes results/audit_pairs_brain.csv, results/E2_sitepair_cost_brain.csv and
results/E1_protocol_brain.json.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.audit import ground_truth as gt  # noqa: E402
from fedaudit.audit.psi_protocol import pixel_sha256  # noqa: E402
from fedaudit.audit.stage_a import AuditParams, run_stage_a  # noqa: E402


def group_hashes(n, exact_pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in exact_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [hashlib.sha256(str(find(i)).encode()).digest() for i in range(n)]


def image_hashes(paths):
    from PIL import Image
    return [pixel_sha256(np.asarray(Image.open(p))) for p in paths]


def main(res):
    df = pd.read_csv(f"{res}/data/brain.csv").fillna({"group": ""})
    z = np.load(f"{res}/emb/brain_pixel.npz", allow_pickle=True)
    if list(z["ids"]) != list(df.image_id):
        raise ValueError("embedding ids do not match the manifest")
    ref = pickle.load(open(f"{res}/data/brain_truth.pkl", "rb"))
    if all(os.path.exists(p) for p in df.path):
        sha = image_hashes(df.path)
    else:
        sha = group_hashes(len(df), ref["exact"])

    cfg = json.load(open(f"{res}/results/audit_config_brain.json"))
    P = AuditParams(n_bands=cfg["n_bands"], band_width=cfg["band_width"], tau=cfg["tau"], psi="real", verify="he",
                    within_site=True)
    t = time.time()
    r = run_stage_a(df, z["emb"], P, sha=sha, verbose=True)
    sec = time.time() - t

    r.pairs.to_csv(f"{res}/results/audit_pairs_brain.csv", index=False)
    pc = pd.DataFrame(r.per_pair_cost)
    pc.to_csv(f"{res}/results/E2_sitepair_cost_brain.csv", index=False)
    site = df.site.values
    pred_all = gt.pairs_from_audit(r.pairs)
    pred = gt.restrict(pred_all, site, True)
    est_s = float(pc.get("he_est_seconds", pd.Series(dtype=float)).fillna(0).sum())
    est_b = float(pc.get("he_est_bytes", pd.Series(dtype=float)).fillna(0).sum())
    out = dict(seconds=sec, mb=r.cost.bytes / 1e6, candidates=r.n_candidates,
               est_full_he_seconds=sec + est_s, est_full_he_mb=(r.cost.bytes + est_b) / 1e6,
               he_agreement=float(pc.get("he_agreement", pd.Series(dtype=float)).mean()),
               cross=gt.score(pred, gt.restrict(ref["truth"], site, True)), all=gt.score(pred_all, ref["truth"]),
               n_pairs=len(r.pairs))
    json.dump(out, open(f"{res}/results/E1_protocol_brain.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="fedaudit_results folder")
    main(ap.parse_args().results)
