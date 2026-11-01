"""What a hospital learns from the band-token PSI of Stage A, measured on the brain MRI federation.

    python scripts/token_leakage.py path/to/fedaudit_results

In the PSI on band tokens, hospital i learns which of its own tokens also occur at hospital j. For every
ordered pair of sites this script reports:
  - how many of i's images become candidates, and how many of those have no duplicate at j in the
    central reference (images about which i learns something although they are not copies);
  - for those non-duplicate candidates, the highest cosine similarity to any image at j, i.e. how close
    the image whose existence is revealed actually is;
  - how many images at j carry a matched token (the set of j's images consistent with what i learns);
  - how well the number of matched bands predicts a duplicate at j (AUC), i.e. how much i can infer
    before the encrypted test.
Writes results/token_leakage_brain.csv and results/token_leakage_brain_summary.json.
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.audit.lsh import band_tokens, hyperplanes, simhash_bits  # noqa: E402
from fedaudit.audit.stage_a import center_embeddings  # noqa: E402


def max_cos(A, B, chunk=2048):
    out = np.empty(len(A), np.float32)
    for k in range(0, len(A), chunk):
        out[k:k + chunk] = (A[k:k + chunk] @ B.T).max(1)
    return out


def main(res):
    df = pd.read_csv(f"{res}/data/brain.csv")
    z = np.load(f"{res}/emb/brain_pixel.npz", allow_pickle=True)
    if list(z["ids"]) != list(df.image_id):
        raise ValueError("embedding ids do not match the manifest")
    cfg = json.load(open(f"{res}/results/audit_config_brain.json"))
    nb, bw = cfg["n_bands"], cfg["band_width"]
    E = center_embeddings(z["emb"], z["emb"].mean(0))
    toks = band_tokens(simhash_bits(E, hyperplanes(E.shape[1], nb * bw, 2026), None), nb, bw)
    truth = pickle.load(open(f"{res}/data/brain_truth.pkl", "rb"))["truth"]
    site = df.site.values
    dup_at = {}
    for a, b in truth:
        dup_at.setdefault(a, set()).add(site[b])
        dup_at.setdefault(b, set()).add(site[a])

    rows, cos_nondup, cos_none, anon, scores, labels = [], [], [], [], [], []
    sites = sorted(set(site))
    idx = {s: np.where(site == s)[0] for s in sites}
    for si in sites:
        for sj in sites:
            if si == sj:
                continue
            count_j = Counter(t for k in idx[sj] for t in toks[k])
            ii = idx[si]
            n_match = np.array([sum(t in count_j for t in toks[k]) for k in ii])
            has_dup = np.array([sj in dup_at.get(k, ()) for k in ii])
            cand = n_match > 0
            mc = max_cos(E[ii], E[idx[sj]])
            nondup = cand & ~has_dup
            cos_nondup += mc[nondup].tolist()
            cos_none += mc[~cand & ~has_dup].tolist()
            for k in ii[nondup]:
                anon += [count_j[t] for t in toks[k] if t in count_j]
            scores += n_match.tolist()
            labels += has_dup.tolist()
            rows.append(dict(site_i=si, site_j=sj, images=len(ii), candidates=int(cand.sum()),
                             candidates_with_duplicate=int((cand & has_dup).sum()), candidates_without_duplicate=int(nondup.sum()),
                             duplicates_missed=int((~cand & has_dup).sum())))
    T = pd.DataFrame(rows)
    T.to_csv(f"{res}/results/token_leakage_brain.csv", index=False)
    cn, c0, an = np.array(cos_nondup), np.array(cos_none), np.array(anon)
    S = dict(images=int(T.images.sum()), candidates=int(T.candidates.sum()),
             candidates_without_duplicate=int(T.candidates_without_duplicate.sum()),
             share_without_duplicate_of_all_images=float(T.candidates_without_duplicate.sum() / T.images.sum()),
             share_without_duplicate_of_candidates=float(T.candidates_without_duplicate.sum() / T.candidates.sum()),
             maxcos_nondup_candidates=dict(zip(("q25", "median", "q75"), np.quantile(cn, [0.25, 0.5, 0.75]).tolist())),
             maxcos_noncandidates=dict(zip(("q25", "median", "q75"), np.quantile(c0, [0.25, 0.5, 0.75]).tolist())),
             share_nondup_candidates_maxcos_ge_0_92=float((cn >= cfg["tau"]).mean()),
             matched_token_images_at_j=dict(median=float(np.median(an)), share_unique=float((an == 1).mean())),
             auc_matched_bands_vs_duplicate=float(roc_auc_score(labels, scores)))
    json.dump(S, open(f"{res}/results/token_leakage_brain_summary.json", "w"), indent=1)
    print(T.to_string(index=False))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="fedaudit_results folder")
    main(ap.parse_args().results)
