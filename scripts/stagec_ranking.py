"""How well the Stage C statistics track the measured shortcut strength (experiments E5/E6).

    python scripts/stagec_ranking.py path/to/fedaudit_results

Shortcut strength of a model = accuracy on the standard test set minus accuracy with text of the other
class pasted in (swap test). For each CT/MRI run (all variants except head cropping, which serves as the
reference model) the script releases the federation statistics of Stage C without noise and at each
privacy budget (20 noise draws), and reports the Spearman correlation between the released swap and
masking sensitivities and the shortcut strength. It also calibrates the sensitivity margin of the flag
rule by leave-one-partition-out: the margin is chosen on four partitions to flag runs whose swap accuracy
drops by more than `--drop` and evaluated on the fifth.
Writes results/stagec_ranking.csv and results/stagec_ranking_summary.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.experiments.jobs import job_id  # noqa: E402
from fedaudit.shortcut.stage_c import AuditConfig, aggregate, site_vector, summarise  # noqa: E402

EPS = [None, 0.5, 1, 2, 4]


def release(stats, eps, seed):
    sites = sorted(stats)
    cfg = AuditConfig(eps=eps, seed=seed)
    tot, sig = aggregate([site_vector(stats[s], 2) for s in sites], cfg)
    return summarise(tot, sig, len(sites), cfg.mode)


def main(res, drop_thr, draws=20):
    rows = []
    for p in glob.glob(f"{res}/runs/*exp-E6_*.json"):
        r = json.load(open(p))
        j = r["job"]
        if j["variant"] == "headcrop":
            continue
        acc = {m["view"]: m["acc"] for m in r["metrics"] if m["site"] == "ALL"}
        f = f"{res}/runs/stagec/{job_id(j)}.csv.gz"
        fr = f"{res}/runs/stagec/{job_id(dict(j, variant='headcrop'))}.csv.gz"
        if not (os.path.exists(f) and os.path.exists(fr)):
            continue
        st = {s: d for s, d in pd.read_csv(f).groupby("site")}
        sr = {s: d for s, d in pd.read_csv(fr).groupby("site")}
        for eps in EPS:
            for k in range(1 if eps is None else draws):
                m = release(st, eps, 1000 * k + j["seed"])
                ref = release(sr, eps, 1000 * k + j["seed"] + 1)
                rows.append(dict(algo=j["algo"], part=j["part"], variant=j["variant"], seed=j["seed"],
                                 eps="none" if eps is None else eps, draw=k, drop=acc["standard"] - acc["swap"],
                                 swap_s=m["swap_s"], mask_s=m["mask_s"], swap_se=m["swap_se"], mask_se=m["mask_se"],
                                 attr_gap=m["attr_out"] - ref["attr_out"]))
    D = pd.DataFrame(rows)
    D.to_csv(f"{res}/results/stagec_ranking.csv", index=False)

    S = {"runs": int(D[D.eps == "none"].shape[0]), "drop_threshold": drop_thr, "spearman": {}, "calibrated": {}}
    for e, g in D.groupby("eps", sort=False):
        S["spearman"][str(e)] = {c: float(np.mean([spearmanr(d[c], d["drop"])[0] for _, d in g.groupby("draw")]))
                                 for c in ("swap_s", "mask_s", "attr_gap")}

    from scipy.stats import norm
    z = norm.ppf(0.95)
    grid = np.round(np.arange(0.0, 0.40, 0.005), 3)

    def flags(d, m):
        return (d.attr_gap > 0.05) & (((d.swap_s - m) / d.swap_se > z) | ((d.mask_s - m) / d.mask_se > z))

    def bal_acc(d, m):
        y, f = d["drop"] > drop_thr, flags(d, m)
        return 0.5 * (f[y].mean() + (~f[~y]).mean())

    for e, g in D.groupby("eps", sort=False):
        res_e, chosen = [], []
        for held in sorted(g.part.unique()):
            tr, te = g[g.part != held], g[g.part == held]
            m = grid[int(np.argmax([bal_acc(tr, x) for x in grid]))]
            chosen.append(float(m))
            y, f = te["drop"] > drop_thr, flags(te, m)
            res_e.append((f[y].sum(), y.sum(), f[~y].sum(), (~y).sum()))
        tp, P, fp, N = (sum(x[i] for x in res_e) for i in range(4))
        S["calibrated"][str(e)] = dict(margins=chosen, detection=float(tp / P), false_alarm=float(fp / N))
    by_var = D[D.eps == "none"].groupby("variant")["drop"].agg(["mean", "min", "max"]).to_dict("index")
    S["drop_by_variant"] = by_var
    S["share_runs_above_threshold"] = float((D[D.eps == "none"]["drop"] > drop_thr).mean())
    json.dump(S, open(f"{res}/results/stagec_ranking_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="fedaudit_results folder")
    ap.add_argument("--drop", type=float, default=0.10, help="swap accuracy drop that counts as a shortcut")
    a = ap.parse_args()
    main(a.results, a.drop)
