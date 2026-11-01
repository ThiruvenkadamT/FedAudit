"""Leakage inflation on the brain MRI federation (experiment E4) for one results folder.

    python scripts/e4_inflation.py path/to/results_folder [--out table.csv]

Works for the main experiments (compact CNN) and for the ResNet-18 run of
notebooks/FedAudit_E4_ResNet18.ipynb. For each training method it reports accuracy on the full and on
the audited test set, the inflation with a 95% paired confidence interval over seeds, and the
class-standardized inflation (classes with fewer than 50 test images left after removal excluded).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.analysis.stats import paired_ci  # noqa: E402
from fedaudit.experiments.jobs import leaked_ids  # noqa: E402

BRAIN4 = ["glioma", "meningioma", "notumor", "pituitary"]
ALGOS = ["fedavg", "fedprox", "scaffold", "fednova", "feddyn", "fedadam", "moon", "fedbn", "local", "central"]


def predictions(path):
    z = np.load(path)
    ks = sorted(k[:-4] for k in z if k.endswith("|ids"))
    return (np.concatenate([z[k + "|ids"] for k in ks]), np.concatenate([z[k + "|y"] for k in ks]),
            np.concatenate([z[k + "|p"] for k in ks]).astype(float))


def inflation(res, min_clean=50):
    b = pd.read_csv(f"{res}/data/brain.csv")
    b4 = b[b.label_name.isin(BRAIN4)]
    bad = leaked_ids(pd.read_csv(f"{res}/results/audit_pairs_brain.csv"), dict(zip(b4.image_id, b4.split)))
    rows = []
    for p in glob.glob(f"{res}/runs/*exp-E4brain_*.npz"):
        j = json.load(open(p[:-4] + ".json"))["job"]
        ids, y, pr = predictions(p)
        L, c = np.isin(ids, list(bad)), pr.argmax(1) == y
        w = np.array([(y == k).sum() for k in range(4)], float)
        ok = np.array([((y == k) & ~L).sum() >= min_clean for k in range(4)])
        rows.append(dict(algo=j["algo"], seed=j["seed"], acc_un=c.mean(), acc_au=c[~L].mean(), acc_leak=c[L].mean(),
                         std_un=np.average([c[y == k].mean() for k in range(4)], weights=w * ok),
                         std_au=np.average([c[(y == k) & ~L].mean() if ((y == k) & ~L).any() else 0 for k in range(4)],
                                           weights=w * ok)))
    E = pd.DataFrame(rows)
    out = []
    for a in [x for x in ALGOS if x in set(E.algo)]:
        g = E[E.algo == a]
        ca, cs = paired_ci(g.acc_un - g.acc_au), paired_ci(g.std_un - g.std_au)
        out.append(dict(algo=a, seeds=len(g), acc_full=g.acc_un.mean(), acc_audited=g.acc_au.mean(),
                        acc_leaked=g.acc_leak.mean(), infl=ca["mean"], infl_lo=ca["lo"], infl_hi=ca["hi"], infl_p=ca["p"],
                        std_infl=cs["mean"], std_lo=cs["lo"], std_hi=cs["hi"], std_p=cs["p"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out")
    a = ap.parse_args()
    T = inflation(a.results)
    if a.out:
        T.to_csv(a.out, index=False)
    print(T.round(4).to_string(index=False))
