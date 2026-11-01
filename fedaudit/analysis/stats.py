"""Statistics for reporting: paired CIs over seeds, exact p-values, per-run bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_ci(diffs, conf: float = 0.95) -> dict:
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    m = d.mean()
    if n == 1:
        return dict(n=1, mean=m, lo=np.nan, hi=np.nan, p=np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + conf / 2, n - 1)
    p = stats.ttest_1samp(d, 0.0).pvalue if se > 0 else (0.0 if m != 0 else 1.0)
    return dict(n=n, mean=m, lo=m - t * se, hi=m + t * se, p=float(p))


def bootstrap_inflation(y, prob, leaked_mask, n_boot: int = 2000, seed: int = 0) -> dict:
    """Inflation = acc(all test) − acc(test without leaked images), bootstrapping test images."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(prob).argmax(1) == np.asarray(y)).astype(float)
    leaked = np.asarray(leaked_mask, bool)
    n = len(correct)
    est = correct.mean() - correct[~leaked].mean() if (~leaked).any() else np.nan
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        c, l = correct[i], leaked[i]
        if (~l).any():
            bs.append(c.mean() - c[~l].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return dict(inflation=est, lo=lo, hi=hi)


def summarise_inflation(df: pd.DataFrame, by=("dataset", "algo"), metric: str = "acc") -> pd.DataFrame:
    """df rows: one per run with columns metric_unaudited, metric_audited."""
    rows = []
    for k, d in df.groupby(list(by)):
        diff = d[f"{metric}_unaudited"] - d[f"{metric}_audited"]
        ci = paired_ci(diff)
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), metric=metric,
                         unaudited=d[f"{metric}_unaudited"].mean(), audited=d[f"{metric}_audited"].mean(),
                         inflation=ci["mean"], ci_lo=ci["lo"], ci_hi=ci["hi"], p=ci["p"], n_seeds=ci["n"]))
    return pd.DataFrame(rows)
