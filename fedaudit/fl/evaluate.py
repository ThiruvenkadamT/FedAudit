"""Scoring federated models on several test views (unaudited, audited, clean, swap...)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .algorithms import predict


def metrics(y: np.ndarray, prob: np.ndarray, n_classes: int) -> dict:
    if len(y) == 0:
        return dict(n=0, acc=np.nan, bal_acc=np.nan, auc=np.nan)
    pred = prob.argmax(1)
    out = dict(n=len(y), acc=float((pred == y).mean()), bal_acc=float(balanced_accuracy_score(y, pred)))
    try:
        if n_classes == 2:
            out["auc"] = float(roc_auc_score(y, prob[:, 1]))
        else:
            present = np.unique(y)
            out["auc"] = float(roc_auc_score(y, prob[:, present] / prob[:, present].sum(1, keepdims=True),
                                             multi_class="ovr", average="macro", labels=present)) if len(present) > 1 else np.nan
    except ValueError:
        out["auc"] = np.nan
    return out


def evaluate(models: dict, views: dict, cfg, keep_probs: bool = False):
    """models: {site: model}; views: {view: {site: TensorSet}} → long DataFrame (+ optional probs)."""
    rows, probs = [], {}
    for v, sets in views.items():
        ys, ps = [], []
        for s, ds in sets.items():
            if len(ds) == 0:
                continue
            p = predict(models[s], ds, cfg)
            y = ds.y.cpu().numpy()
            rows.append(dict(view=v, site=s, **metrics(y, p, cfg.n_classes)))
            ys.append(y)
            ps.append(p)
            if keep_probs:
                probs[(v, s)] = (ds.ids, y, p)
        if ys:
            rows.append(dict(view=v, site="ALL", **metrics(np.concatenate(ys), np.concatenate(ps), cfg.n_classes)))
    df = pd.DataFrame(rows)
    return (df, probs) if keep_probs else df
