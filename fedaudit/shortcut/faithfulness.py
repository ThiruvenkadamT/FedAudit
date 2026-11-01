"""Deletion test of explanation faithfulness (Stage D evaluation).

Pixels are removed (set to the background value) in order of decreasing SmoothGrad saliency,
in steps; the area under the predicted-class probability curve is the deletion AUC (lower =
the explanation points at pixels the model really uses). Random-order deletion is the reference.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..fl.data import TensorSet, to_float
from .stage_c import smoothgrad


@torch.no_grad()
def _prob(model, x, c):
    return F.softmax(model(x), 1).gather(1, c[:, None]).squeeze(1)


def deletion_auc(model, ds: TensorSet, device: str, n_images: int = 128, steps: int = 10, sg_n: int = 8,
                 seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_images]
    x = to_float(ds.x[idx.to(ds.x.device)]).to(device)
    model.eval()
    with torch.no_grad():
        c = model(x).argmax(1)
    sal = smoothgrad(model, x, c, n=sg_n)  # (B,H,W)
    B, H, W = sal.shape
    order_sal = sal.view(B, -1).argsort(1, descending=True)
    order_rnd = torch.stack([torch.randperm(H * W, generator=g) for _ in range(B)]).to(device)
    bg = to_float(torch.zeros(1, dtype=torch.uint8)).item()
    res = {}
    for name, order in (("saliency", order_sal), ("random", order_rnd)):
        curve = []
        for k in range(steps + 1):
            n_del = int(k / steps * H * W)
            xm = x.clone().view(B, x.shape[1], -1)
            if n_del:
                sel = order[:, :n_del]
                xm.scatter_(2, sel[:, None, :].expand(-1, x.shape[1], -1), bg)
            curve.append(_prob(model, xm.view_as(x), c).mean().item())
        res[f"deletion_auc_{name}"] = float(np.trapezoid(curve, dx=1 / steps))
    return res
