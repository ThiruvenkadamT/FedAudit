"""Single-process federated simulator with standard algorithms, implemented afresh for this study.

FedAvg (McMahan 2017) · FedProx (Li 2020) · SCAFFOLD (Karimireddy 2020, option II) ·
FedNova (Wang 2020) · FedDyn (Acar 2021) · FedAdam (Reddi 2021) · MOON (Li 2021) ·
FedBN (Li 2021) · local-only · centralised.

Stage D hooks into local training only (robust augmentation + consistency loss), so it composes
with every algorithm and adds no communication. The feature-masking baseline keeps a local,
never-averaged channel mask.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import TensorSet, to_float
from .models import build_model

ALGOS = ["fedavg", "fedprox", "scaffold", "fednova", "feddyn", "fedadam", "moon", "fedbn", "local", "central"]
PERSONAL = {"fedbn", "local"}


@dataclass
class FLConfig:
    algo: str = "fedavg"
    model: str = "cnn"
    in_ch: int = 1
    n_classes: int = 4
    rounds: int = 30
    local_epochs: int = 1
    batch: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    wd: float = 1e-4
    mu: float = 0.01          # FedProx
    alpha: float = 0.01       # FedDyn
    moon_mu: float = 1.0
    moon_temp: float = 0.5
    server_lr: float = 0.01   # FedAdam
    seed: int = 0
    device: str = "cpu"
    robust: dict | None = None      # {'lam': 1.0, 'p': 0.5}  Stage D
    featmask: float = 0.0            # fraction of channels masked locally (baseline)
    flip_aug: bool = True
    log_every: int = 0
    extra: dict = field(default_factory=dict)


def _bn_keys(model: nn.Module) -> set:
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for k in m.state_dict():
                keys.add(f"{name}.{k}")
    return keys


LOCAL_ONLY_KEYS = ("chan_mask",)


def _avg_state(states, weights, skip=set()):
    out = copy.deepcopy(states[0])
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    for k in out:
        if k in skip or any(s in k for s in LOCAL_ONLY_KEYS) or "num_batches_tracked" in k:
            continue
        out[k] = sum(wi * s[k].float() for wi, s in zip(w, states)).to(states[0][k].dtype)
    return out


def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def _augment(x: torch.Tensor, gen) -> torch.Tensor:
    if torch.rand(1, generator=gen).item() < 0.5:
        x = torch.flip(x, dims=[3])
    return x


def _robust_pair(ds: TensorSet, idx, gen):
    """Pick one Stage-D variant per sample: returns T(x) for the batch."""
    keys = list(ds.variants.keys())
    choice = torch.randint(len(keys), (len(idx),), generator=gen)
    xt = ds.x[idx].clone()
    for j, k in enumerate(keys):
        m = (choice == j).to(ds.x.device)
        if m.any():
            xt[m] = ds.variants[k][idx.to(ds.x.device)[m]]
    return xt


def local_train(model, ds: TensorSet, cfg: FLConfig, gen, *, global_model=None, c_glob=None, c_loc=None,
                dyn_grad=None, prev_model=None, epochs=None):
    """Returns number of local steps."""
    model.train()
    dev = cfg.device
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0 if cfg.algo in ("scaffold", "fednova") else cfg.momentum,
                          weight_decay=cfg.wd)
    gparams = [p.detach().clone() for p in _params(global_model)] if global_model is not None else None
    steps = 0
    for _ in range(epochs or cfg.local_epochs):
        for idx in _batches(len(ds), cfg.batch, gen):
            if len(idx) < 2:
                continue
            idx_d = idx.to(ds.x.device)
            xb_u8 = ds.x[idx_d]
            yb = ds.y[idx_d].to(dev)
            use_robust = cfg.robust is not None and ds.variants
            if use_robust:
                xt_u8 = _robust_pair(ds, idx, gen)
                swap = (torch.rand(len(idx), generator=gen) < cfg.robust.get("p", 0.5)).to(ds.x.device)
                xa_u8 = torch.where(swap[:, None, None, None], xt_u8, xb_u8)
            else:
                xa_u8 = xb_u8
            flip = torch.rand(1, generator=gen).item() < 0.5 and cfg.flip_aug
            prep = lambda t: to_float(torch.flip(t, dims=[3]) if flip else t).to(dev)
            xa = prep(xa_u8)
            logits = model(xa)
            loss = F.cross_entropy(logits, yb)
            if use_robust:
                lo = model(prep(xb_u8))
                lt = model(prep(xt_u8))
                loss = loss + cfg.robust.get("lam", 1.0) * F.kl_div(F.log_softmax(lt, 1), F.softmax(lo, 1), reduction="batchmean")
            if cfg.algo == "fedprox":
                loss = loss + cfg.mu / 2 * sum(((p - g) ** 2).sum() for p, g in zip(_params(model), gparams))
            elif cfg.algo == "feddyn":
                lin = sum((p * dg).sum() for p, dg in zip(_params(model), dyn_grad))
                quad = sum(((p - g) ** 2).sum() for p, g in zip(_params(model), gparams))
                loss = loss - lin + cfg.alpha / 2 * quad
            elif cfg.algo == "moon":
                z = F.normalize(model.proj(model.features(xa)), dim=1)
                with torch.no_grad():
                    zg = F.normalize(global_model.proj(global_model.features(xa)), dim=1)
                    zp = F.normalize(prev_model.proj(prev_model.features(xa)), dim=1)
                pos = (z * zg).sum(1) / cfg.moon_temp
                neg = (z * zp).sum(1) / cfg.moon_temp
                loss = loss + cfg.moon_mu * F.cross_entropy(torch.stack([pos, neg], 1), torch.zeros(len(z), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.algo == "scaffold":
                for p, cg, cl in zip(_params(model), c_glob, c_loc):
                    if p.grad is not None:
                        p.grad.add_(cg - cl)
            opt.step()
            steps += 1
    return steps


@torch.no_grad()
def predict(model, ds: TensorSet, cfg: FLConfig, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ds), bs):
        out.append(F.softmax(model(to_float(ds.x[i:i + bs]).to(cfg.device)), 1).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, cfg.n_classes))


def fit_featmask(model, ds: TensorSet, cfg: FLConfig, has_text: np.ndarray | None):
    """Feature-masking baseline: locally zero the channels most correlated with the site's
    nuisance indicator (text present). Mask stays local."""
    if cfg.featmask <= 0 or has_text is None or has_text.std() == 0:
        return
    model.eval()
    with torch.no_grad():
        f = torch.cat([model.features(to_float(ds.x[i:i + 256]).to(cfg.device)).cpu() for i in range(0, len(ds), 256)]).numpy()
    t = (has_text - has_text.mean()) / (has_text.std() + 1e-8)
    fz = (f - f.mean(0)) / (f.std(0) + 1e-8)
    corr = np.abs((fz * t[:, None]).mean(0))
    k = int(cfg.featmask * len(corr))
    mask = torch.ones(len(corr))
    mask[np.argsort(-corr)[:k]] = 0
    model.chan_mask.copy_(mask.to(model.chan_mask.device))


def run_federated(cfg: FLConfig, train: dict, nuisance: dict | None = None, callback=None):
    """train: {site: TensorSet}. Returns {'global': model, 'personal': {site: model}, 'history': [...]}."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    sites = sorted(train)
    n = np.array([len(train[s]) for s in sites], float)
    dev = cfg.device
    g = build_model(cfg.model, cfg.in_ch, cfg.n_classes).to(dev)
    hist = []

    if cfg.algo == "central":
        pooled = TensorSet(torch.cat([train[s].x for s in sites]), torch.cat([train[s].y for s in sites]),
                           np.concatenate([train[s].ids for s in sites]),
                           {k: torch.cat([train[s].variants[k] for s in sites]) for k in train[sites[0]].variants}
                           if train[sites[0]].variants else None)
        for r in range(cfg.rounds):
            local_train(g, pooled, cfg, gen)
            if callback:
                hist.append(callback(r, g, {s: g for s in sites}))
        return dict(global_=g, personal={s: g for s in sites}, history=hist)

    personal = {s: copy.deepcopy(g) for s in sites}
    if cfg.algo == "local":
        for s in sites:
            m = personal[s]
            if cfg.featmask and nuisance:
                fit_featmask(m, train[s], cfg, nuisance.get(s))
            for r in range(cfg.rounds):
                local_train(m, train[s], cfg, gen)
        if callback:
            hist.append(callback(cfg.rounds - 1, None, personal))
        return dict(global_=None, personal=personal, history=hist)

    bn = _bn_keys(g) if cfg.algo == "fedbn" else set()
    zeros = lambda: [torch.zeros_like(p) for p in _params(g)]
    c_glob, c_loc = zeros(), {s: zeros() for s in sites}
    dyn_grad = {s: zeros() for s in sites}
    h_dyn = zeros()
    adam_m, adam_v = zeros(), zeros()
    prev = {s: copy.deepcopy(g) for s in sites}

    for r in range(cfg.rounds):
        states, steps_l, deltas_c = [], [], []
        gstate = copy.deepcopy(g.state_dict())
        gparams = [p.detach().clone() for p in _params(g)]
        for s in sites:
            m = personal[s]
            sd = copy.deepcopy(gstate)
            if cfg.algo == "fedbn":
                own = m.state_dict()
                for k in bn:
                    sd[k] = own[k]
            for k in sd:
                if any(t in k for t in LOCAL_ONLY_KEYS):
                    sd[k] = m.state_dict()[k]
            m.load_state_dict(sd)
            if cfg.featmask and nuisance and r % 5 == 0:
                fit_featmask(m, train[s], cfg, nuisance.get(s))
            K = local_train(m, train[s], cfg, gen, global_model=g, c_glob=c_glob, c_loc=c_loc[s],
                            dyn_grad=dyn_grad[s], prev_model=prev[s])
            K = max(K, 1)
            steps_l.append(K)
            if cfg.algo == "scaffold":
                new_c = [cl - cg + (gp - p.detach()) / (K * cfg.lr) for cl, cg, gp, p in zip(c_loc[s], c_glob, gparams, _params(m))]
                deltas_c.append([nc - cl for nc, cl in zip(new_c, c_loc[s])])
                c_loc[s] = new_c
            if cfg.algo == "feddyn":
                dyn_grad[s] = [dg - cfg.alpha * (p.detach() - gp) for dg, p, gp in zip(dyn_grad[s], _params(m), gparams)]
            if cfg.algo == "moon":
                prev[s] = copy.deepcopy(m)
            states.append(copy.deepcopy(m.state_dict()))

        avg = _avg_state(states, n, skip=bn)
        if cfg.algo == "fednova":
            p = n / n.sum()
            tau_eff = float((p * np.array(steps_l)).sum())
            names = [k for k, _ in g.named_parameters()]
            new = copy.deepcopy(avg)
            for k in names:
                d = sum(pi * (gstate[k].float() - st[k].float()) / ki for pi, st, ki in zip(p, states, steps_l))
                new[k] = gstate[k] - tau_eff * d
            avg = new
        elif cfg.algo == "scaffold":
            c_glob = [cg + sum(dc[i] for dc in deltas_c) / len(sites) for i, cg in enumerate(c_glob)]
        elif cfg.algo == "feddyn":
            names = [k for k, _ in g.named_parameters()]
            mean_delta = {k: sum(st[k].float() for st in states) / len(sites) - gstate[k].float() for k in names}
            h_dyn = [h - cfg.alpha * mean_delta[k] for h, k in zip(h_dyn, names)]
            for h, k in zip(h_dyn, names):
                avg[k] = sum(st[k].float() for st in states) / len(sites) - h / cfg.alpha
        elif cfg.algo == "fedadam":
            names = [k for k, _ in g.named_parameters()]
            b1, b2, tau = 0.9, 0.99, 1e-3
            for i, k in enumerate(names):
                delta = avg[k].float() - gstate[k].float()
                adam_m[i] = b1 * adam_m[i] + (1 - b1) * delta
                adam_v[i] = b2 * adam_v[i] + (1 - b2) * delta * delta
                avg[k] = gstate[k] + cfg.server_lr * adam_m[i] / (adam_v[i].sqrt() + tau)
        g.load_state_dict(avg)
        if callback and (cfg.log_every and (r + 1) % cfg.log_every == 0 or r == cfg.rounds - 1):
            if cfg.algo in PERSONAL:
                for s in sites:  # personal model = global weights + own BN
                    sd = copy.deepcopy(g.state_dict())
                    own = personal[s].state_dict()
                    for k in bn | {k for k in sd if any(t in k for t in LOCAL_ONLY_KEYS)}:
                        sd[k] = own[k]
                    personal[s].load_state_dict(sd)
                hist.append(callback(r, g, personal))
            else:
                hist.append(callback(r, g, {s: g for s in sites}))

    for s in sites:  # final per-site model: global weights + local-only state (own BN for FedBN, own mask)
        sd = copy.deepcopy(g.state_dict())
        own = personal[s].state_dict()
        for k in bn | {k for k in sd if any(t in k for t in LOCAL_ONLY_KEYS)}:
            sd[k] = own[k]
        personal[s].load_state_dict(sd)
    return dict(global_=g, personal=personal, history=hist)
