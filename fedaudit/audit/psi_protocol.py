"""Stage A steps 1 and 4: ECDH private set intersection between two sites.

Uses the OpenMined PSI library (ECDH, semi-honest). Both directions are run so each site
learns which of *its own* items are in the other site's set, and nothing about the others.
Every call records wall time and the exact bytes exchanged, which E2 reports.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import private_set_intersection.python as _psi
except Exception:  # pragma: no cover
    _psi = None


@dataclass
class CostLog:
    seconds: float = 0.0
    bytes: int = 0
    calls: int = 0
    detail: list = field(default_factory=list)

    def add(self, what: str, sec: float, nbytes: int):
        self.seconds += sec
        self.bytes += nbytes
        self.calls += 1
        self.detail.append((what, sec, nbytes))


def psi_client_learns(client_items: list[bytes], server_items: list[bytes], log: CostLog | None = None,
                      fpr: float = 1e-9, tag: str = "psi") -> set[int]:
    """Indices of client_items that are in server_items. The server learns only |client_items|."""
    if not client_items or not server_items:
        return set()
    if _psi is None:
        raise ImportError("pip install openmined.psi")
    t = time.perf_counter()
    c = _psi.client.CreateWithNewKey(True)
    s = _psi.server.CreateWithNewKey(True)
    # The library reports each matching element once, so repeated items (several images sharing a
    # band token) are sent once and mapped back to every position afterwards.
    uniq = list(dict.fromkeys(client_items))
    cl = [x.hex() for x in uniq]
    sv = list(dict.fromkeys(x.hex() for x in server_items))
    setup = s.CreateSetupMessage(fpr, len(cl), sv, _psi.DataStructure.RAW)
    req = c.CreateRequest(cl)
    resp = s.ProcessRequest(req)
    hit = {uniq[i] for i in c.GetIntersection(setup, resp)}
    idx = {i for i, x in enumerate(client_items) if x in hit}
    if log is not None:
        log.add(tag, time.perf_counter() - t,
                len(setup.SerializeToString()) + len(req.SerializeToString()) + len(resp.SerializeToString()))
    return idx


def mutual_psi(items_a: list[bytes], items_b: list[bytes], log: CostLog | None = None, tag="psi"):
    """Run PSI in both directions; returns (indices of a in B, indices of b in A)."""
    ia = psi_client_learns(items_a, items_b, log, tag=tag + ":a")
    ib = psi_client_learns(items_b, items_a, log, tag=tag + ":b")
    return ia, ib


def pixel_sha256(arr: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes() + str(arr.shape).encode()).digest()


def simulated_psi(items_a, items_b, log: CostLog | None = None, tag="sim"):
    """Same output as mutual_psi (PSI is exact up to fpr=1e-9) without the cryptography; used for
    parameter sweeps. Costs are not logged (E2 measures them with the real protocol)."""
    sa, sb = set(items_a), set(items_b)
    return ({i for i, x in enumerate(items_a) if x in sb}, {i for i, x in enumerate(items_b) if x in sa})
