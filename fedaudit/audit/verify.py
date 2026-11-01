"""Stage A step 5: secure cosine-threshold verification of candidate pairs.

Protocol (two semi-honest sites, A holds key):
  1. A encrypts each candidate embedding a under CKKS and sends ciphertexts + public context.
  2. For each candidate pair (a, b), B computes Enc(r·(⟨a,b⟩ − τ)) with a fresh random r ∈ [1, R]
     using only its plaintext b, and returns it.
  3. A decrypts and learns only the sign (match / no match) — the random scale hides the
     magnitude — and shares the yes/no bit with B.
Neither side sees the other's embedding. `mode='plain'` computes the identical decision in the
clear for fast parameter sweeps (E1); `mode='he'` runs the real protocol and logs its cost (E2).
"""
from __future__ import annotations

import time

import numpy as np

from .psi_protocol import CostLog


class HEVerifier:
    def __init__(self, poly_mod: int = 8192, scale_bits: int = 40, r_max: float = 64.0, seed: int = 0):
        import tenseal as ts
        self.ts = ts
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod, -1, [60, scale_bits, scale_bits, 60])
        self.ctx.global_scale = 2 ** scale_bits
        self.ctx.generate_galois_keys()
        self.public_bytes = len(self.ctx.serialize(save_secret_key=False, save_galois_keys=True))
        self.r_max = r_max
        self.rng = np.random.default_rng(seed)

    def verify(self, A: np.ndarray, B: np.ndarray, pairs: list[tuple[int, int]], tau: float,
               log: CostLog | None = None) -> np.ndarray:
        """A, B: (n, d) normalized embeddings at each site; pairs index into A and B."""
        t = time.perf_counter()
        nbytes = self.public_bytes
        enc = {}
        for i in sorted({p[0] for p in pairs}):
            enc[i] = self.ts.ckks_vector(self.ctx, A[i].astype(np.float64).tolist())
            nbytes += len(enc[i].serialize())
        out = np.zeros(len(pairs), bool)
        for k, (i, j) in enumerate(pairs):
            r = float(self.rng.uniform(1.0, self.r_max))
            c = enc[i].dot(B[j].astype(np.float64).tolist())
            c = (c - tau) * r
            nbytes += len(c.serialize())
            out[k] = c.decrypt()[0] > 0
        nbytes += len(pairs) // 8 + 1  # A returns the decision bits
        if log is not None:
            log.add("he_verify", time.perf_counter() - t, nbytes)
        return out


def plain_verify(A: np.ndarray, B: np.ndarray, pairs, tau: float, chunk: int = 100_000) -> np.ndarray:
    if not len(pairs):
        return np.zeros(0, bool)
    p = np.asarray(pairs)
    out = np.empty(len(p), bool)
    for k in range(0, len(p), chunk):  # chunked: gathering all candidate rows at once can need >100 GB
        q = p[k:k + chunk]
        out[k:k + chunk] = np.einsum("ij,ij->i", A[q[:, 0]], B[q[:, 1]]) > tau
    return out
