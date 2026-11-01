"""Time and traffic of the private audit on synthetic embeddings (experiment E2).

    python scripts/cost_scaling.py path/to/fedaudit_results

Runs the real ECDH PSI and CKKS verification for each (sites, images per site) setting and writes
results/E2_scaling.csv. Finished settings are skipped, so an interrupted run can be resumed.
The 2 x 100,000 setting needs more than 8 GB of memory and is left out unless --large is given.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.experiments.prepare import e2_scaling  # noqa: E402

GRID = [(2, 1000), (2, 5000), (3, 2000), (4, 2000), (5, 2000), (2, 20000), (8, 2000), (10, 2000)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="output folder (results/E2_scaling.csv is written inside it)")
    ap.add_argument("--large", action="store_true", help="also run two sites of 100,000 images")
    a = ap.parse_args()
    os.makedirs(f"{a.results}/results", exist_ok=True)
    grid = GRID + ([(2, 100000)] if a.large else [])
    print(e2_scaling(dict(out=a.results, he_verify=True, e2_grid=grid)).to_string(index=False))
