"""End-to-end smoke test on a fake Kaggle input tree (CPU, a few minutes).

    python tests/make_fake_kaggle_input.py /tmp/fake 20
    python tests/smoke_test.py /tmp/fake /tmp/out
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fedaudit.experiments.analyze import run_analysis  # noqa: E402
from fedaudit.experiments.prepare import e2_scaling, run_prepare  # noqa: E402
from fedaudit.experiments.runner import run_all  # noqa: E402


def main(inp, out):
    S = dict(out=os.path.join(out, "fedaudit_results"), input_root=inp, code_dir=ROOT, quick=1, size=64,
             encoders=["pixel"], band_grid=[(16, 16), (32, 8)], tau_grid=[0.9, 0.95], he_verify=True,
             e2_grid=[(2, 200), (3, 200)], e3_n=30, rounds=2, seeds=[0], stagec_per_site=16,
             deadline=time.time() + 3600)
    S["quick"] = 0  # fake data is already small: use all of it in prepare
    run_prepare(S, skip_e2=True)
    S["quick"] = 1  # one job per experiment type
    run_all(S, n_gpus=1, cpu_task=lambda: e2_scaling(S))
    T = run_analysis(S["out"])
    errs = os.listdir(os.path.join(S["out"], "errors")) if os.path.isdir(os.path.join(S["out"], "errors")) else []
    print(json.dumps(dict(tables=len(T), failed_runs=len(errs)), indent=1))
    assert not errs, errs


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
