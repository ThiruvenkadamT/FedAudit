"""Step 3: every table and figure for the paper from out/results and out/runs."""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.stats import paired_ci
from ..shortcut.stage_c import AuditConfig, run_audit
from .jobs import job_id

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
ALGO_ORDER = ["fedavg", "fedprox", "scaffold", "fednova", "feddyn", "fedadam", "moon", "fedbn", "local", "central"]
ALGO_NAME = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fednova": "FedNova", "feddyn": "FedDyn",
             "fedadam": "FedAdam", "moon": "MOON", "fedbn": "FedBN", "local": "Local-only", "central": "Centralised"}
VARIANT_NAME = {"none": "No augmentation", "headcrop": "Head crop", "blanktrain": "Text blanked in training",
                "featmask": "Feature masking", "robust": "FedAudit-D (ours)"}


def load_runs(out: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(f"{out}/runs/*.json"):
        r = json.load(open(p))
        base = dict(r["job"])
        for k, v in r.items():
            if k not in ("job", "metrics") and not isinstance(v, dict):
                base[k] = v
        for k, v in (r.get("deletion") or {}).items():
            base[k] = v
        for m in r["metrics"]:
            rows.append({**base, **m})
    return pd.DataFrame(rows)


def _save_fig(fig, out, name):
    os.makedirs(f"{out}/analysis/figures", exist_ok=True)
    fig.savefig(f"{out}/analysis/figures/{name}.png", dpi=600, bbox_inches="tight")
    try:
        fig.savefig(f"{out}/analysis/figures/{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        pass
    plt.close(fig)


def _mean_sd(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return ""
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


# ------------------------------------------------------------------ audit tables (E1–E3)
def tables_audit(out, T):
    for name in ("brain", "ham"):
        p = f"{out}/results/E1_sweep_{name}.csv"
        if not os.path.exists(p):
            continue
        sw = pd.read_csv(p)
        cols_all = [c for c in ("all_precision", "all_recall", "all_f1", "all_n_true") if c in sw]
        base = sw[sw.method != "fedaudit"][["method", "precision", "recall", "f1", "n_pred", "n_true"] + cols_all]
        fa = sw[sw.method == "fedaudit"]
        best = fa.sort_values("f1", ascending=False).groupby("encoder").head(1)
        T[f"E1_{name}_baselines"] = base
        T[f"E1_{name}_best_per_encoder"] = best[["encoder", "n_bands", "band_width", "tau", "precision", "recall", "f1", "candidates"]
                                                + cols_all + [c for c in ("group_recall",) if c in best]]
        fin = f"{out}/results/E1_final_{name}.csv"
        if os.path.exists(fin):
            T[f"E1_{name}_real_protocol"] = pd.read_csv(fin)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        for enc, d in fa.groupby("encoder"):
            ax.scatter(d["recall"], d["precision"], s=8, alpha=0.6, label=f"FedAudit ({enc})")
        for _, r in base.iterrows():
            ax.scatter(r["recall"], r["precision"], marker="x", s=30, label=r["method"].replace("central_", "Central "))
        ax.set_xlabel("Recall (cross-site pairs)")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, frameon=False)
        ax.set_title({"brain": "Brain MRI sources", "ham": "HAM10000"}[name])
        _save_fig(fig, out, f"F_E1_precision_recall_{name}")
    p = f"{out}/results/E2_scaling.csv"
    if os.path.exists(p):
        e2 = pd.read_csv(p)
        T["E2_scaling"] = e2
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
        two = e2[e2.n_sites == 2].sort_values("images_per_site")
        axs[0].plot(two.images_per_site, two.seconds / 60, "o-", label="time (min)")
        axs[0].set_xscale("log")
        axs[0].set_xlabel("Images per site (2 sites)")
        axs[0].set_ylabel("Minutes")
        ax2 = axs[0].twinx()
        ax2.plot(two.images_per_site, two.mb, "s--", color="tab:orange")
        ax2.set_ylabel("Communication (MB)")
        k = e2[e2.images_per_site == e2.images_per_site.mode()[0]].sort_values("n_sites")
        axs[1].plot(k.n_sites, k.seconds / 60, "o-")
        axs[1].set_xlabel("Number of sites")
        axs[1].set_ylabel("Minutes")
        _save_fig(fig, out, "F_E2_cost_scaling")
    for name in ("brain", "ham"):
        p = f"{out}/results/E3_robustness_{name}.csv"
        if os.path.exists(p):
            e3 = pd.read_csv(p).pivot(index="transform", columns="method", values="recall")
            T[f"E3_{name}"] = e3.reset_index()
            fig, ax = plt.subplots(figsize=(4.5, 2.6))
            e3.plot.bar(ax=ax, width=0.8)
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6, frameon=False, ncol=2)
            _save_fig(fig, out, f"F_E3_robustness_{name}")


# ------------------------------------------------------------------ E4
def class_standardized(out: str, exp: str = "E4brain", min_clean: int = 50) -> pd.DataFrame:
    """Accuracy on the unaudited and audited test sets with both reweighted to the unaudited class
    mix (classes with fewer than min_clean audited images are left out). Separates the effect of
    leaked images from the change in class composition that removing them causes."""
    from .jobs import leaked_ids
    df = pd.read_csv(f"{out}/data/brain.csv")
    df = df[df["label_name"].isin(["glioma", "meningioma", "notumor", "pituitary"])]  # images used in training
    pairs = pd.read_csv(f"{out}/results/audit_pairs_brain.csv")
    bad = leaked_ids(pairs, dict(zip(df["image_id"], df["split"])))
    rows = []
    for p in glob.glob(f"{out}/runs/*exp-{exp}_*.npz"):
        j = json.load(open(p[:-4] + ".json"))["job"]
        z = np.load(p)
        ks = sorted(k[:-4] for k in z if k.endswith("|ids"))
        ids = np.concatenate([z[k + "|ids"] for k in ks])
        y = np.concatenate([z[k + "|y"] for k in ks])
        pr = np.concatenate([z[k + "|p"] for k in ks]).astype(float)
        c = pr.argmax(1) == y
        L = np.isin(ids, list(bad))
        classes = np.unique(y)
        w = np.array([(y == k).sum() for k in classes], float)
        ok = np.array([((y == k) & ~L).sum() >= min_clean for k in classes])
        un = np.average([c[y == k].mean() for k in classes], weights=w * ok)
        au = np.average([c[(y == k) & ~L].mean() if ((y == k) & ~L).any() else 0 for k in classes], weights=w * ok)
        rows.append(dict(algo=j["algo"], seed=j["seed"], std_unaudited=un, std_audited=au))
    R = pd.DataFrame(rows)
    out_rows = []
    for a, g in R.groupby("algo"):
        ci = paired_ci(g["std_unaudited"] - g["std_audited"])
        out_rows.append(dict(algo=a, unaudited=g["std_unaudited"].mean(), audited=g["std_audited"].mean(),
                             inflation=ci["mean"], ci_lo=ci["lo"], ci_hi=ci["hi"], p=ci["p"]))
    return pd.DataFrame(out_rows)


def tables_e4(out, R, T):
    for exp, label, extra in (("E4brain", "brain", []), ("E4ham", "ham", ["split"])):
        d = R[(R.exp == exp) & (R.site == "ALL")]
        if d.empty:
            continue
        piv = d.pivot_table(index=["algo", "seed"] + extra, columns="view", values=["acc", "auc"]).reset_index()
        piv.columns = ["_".join(c).strip("_") for c in piv.columns]
        rows = []
        for keys, g in piv.groupby(["algo"] + extra):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(["algo"] + extra, keys))
            for met in ("acc", "auc"):
                for v in ("unaudited", "audited"):
                    row[f"{met}_{v}"] = _mean_sd(g[f"{met}_{v}"])
                ci = paired_ci(g[f"{met}_unaudited"] - g[f"{met}_audited"])
                row[f"{met}_inflation"] = f"{ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"
                row[f"{met}_p"] = ci["p"]
            if "acc_leaked_only" in g:
                row["acc_on_leaked_images"] = _mean_sd(g["acc_leaked_only"])
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab["algo"] = pd.Categorical(tab["algo"], ALGO_ORDER)
        T[f"E4_{label}_inflation"] = tab.sort_values(["algo"] + extra)
        # figure
        g = piv.copy()
        g["infl"] = g["acc_unaudited"] - g["acc_audited"]
        if extra:
            g = g[g.split == "image"]
        s = g.groupby("algo")["infl"].agg(["mean", "std"]).reindex([a for a in ALGO_ORDER if a in g.algo.unique()])
        fig, ax = plt.subplots(figsize=(4.8, 2.6))
        ax.bar([ALGO_NAME[a] for a in s.index], s["mean"] * 100, yerr=s["std"] * 100, capsize=2, color="#4C78A8")
        ax.set_ylabel("Accuracy inflation (pp)")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title({"brain": "Brain MRI sources (published splits)", "ham": "HAM10000 (image-level split)"}[label])
        _save_fig(fig, out, f"F_E4_inflation_{label}")
        if extra:  # image-level vs lesion-level split
            c = piv.pivot_table(index=["algo", "seed"], columns="split", values="acc_unaudited").dropna().reset_index()
            if {"image", "lesion"} <= set(c.columns):
                rows = []
                for a, gg in c.groupby("algo"):
                    ci = paired_ci(gg["image"] - gg["lesion"])
                    rows.append(dict(algo=a, acc_image_split=_mean_sd(gg["image"]), acc_lesion_split=_mean_sd(gg["lesion"]),
                                     gap=f"{ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]", p=ci["p"]))
                T["E4_ham_image_vs_lesion_split"] = pd.DataFrame(rows)
    if not R[R.exp == "E4brain"].empty:
        T["E4_brain_class_standardized"] = class_standardized(out)
    d = R[(R.exp == "E4leak") & (R.site == "ALL")]
    if not d.empty:
        piv = d.pivot_table(index=["algo", "part", "rate", "seed"], columns="view", values="acc").reset_index()
        piv["inflation"] = piv["unaudited"] - piv["truth_clean"]
        piv["residual_after_audit"] = piv["audited"] - piv["truth_clean"]
        meta = d.drop_duplicates(["algo", "part", "rate", "seed"])[["algo", "part", "rate", "seed", "audit_recall", "audit_precision"]]
        piv = piv.merge(meta, on=["algo", "part", "rate", "seed"])
        if "leaked_only" not in piv:  # no run with injected copies yet (e.g. quick test at rate 0)
            piv["leaked_only"] = np.nan
        agg = piv.groupby(["algo", "part", "rate"]).agg(inflation=("inflation", "mean"), inflation_sd=("inflation", "std"),
                                                        residual=("residual_after_audit", "mean"),
                                                        audit_recall=("audit_recall", "mean"),
                                                        audit_precision=("audit_precision", "mean"),
                                                        acc_leaked=("leaked_only", "mean")).reset_index()
        T["E4_controlled_leakage"] = agg
        parts = sorted(agg.part.unique())
        fig, axs = plt.subplots(1, len(parts), figsize=(2.4 * len(parts), 2.4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, p in zip(axs, parts):
            for a, g in agg[agg.part == p].groupby("algo"):
                ax.errorbar(g.rate * 100, g.inflation * 100, yerr=g.inflation_sd * 100, marker="o", ms=3, capsize=2, label=ALGO_NAME[a])
            ax.set_title(p)
            ax.set_xlabel("Injected copies (% of test set)")
        axs[0].set_ylabel("Accuracy inflation (pp)")
        axs[-1].legend(fontsize=6, frameon=False)
        _save_fig(fig, out, "F_E4_controlled_leakage")


# ------------------------------------------------------------------ E5 detection vs privacy budget
def tables_e5(out, R, T, eps_list=(0.5, 1, 2, 4, None), noise_draws: int = 20):
    sdir = f"{out}/runs/stagec"
    if not os.path.isdir(sdir):
        return
    runs = R[R.exp.isin(["E6", "E5nat"])].drop_duplicates(["exp", "algo", "part", "variant", "seed"])
    rows = []
    for r in runs.itertuples():
        if r.variant == "headcrop":
            continue
        j = dict(exp=r.exp, algo=r.algo, part=r.part, variant=r.variant, seed=int(r.seed))
        jr = dict(j, variant="headcrop")
        pm, pr = f"{sdir}/{job_id(j)}.csv.gz", f"{sdir}/{job_id(jr)}.csv.gz"
        if not (os.path.exists(pm) and os.path.exists(pr)):
            continue
        sm, sr = pd.read_csv(pm), pd.read_csv(pr)
        sm = {s: d for s, d in sm.groupby("site")}
        sr = {s: d for s, d in sr.groupby("site")}
        for eps in eps_list:
            flags = []
            for k in range(noise_draws if eps else 1):
                a = run_audit(sm, sr, 2, AuditConfig(eps=eps, seed=1000 * k + int(r.seed)))
                flags.append(a["flag"])
            rows.append(dict(exp=r.exp, algo=r.algo, part=r.part, variant=r.variant, seed=r.seed,
                             eps="none" if eps is None else eps, flag_rate=float(np.mean(flags)),
                             attr_out=a["model"]["attr_out"], attr_ref=a["ref"]["attr_out"],
                             mask_s=a["model"]["mask_s"], swap_s=a["model"]["swap_s"], text_rate=a["model"]["text_rate"]))
    if not rows:
        return
    D = pd.DataFrame(rows)
    T["E5_per_run"] = D
    e6 = D[D.exp == "E6"]
    if not e6.empty:
        summ = e6.groupby(["variant", "eps"])["flag_rate"].mean().unstack("eps")
        summ.index = [VARIANT_NAME.get(i, i) + (" [positive]" if i == "none" else " [negative]" if i == "blanktrain" else "")
                      for i in summ.index]
        T["E5_flag_rate_vs_eps"] = summ.reset_index().rename(columns={"index": "model"})
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for v, g in e6.groupby("variant"):
            s = g.groupby("eps")["flag_rate"].mean()
            order = [e for e in [0.5, 1, 2, 4, "none"] if e in s.index]
            ax.plot(range(len(order)), [s[e] for e in order], "o-", label=VARIANT_NAME.get(v, v))
        ax.set_xticks(range(len(order)), [str(e) for e in order])
        ax.set_xlabel("Privacy budget ε")
        ax.set_ylabel("Shortcut flag rate")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=6, frameon=False)
        _save_fig(fig, out, "F_E5_detection_vs_eps")
    nat = D[D.exp == "E5nat"]
    if not nat.empty:
        T["E5_natural_ct_text"] = nat.groupby(["variant", "eps"])[["flag_rate", "attr_out", "attr_ref", "mask_s", "swap_s", "text_rate"]].mean().reset_index()


# ------------------------------------------------------------------ E6
def tables_e6(out, R, T):
    d = R[(R.exp == "E6") & (R.site == "ALL")]
    if d.empty:
        return
    piv = d.pivot_table(index=["algo", "part", "variant", "seed"], columns="view", values=["acc", "auc"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    meta = d.drop_duplicates(["algo", "part", "variant", "seed"])[["algo", "part", "variant", "seed"] +
                                                                    [c for c in d.columns if c.startswith("deletion_auc")]]
    piv = piv.merge(meta, on=["algo", "part", "variant", "seed"])
    piv["swap_drop"] = piv["acc_standard"] - piv["acc_swap"]
    cols = ["acc_standard", "acc_clean", "acc_swap", "auc_standard", "auc_clean", "auc_swap", "swap_drop",
            "acc_standard_CT", "acc_swap_CT", "acc_clean_CT", "acc_standard_MRI", "acc_swap_MRI", "acc_clean_MRI",
            "deletion_auc_saliency", "deletion_auc_random"]
    cols = [c for c in cols if c in piv]
    main = piv.groupby(["algo", "variant"])[cols].agg(_mean_sd).reset_index()
    main["variant"] = main["variant"].map(VARIANT_NAME)
    T["E6_main_by_algo_variant"] = main
    T["E6_by_partition"] = piv.groupby(["part", "algo", "variant"])[cols].mean().reset_index()
    rows = []  # robust vs each baseline, paired over (algo, part, seed)
    rb = piv[piv.variant == "robust"].set_index(["algo", "part", "seed"])
    for v in ("none", "headcrop", "featmask", "blanktrain"):
        b = piv[piv.variant == v].set_index(["algo", "part", "seed"])
        k = rb.index.intersection(b.index)
        for met in ("acc_swap", "acc_clean", "acc_standard", "swap_drop"):
            ci = paired_ci((rb.loc[k, met] - b.loc[k, met]).values)
            rows.append(dict(comparison=f"FedAudit-D − {VARIANT_NAME[v]}", metric=met, mean_diff=ci["mean"], ci_lo=ci["lo"],
                             ci_hi=ci["hi"], p=ci["p"], n_pairs=ci["n"]))
    T["E6_paired_tests"] = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, view in zip(axs, ("standard", "clean", "swap")):
        g = piv.groupby(["variant", "algo"])[f"acc_{view}"].mean().unstack("algo")
        g = g.reindex([v for v in ["none", "headcrop", "blanktrain", "featmask", "robust"] if v in g.index])
        g.index = [VARIANT_NAME[i] for i in g.index]
        g.plot.bar(ax=ax, width=0.8, legend=(view == "swap"))
        ax.set_title(f"{view} test")
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(axis="x", rotation=60, labelsize=6)
    axs[0].set_ylabel("Accuracy")
    if axs[2].get_legend():
        axs[2].legend([ALGO_NAME.get(t.get_text(), t.get_text()) for t in axs[2].get_legend().get_texts()], fontsize=6, frameon=False)
    _save_fig(fig, out, "F_E6_robust_training")


def _safe(fn, *a):
    import traceback
    try:
        fn(*a)
    except Exception:
        print(f"!!! analysis step {fn.__name__} failed (other tables continue):\n{traceback.format_exc()[-1500:]}", flush=True)
        plt.close("all")


def run_analysis(out: str):
    os.makedirs(f"{out}/analysis", exist_ok=True)
    T = {}
    rep = f"{out}/dataset_report.json"
    if os.path.exists(rep):
        T["T0_dataset_report"] = pd.json_normalize(json.load(open(rep)), sep=".").T.reset_index()
    _safe(tables_audit, out, T)
    R = load_runs(out)
    if not R.empty:
        R.to_csv(f"{out}/analysis/all_runs_long.csv", index=False)
        _safe(tables_e4, out, R, T)
        _safe(tables_e5, out, R, T)
        _safe(tables_e6, out, R, T)
        err = glob.glob(f"{out}/errors/*.txt")
        T["Z_run_status"] = pd.DataFrame([dict(finished_runs=R.drop_duplicates([c for c in ("exp", "algo", "seed", "part", "variant", "rate", "split") if c in R]).shape[0],
                                               failed_runs=len(err))])
    with pd.ExcelWriter(f"{out}/analysis/ALL_TABLES.xlsx") as xw:
        for k, v in T.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    with open(f"{out}/analysis/SUMMARY.txt", "w") as f:
        for k, v in T.items():
            f.write(f"\n===== {k} =====\n{v.to_string(index=False)}\n")
    n_fig = len(glob.glob(f"{out}/analysis/figures/*.png"))
    print(f"Wrote {len(T)} tables and {n_fig} figures")
    return T
