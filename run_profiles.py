#!/usr/bin/env python3
"""
Profiles and sensitivity analyses (review comment 5).

    # profile of D0 on a log grid (km^2/yr), 3 seeds per point
    python run_profiles.py --data-dir data --param D0 --values 0,1,3,10,30,100,300,1000,3000 --seeds 0 1 2

    # profile of any other coefficient (rmax in 1/yr; alpha, beta, gamma dimensionless)
    python run_profiles.py --data-dir data --param rmax --values 0.01,0.03,0.1,0.3,1

    # estimates versus the PDE weight lambda_f, and versus the Kmax anchor
    python run_profiles.py --data-dir data --lambda-sweep 0.05,0.15,0.5,1 --seeds 0 1 2 3 4
    python run_profiles.py --data-dir data --kmax-sweep 0.8,1.0,1.2,1.6,2.0

Profile: the coefficient is fixed, every other coefficient and the network are
re-optimised, and the TRAINING objective is recorded.  For each value the best seed
is the one with the lowest training objective (never the test score).  A nominal
Wilks interval is reported,

    Delta(theta) = N_eff * log(SSE(theta) / SSE_min) <= 3.84   (chi2_1, 95 %),

where SSE is the data misfit.  N_eff defaults to the number of training records,
which is OPTIMISTIC for a spatially correlated field: pass --n-eff (for instance the
number of independent spatial blocks x years) for a conservative interval.
"""
import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from experiment_utils import add_common_args, config_from_args
from pinn_yellowhammer import Trainer, write_json


def to_tilde(param, value, L_km, T, U):
    if param == "D0":
        return value * T / L_km ** 2
    if param == "rmax":
        return value * T
    if param == "Kmax":
        return value / U
    return value


def run_one(cfg, verbose=False):
    tr = Trainer(cfg)
    tr.train(verbose=verbose)
    o = tr.final_objective
    row = dict(seed=cfg.seed, obj_total=o["total"], obj_data=o["data"], obj_res=o["res"], obj_ic=o["ic"],
               obj_bc=o["bc"], test_r2=tr.val_glm_metrics()["r2"],
               select_r2=(tr.sel_metrics() or {}).get("r2"), n_train=len(tr.data["u"]))
    row.update(tr.physical_params())
    return row


def best_by_objective(df, by):
    return df.loc[df.groupby(by)["obj_total"].idxmin()].sort_values(by).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--param", choices=["D0", "alpha", "rmax", "beta", "gamma"], default=None)
    ap.add_argument("--values", default="0,1,3,10,30,100,300,1000,3000")
    ap.add_argument("--lambda-sweep", default=None)
    ap.add_argument("--kmax-sweep", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-eff", type=float, default=None)
    args = ap.parse_args()
    cfg0 = config_from_args(args)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    probe = Trainer(cfg0, build_points=False)
    L_km, T, U = probe.L / 1000.0, probe.T, probe.U
    rows = []

    if args.param:
        values = [float(v) for v in args.values.split(",")]
        for s in args.seeds:                                          # free fit (reference)
            r = run_one(replace(cfg0, seed=s), args.verbose); r.update(value=np.nan, kind="free"); rows.append(r)
            print(f"free seed {s}: obj={r['obj_total']:.4e} test R2={r['test_r2']:.3f}")
        for v in values:
            vt = to_tilde(args.param, v, L_km, T, U)
            if args.param == "beta" and cfg0.reaction == "logistic" and not 0 < v < 1:
                raise SystemExit("beta must be in (0,1) for the logistic reaction")
            for s in args.seeds:
                cfg = replace(cfg0, seed=s, fixed={**cfg0.fixed, args.param: vt})
                if args.param == "D0":
                    cfg = replace(cfg, fix_D0_tilde=-1.0)
                r = run_one(cfg, args.verbose); r.update(value=v, kind="profile"); rows.append(r)
                print(f"{args.param}={v:g} seed {s}: obj={r['obj_total']:.4e} data={r['obj_data']:.4e} "
                      f"test R2={r['test_r2']:.3f}")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"profile_{args.param}_all_runs.csv"), index=False)
        prof = best_by_objective(df[df.kind == "profile"], "value")
        free = df[df.kind == "free"].sort_values("obj_total").iloc[0]
        n_eff = args.n_eff or float(free["n_train"])
        sse = prof["obj_data"].to_numpy()
        sse_min = min(sse.min(), free["obj_data"])
        prof["delta_wilks"] = n_eff * np.log(sse / sse_min)
        prof["inside_95"] = prof["delta_wilks"] <= 3.84
        prof.to_csv(os.path.join(out, f"profile_{args.param}.csv"), index=False)
        free_key = {"D0": "D0_km2_per_yr", "rmax": "rmax_per_yr"}.get(args.param, args.param)
        summary = dict(param=args.param, n_eff=n_eff, free_fit_value=float(free[free_key]),
                       free_fit_objective=float(free["obj_total"]),
                       values_inside_nominal_95=prof.loc[prof.inside_95, "value"].tolist(),
                       note="N_eff = training records is optimistic for correlated data")
        write_json(summary, os.path.join(out, f"profile_{args.param}_summary.json"))
        _plot_profile(prof, free, args.param, free_key, os.path.join(out, f"fig_profile_{args.param}.pdf"))
        print(prof[["value", "obj_total", "obj_data", "test_r2", "delta_wilks", "inside_95"]].to_string(index=False))
        print(f"free fit {args.param} = {float(free[free_key]):.4g}")

    for kind, text, field in (("lambda_f", args.lambda_sweep, "w_res"), ("Kmax_tilde", args.kmax_sweep, "fix_Kmax_tilde")):
        if not text:
            continue
        rows = []
        for v in [float(x) for x in text.split(",")]:
            for s in args.seeds:
                r = run_one(replace(cfg0, seed=s, **{field: v}), args.verbose); r.update(value=v); rows.append(r)
                print(f"{kind}={v:g} seed {s}: test R2={r['test_r2']:.3f} rmax={r['rmax_per_yr']:.4f} "
                      f"beta={r['beta']:.3f} D0={r['D0_km2_per_yr']:.1f}")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"sweep_{kind}_all_runs.csv"), index=False)
        cols = ["D0_km2_per_yr", "alpha", "rmax_per_yr", "beta", "gamma", "test_r2", "obj_total"]
        agg = df.groupby("value")[cols].agg(["mean", "std"])
        agg.to_csv(os.path.join(out, f"sweep_{kind}.csv"))
        print(agg.round(4).to_string())
        _plot_sweep(df, kind, cols[:-1], os.path.join(out, f"fig_sweep_{kind}.pdf"))


def _plot_profile(prof, free, param, free_key, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = prof["value"].to_numpy(float)
    xs = np.where(x <= 0, np.nan, x) if param in ("D0", "rmax") else x
    for a_, key, lab in [(ax[0], "obj_total", "training objective"), (ax[1], "delta_wilks", r"$\Delta$ (nominal)")]:
        a_.plot(xs, prof[key], "-o", ms=4)
        if param in ("D0", "rmax"):
            a_.set_xscale("log")
        a_.set_xlabel(param); a_.set_ylabel(lab); a_.grid(alpha=0.3)
    ax[0].axvline(float(free[free_key]), color="r", ls="--", lw=1, label="free fit")
    ax[0].legend(fontsize=8)
    ax[1].axhline(3.84, color="k", lw=0.8, ls=":")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _plot_sweep(df, kind, cols, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3))
    g = df.groupby("value")
    for a_, c in zip(axes, cols):
        m, s = g[c].mean(), g[c].std()
        a_.errorbar(m.index, m.values, yerr=s.values, fmt="-o", ms=3, capsize=3)
        a_.set_title(c, fontsize=9); a_.set_xlabel(kind); a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
