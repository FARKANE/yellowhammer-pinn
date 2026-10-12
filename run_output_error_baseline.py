#!/usr/bin/env python3
"""
Classical identification baseline: output-error least squares
on the SAME reference field, the SAME model and the SAME train/test geometry as
the PINN, with gradients by automatic differentiation through a finite-volume
solver (equivalent to a discrete adjoint).

    python run_output_error_baseline.py --data-dir data --pinn-summary outputs/summary.json
    python run_output_error_baseline.py --data-dir data --test-years 2019-2024 --coarsen 10

Reports, for the output-error (OE) estimate and for the PINN estimate:
coefficients, training objective, forward-simulation R2 on the held-out cells /
years, and computation time.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from experiment_utils import add_common_args, config_from_args
from forward_solver import FVSolver, fit_output_error, pinn_to_phys
from pinn_yellowhammer import Trainer, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--coarsen", type=int, default=10, help="solver cell = coarsen x 2 km (10 -> 20 km)")
    ap.add_argument("--starts", type=int, default=4, help="random multi-starts (plus one default start)")
    ap.add_argument("--n-adam", type=int, default=250)
    ap.add_argument("--n-lbfgs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--free-kmax", action="store_true", help="estimate Kmax instead of fixing it at the PINN anchor")
    ap.add_argument("--pinn-summary", default=None, help="summary.json written by run_experiment1.py")
    ap.add_argument("--init-from-pinn", action="store_true",
                    help="add the PINN estimate (--pinn-summary) as a starting point: two-stage PINN -> output-error")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.init_from_pinn and not args.pinn_summary:
        ap.error("--init-from-pinn needs --pinn-summary")
    cfg = config_from_args(args)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # same split geometry as the PINN (no training here)
    geo = Trainer(cfg)
    solver = FVSolver(geo.hf, coarsen=args.coarsen, diffusion=cfg.diffusion, reaction=cfg.reaction)
    test_cells = (solver.cells_in_blocks(geo._block_ids, geo.test_blocks)
                  if cfg.split == "block" else np.zeros(solver.N, bool))
    train_years = [y for y in solver.years_ref if y not in cfg.test_years]
    print(f"solver: {solver.N} cells of {solver.dx_km:.0f} km; train cells {int((~test_cells).sum())}, "
          f"test cells {int(test_cells.sum())}; train years {train_years[0]}-{train_years[-1]}")

    Kmax = cfg.fix_Kmax_tilde * cfg.U
    fixed = {} if args.free_kmax else dict(Kmax=Kmax)
    rng = np.random.default_rng(args.seed)
    starts = [dict()]
    if args.init_from_pinn:
        pp = pinn_to_phys(json.load(open(args.pinn_summary))["best"]["params"])
        starts.insert(0, {k: float(v) for k, v in pp.items()})
    for _ in range(args.starts):
        starts.append(dict(D0=float(10 ** rng.uniform(1, 3.7)), alpha=float(rng.uniform(0.1, 0.9)),
                           rmax=float(10 ** rng.uniform(-2, 0)),
                           beta=float(rng.uniform(cfg.beta_min + 0.1, 0.9) if cfg.reaction == "source_sink"
                                      else rng.uniform(0.1, 0.9)),
                           gamma=float(rng.uniform(0.1, 0.9)), Kmax=Kmax))
    fits = []
    for i, init in enumerate(starts):
        res = fit_output_error(solver, U=cfg.U, train_years=train_years, train_cells=~test_cells,
                               ic_known=~test_cells, fixed=fixed, init=init, beta_min=cfg.beta_min,
                               n_adam=args.n_adam, lr=args.lr, n_lbfgs=args.n_lbfgs, verbose=args.verbose)
        fits.append(res)
        print(f"start {i}: J={res['objective']:.4e}  " +
              "  ".join(f"{k}={v:.4g}" for k, v in res["params"].items()) + f"  ({res['runtime_s'] / 60:.1f} min)")
    best = min(fits, key=lambda r: r["objective"] if math.isfinite(r["objective"]) else np.inf)

    def evaluate(p, u0):
        with torch.no_grad():
            sim = solver.simulate(p, u0=u0)
        ev = dict(all=solver.metrics(sim)["pooled"])
        if test_cells.any():
            ev["test_cells"] = solver.metrics(sim, cells=test_cells)["pooled"]
        ty = [y for y in cfg.test_years if y in solver.years_ref]
        if ty:
            ev["test_years"] = solver.metrics(sim, years=ty)["pooled"]
        ev["train"] = solver.metrics(sim, years=train_years[1:], cells=~test_cells)["pooled"]
        return ev

    report = dict(solver=dict(cells=solver.N, cell_km=solver.dx_km), split=cfg.split, test_years=list(cfg.test_years),
                  output_error=dict(params=best["params"], objective=best["objective"],
                                    runtime_s=sum(f["runtime_s"] for f in fits),
                                    all_starts=[dict(params=f["params"], objective=f["objective"]) for f in fits],
                                    evaluation=evaluate(best["params"], best["u0"])))
    if args.pinn_summary:
        s = json.load(open(args.pinn_summary))
        pp = pinn_to_phys(s["best"]["params"])
        report["pinn"] = dict(params=pp, runtime_s=60 * sum(r["minutes"] for r in s["runs"]),
                              evaluation=evaluate(pp, best["u0"]))
    write_json(report, os.path.join(out, "output_error_baseline.json"))

    print("\n{:24s} {:>14s} {:>14s}".format("", "output-error", "PINN" if "pinn" in report else ""))
    for k in best["params"]:
        pv = report.get("pinn", {}).get("params", {}).get(k)
        print("{:24s} {:>14.4g} {:>14s}".format(k, best["params"][k], "" if pv is None else f"{pv:.4g}"))
    for tag in ("train", "test_cells", "test_years", "all"):
        if tag in report["output_error"]["evaluation"]:
            a = report["output_error"]["evaluation"][tag]["r2"]
            b = report.get("pinn", {}).get("evaluation", {}).get(tag, {}).get("r2")
            print("{:24s} {:>14.3f} {:>14s}".format("simulation R2 " + tag, a, "" if b is None else f"{b:.3f}"))
    print("{:24s} {:>14.1f} {:>14s}".format("runtime [min]", report["output_error"]["runtime_s"] / 60,
                                             f"{report['pinn']['runtime_s'] / 60:.1f}" if "pinn" in report else ""))


if __name__ == "__main__":
    main()
