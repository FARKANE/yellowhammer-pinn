#!/usr/bin/env python3
"""
Experiment 1 - field reconstruction and coefficient identification (offline).

    # corrected protocol (spatial-block test set, selection subset, source-sink reaction)
    python run_experiment1.py --data-dir data

    # reproduce the submitted pipeline exactly (including its defects)
    python run_experiment1.py --data-dir data --preset paper

    # temporal hold-out: fit 2002-2018, predict 2019-2024
    python run_experiment1.py --data-dir data --test-years 2019-2024

    # covariate-only habitat index (removes the circular H)
    python run_experiment1.py --data-dir data --habitat-mat H_covariates_2002_2024.mat

Several seeds are trained.  The kept seed is chosen on the SELECTION records
(carved from training), never on the test records; mean +/- sd over seeds is
reported as well.  Diagnostics on the kept model: mass balance, dense residual,
forward PDE simulation with the identified coefficients, lambda_1 per year.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import diagnostics as dg
from experiment_utils import add_common_args, config_from_args, with_seed
from pinn_yellowhammer import Trainer, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--coarsen", type=int, default=5, help="solver cell = coarsen x grid cell (5 -> 10 km)")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg0 = config_from_args(args)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    write_json(cfg0.to_dict(), os.path.join(out, "config.json"))
    print(f"device={cfg0.device} preset={args.preset} split={cfg0.split} reaction={cfg0.reaction} "
          f"diffusion={cfg0.diffusion} habitat={'covariates' if cfg0.habitat_mat else 'u_GLM (circular)'}")

    runs, best, best_tr = [], None, None
    selection_on_test = cfg0.select_frac <= 0
    for s in args.seeds:
        t0 = time.time()
        tr = Trainer(with_seed(cfg0, s))
        tr.train(verbose=args.verbose)
        vm = tr.val_metrics()
        row = dict(seed=s, minutes=(time.time() - t0) / 60, test=tr.val_glm_metrics(), select=tr.sel_metrics(),
                   train=tr.train_metrics(), counts={k: v for k, v in vm.items() if k not in ("pred", "obs")},
                   params=tr.physical_params(), objective=tr.final_objective)
        runs.append(row)
        tr.save(os.path.join(out, f"checkpoint_seed{s}.pt"))
        crit = row["test"]["r2"] if selection_on_test else row["select"]["r2"]
        sel_txt = "n/a" if row["select"] is None else f"{row['select']['r2']:.4f}"
        print(f"seed {s}: selection R2={sel_txt}  test R2={row['test']['r2']:.4f}  "
              f"rmax={row['params']['rmax_per_yr']:.4f}/yr beta={row['params']['beta']:.3f}  "
              f"D0={row['params']['D0_km2_per_yr']:.1f} km2/yr  ({row['minutes']:.1f} min)")
        if best is None or crit > best[0]:
            best, best_tr = (crit, row), tr
        else:
            del tr

    keys = list(runs[0]["params"].keys())
    summary = dict(
        config=cfg0.to_dict(), habitat_source=best_tr.hf.habitat_source,
        selection_rule=("max TEST R2 (legacy: optimistic)" if selection_on_test else "max SELECTION R2"),
        best_seed=best[1]["seed"], best=best[1],
        across_seeds=dict(
            test_r2=dg.summarise([r["test"]["r2"] for r in runs]),
            test_rmse=dg.summarise([r["test"]["rmse"] for r in runs]),
            params={k: dg.summarise([r["params"].get(k, np.nan) for r in runs]) for k in keys}),
        runs=runs)
    pd.DataFrame([dict(seed=r["seed"], test_r2=r["test"]["r2"], test_rmse=r["test"]["rmse"],
                       select_r2=(r["select"] or {}).get("r2"), train_r2=(r["train"] or {}).get("r2"),
                       counts_r2=r["counts"]["r2"], **r["params"], **{"obj_" + k: v for k, v in r["objective"].items()})
                  for r in runs]).to_csv(os.path.join(out, "runs.csv"), index=False)

    tr = best_tr
    if not args.no_diagnostics:
        print("\n--- diagnostics on the kept model (seed %d) ---" % best[1]["seed"])
        mb = dg.mass_balance_report(tr)
        pd.DataFrame(mb["rows"]).to_csv(os.path.join(out, "mass_balance.csv"), index=False)
        summary["mass_balance_cumulative"] = mb["cumulative"]
        c = mb["cumulative"]
        print("cumulative change {:+.3f} birds/site = reaction {:+.3f} + diffusion {:+.3f} + residual {:+.3f}"
              " (boundary flux {:+.3f})".format(c["dU_dt"], c["reaction"], c["diffusion"], c["residual"],
                                               c["boundary_flux"]))

        dr = tr.dense_residual_stats()
        summary["dense_residual"] = dr
        print("residual RMS: collocation {:.3e} | independent dense {:.3e} | relative to |n_t| {:.2f}".format(
            dr["collocation_rms_residual"], dr["rms_residual"], dr["rel_rms_residual_vs_n_t"]))

        fv, solver, sim, pinn = dg.forward_validation(tr, coarsen=args.coarsen)
        summary["forward_validation"] = {k: v for k, v in fv.items() if k != "mass_budget_simulation"}
        pd.DataFrame(fv["simulation_all"]["per_year"]).to_csv(os.path.join(out, "forward_simulation_per_year.csv"), index=False)
        print("forward PDE simulation vs reference: pooled R2 {:.3f} (PINN field at same cells: {:.3f})".format(
            fv["simulation_all"]["pooled"]["r2"], fv["pinn_all"]["pooled"]["r2"]))
        for tag in ("test_cells", "test_years"):
            if "simulation_" + tag in fv:
                print("   on {}: simulation R2 {:.3f} | PINN R2 {:.3f}".format(
                    tag, fv["simulation_" + tag]["pooled"]["r2"], fv["pinn_" + tag]["pooled"]["r2"]))

        eig = dg.eigenvalue_series(solver, fv["params"])
        summary["lambda1"] = eig
        print("lambda_1: {:+.4f} ({}) ... {:+.4f} ({})".format(eig[0]["lambda1"], eig[0]["year"],
                                                              eig[-1]["lambda1"], eig[-1]["year"]))
        summary["per_year_full_grid"] = dg.per_year_fidelity(tr)
        summary["per_year_heldout_only"] = dg.per_year_fidelity(tr, exclude_training=True)
        sup = tr.H_support()
        summary["H_support"] = {k: v for k, v in sup.items() if k != "values"}

        if not args.no_figures:
            dg.plot_mass_balance(mb, os.path.join(out, "fig_mass_balance.pdf"))
            dg.plot_trajectories(fv, os.path.join(out, "fig_trajectory_simulation.pdf"),
                                 highlight_years=list(tr.cfg.test_years) or None)
            dg.plot_eigenvalues(eig, os.path.join(out, "fig_lambda1.pdf"))
            dg.plot_response_curves(tr, os.path.join(out, "fig_response_curves.pdf"))
            dg.plot_density_maps(tr, os.path.join(out, "fig_density_maps.pdf"))
            dg.plot_calibration(tr, os.path.join(out, "fig_calibration_test.pdf"))
            dg.plot_all_years(tr, os.path.join(out, "fig_all_years.pdf"))

    write_json(summary, os.path.join(out, "summary.json"))
    b = best[1]
    print("\n=== kept seed {} ({}) ===".format(b["seed"], summary["selection_rule"]))
    print("test records ({} split): R2={:.3f} RMSE={:.2f} MAE={:.2f} corr={:.3f}".format(
        tr.cfg.split, b["test"]["r2"], b["test"]["rmse"], b["test"]["mae"], b["test"]["corr"]))
    a = summary["across_seeds"]["test_r2"]
    print("test R2 over {} seeds: {:.3f} +/- {:.3f}".format(a["n"], a["mean"], a["sd"]))
    print("raw counts (auxiliary, not independent): R2={:.3f} corr={:.3f}".format(b["counts"]["r2"], b["counts"]["corr"]))
    for k, v in b["params"].items():
        sd = summary["across_seeds"]["params"].get(k, {}).get("sd", float("nan"))
        print(f"  {k:22s} = {float(v):.4g}   (sd over seeds {sd:.3g})")
    print(f"\nresults written to {out}/")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    main()
