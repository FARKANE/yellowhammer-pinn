#!/usr/bin/env python3
"""
Ablation of targets, architecture and PDE weight (Table 3 of the manuscript).

Every field row is evaluated on the SAME held-out test records (same seed, same
split).  The "no physics" row is scored on held-out records too (version 1
reported its TRAINING R2, which is not comparable).

    python run_ablation.py --data-dir data                 # recommended protocol
    python run_ablation.py --data-dir data --preset v1     # version-1 protocol
"""
import argparse
import os
from dataclasses import replace

import pandas as pd

from experiment_utils import add_common_args, config_from_args
from pinn_yellowhammer import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fourier-m", type=int, default=96)
    args = ap.parse_args()
    base = config_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    lam = base.w_res
    fm = base.fourier_m if args.quick else args.fourier_m
    rows_def = [
        ("A", "Counts, plain MLP", dict(use_glm_data=False, fourier_m=0, w_res=1.0)),
        ("B", "Field, plain MLP", dict(fourier_m=0, w_res=1.0)),
        ("NP", "Field, Fourier, no physics", dict(fourier_m=fm, w_res=0.0, w_bc=0.0)),
        ("C", "Field, Fourier, lambda_f=1", dict(fourier_m=fm, w_res=1.0)),
        ("D", f"Field, Fourier, lambda_f={lam:g} (adopted)", dict(fourier_m=fm, w_res=lam)),
    ]
    out = []
    for s in args.seeds:
        for tag, name, kw in rows_def:
            tr = Trainer(replace(base, seed=s, **kw))
            tr.train(verbose=args.verbose)
            if kw.get("use_glm_data", True):
                test = tr.val_glm_metrics(); train = tr.train_metrics(); target = "field (held-out test)"
            else:
                test = tr.val_metrics(); train = None; target = "counts (held-out 15 %)"
            p = tr.physical_params()
            out.append(dict(seed=s, row=tag, configuration=name, target=target, lambda_f=kw["w_res"],
                            test_r2=test["r2"], test_rmse=test["rmse"], train_r2=(train or {}).get("r2"),
                            D0_km2_per_yr=p["D0_km2_per_yr"], rmax_per_yr=p["rmax_per_yr"], beta=p["beta"],
                            Kmax=p["Kmax_birds"], gamma=p["gamma"]))
            print(f"seed {s} {tag:3s} {name:38s} test R2={test['r2']:.3f}  "
                  f"train R2={'%.3f' % train['r2'] if train else '  -  '}  rmax={p['rmax_per_yr']:.4f}")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(args.out_dir, "ablation.csv"), index=False)
    agg = df.groupby(["row", "configuration"], sort=False)[["test_r2", "train_r2"]].agg(["mean", "std"])
    print("\n" + agg.round(3).to_string())
    print("\nLaTeX rows (mean test R2 over seeds):")
    for (tag, name), r in agg.iterrows():
        print(f"{tag} & {name} & {r[('test_r2', 'mean')]:.2f} \\\\")


if __name__ == "__main__":
    main()
