#!/usr/bin/env python3
"""
Field-reconstruction experiment (offline, no notebook / no Colab required).

Usage:
    python run_experiment1.py

Place the three data files in the same directory (or pass --data-dir):
    u_GLM_Bruant_jaune.mat, Map_France.mat, Yellowhammer_2002_2024.csv

Outputs metrics, recovered coefficients, and the figures into ./outputs/.
"""
import argparse, os, sys, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="directory with the data files")
    ap.add_argument("--out-dir",  default="outputs", help="where to write figures/results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # locate data files
    needed = ["u_GLM_Bruant_jaune.mat", "Map_France.mat", "Yellowhammer_2002_2024.csv"]
    for f in needed:
        p = os.path.join(args.data_dir, f)
        if not os.path.exists(p):
            sys.exit(f"ERROR: data file not found: {p}\n"
                     f"Download the data from the Yellowhammer benchmark:\n"
                     f"  https://github.com/lias-laboratory/yellowhammer-benchmark")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from pinn_yellowhammer import Config, Trainer
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config(
        layers=(3,128,128,128,128,1), fourier_m=96, fourier_sigma=5.0,
        n_collocation=5000, n_glm=60000, train_frac=0.70,
        n_ic=2500, n_boundary=1200, use_glm_data=True, fix_Kmax_tilde=1.2,
        use_bc=True, w_res=0.15, w_data=6.0, w_ic=3.0, w_bc=0.05,
        res_ramp_steps=900, adam_steps=3500, adam_lr=3e-3, lbfgs_steps=300,
        seed=args.seed,
    )
    tr = Trainer(cfg)
    tr.train(verbose=True)

    # metrics
    vg = tr.val_glm_metrics()
    with torch.no_grad():
        pred = tr.model.net(tr.val_glm["txy"]).cpu().numpy().ravel() * tr.U
    mae = float(np.mean(np.abs(pred - tr.val_glm["u"])))
    vm = tr.val_metrics()
    print("\n=== FIELD (held-out) ===")
    print(f"  R2={vg['r2']:.3f}  corr={vg['corr']:.3f}  RMSE={vg['rmse']:.2f}  MAE={mae:.2f}")
    print("=== RAW COUNTS (auxiliary) ===")
    print(f"  R2={vm['r2']:.3f}  corr={vm['corr']:.3f}  RMSE={vm['rmse']:.2f}  MAE={vm['mae']:.2f}")
    print("=== PARAMETERS ===")
    for k,v in tr.physical_params().items():
        print(f"  {k:18s} = {float(v):.4f}")

    # save weights
    torch.save(tr.model.state_dict(), os.path.join(args.out_dir, "pinn_weights.pt"))
    print(f"\nWeights saved to {args.out_dir}/pinn_weights.pt")

if __name__ == "__main__":
    main()
