"""
Synthetic data with the same file layout as the real data, generated from a
KNOWN model, for tests and smoke runs (the real data are not redistributed).

    python tests/synthetic.py --out synth_data

writes   u_GLM_2002_2024.mat   (u_GLM, Mat_lat, Mat_long)  - solution of the PDE
         Map_France.mat         (Xfr, Yfr)                  - mainland + one island
         H_covariates_2002_2024.mat (eta_hab, Mat_lat, Mat_long)
         Yellowhammer_2002_2024.csv (Year, Longitude, Latitude, EMBCIT) - Poisson counts
         truth.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TRUTH = dict(D0=300.0, alpha=0.6, rmax=0.35, beta=-0.6, Kmax=18.0, gamma=0.4)


def make(out, spacing_km=10.0, nk=23, seed=0, truth=None, reaction="source_sink",
         diffusion="fickian", n_sites=500, visits_per_site=12):
    import torch
    from habitat_field import HabitatField
    from forward_solver import FVSolver

    truth = dict(TRUTH if truth is None else truth)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    h = spacing_km * 1e3
    x_ax = 100e3 + h * np.arange(int(1100e3 / h))
    y_ax = 6.05e6 + h * np.arange(int(1100e3 / h))
    X, Y = np.meshgrid(x_ax, y_ax)
    cx, cy = 650e3, 6.6e6

    th = np.linspace(0, 2 * np.pi, 400)
    R = 430e3 * (1 + 0.08 * np.sin(3 * th) + 0.05 * np.cos(5 * th))
    isl = np.linspace(0, 2 * np.pi, 60)
    Xfr = np.r_[cx + R * np.cos(th), np.nan, 1.06e6 + 45e3 * np.cos(isl)]
    Yfr = np.r_[cy + R * np.sin(th), np.nan, 6.16e6 + 45e3 * np.sin(isl)]
    savemat(os.path.join(out, "Map_France.mat"), dict(Xfr=Xfr[None, :], Yfr=Yfr[None, :]))

    Mat_long, Mat_lat = X.copy(), Y.copy()
    corner = (np.abs(X - cx) > 520e3) & (np.abs(Y - cy) > 520e3)
    Mat_long[corner] = np.nan
    Mat_lat[corner] = np.nan

    # covariate-only habitat predictor: smooth field + north-east degradation + CLC-like steps
    base = gaussian_filter(rng.standard_normal(X.shape), 60e3 / h)
    base = base / base.std() + 0.6 * (Y - cy) / 4e5
    ne = np.exp(-(((X - 900e3) / 250e3) ** 2 + ((Y - 6.85e6) / 250e3) ** 2))
    eta = np.empty(X.shape + (nk,))
    for k in range(nk):
        step = (k >= 4) * 0.15 + (k >= 10) * 0.15 + (k >= 16) * 0.15
        eta[:, :, k] = base - 1.6 * ne * (k / (nk - 1)) - step * ne
    savemat(os.path.join(out, "H_covariates_2002_2024.mat"),
            dict(eta_hab=eta, Mat_lat=Mat_lat, Mat_long=Mat_long))

    # placeholder field to build the habitat object, then simulate the truth
    u_path = os.path.join(out, "u_GLM_2002_2024.mat")
    savemat(u_path, dict(u_GLM=np.ones(X.shape + (nk,)), Mat_lat=Mat_lat, Mat_long=Mat_long))
    hf = HabitatField(u_path, os.path.join(out, "Map_France.mat"),
                      habitat_mat=os.path.join(out, "H_covariates_2002_2024.mat"))
    sol = FVSolver(hf, coarsen=1, diffusion=diffusion, reaction=reaction)
    H0 = sol.H([2002])[:, 0].numpy()
    K0 = truth["Kmax"] * (truth["gamma"] + (1 - truth["gamma"]) * H0)
    u0 = np.clip(0.45 * K0 * (H0 - 0.25) / 0.75, 0.05, None)
    with torch.no_grad():
        sim = sol.simulate(truth, u0=u0).numpy()                 # (nk, N)
    u_GLM = np.full(X.shape + (nk,), 0.5)
    for k in range(nk):
        img = sol.image(sim[k])
        u_GLM[:, :, k] = np.where(np.isfinite(img), img, 0.5)
    savemat(u_path, dict(u_GLM=u_GLM, Mat_lat=Mat_lat, Mat_long=Mat_long))

    # Poisson counts at random sites
    ii, jj = np.where(hf.M)
    sites = rng.choice(len(ii), size=n_sites, replace=False)
    rows = []
    for s in sites:
        for k in rng.choice(nk, size=visits_per_site, replace=False):
            lam = u_GLM[ii[s], jj[s], k]
            rows.append((2002 + k, x_ax[jj[s]], y_ax[ii[s]], rng.poisson(lam)))
    import pandas as pd
    pd.DataFrame(rows, columns=["Year", "Longitude", "Latitude", "EMBCIT"]).to_csv(
        os.path.join(out, "Yellowhammer_2002_2024.csv"), index=False)
    with open(os.path.join(out, "truth.json"), "w") as f:
        json.dump(dict(params=truth, reaction=reaction, diffusion=diffusion, spacing_km=spacing_km), f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_data")
    ap.add_argument("--spacing-km", type=float, default=10.0)
    ap.add_argument("--diffusion", default="fickian")
    a = ap.parse_args()
    print("written to", make(a.out, a.spacing_km, diffusion=a.diffusion))
