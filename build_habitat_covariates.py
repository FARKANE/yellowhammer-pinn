#!/usr/bin/env python3
"""
Build a covariate-only habitat predictor eta_hab(x, y, year) on the 2 km grid
(review comment 2: the submitted H was a smoothed copy of the training target).

This is a line-by-line port of the GLM4 map construction of the benchmark
(Matlab_src/LIAS_internal_report_production.m, "Simulation of the GLM niche
model"), with two differences:

  * the Year term is NOT added      -> the national trend is no longer inside H;
  * the Longitude/Latitude terms are NOT added (use --keep-lonlat to keep them)
    -> H describes land cover and climate, not geographic position.

The intercept is irrelevant (H is standardised over the domain).

Inputs (from the benchmark's explicative_variables.zip and the R script
R_src/GLM_final_model_YH_Benchmark.R):
    Coef_GLM_final_model.txt  Indice_BIO_2002_2024.mat
    Portions_CLC2000.mat  Portions_CLC2006.mat  Portions_CLC2012.mat  Portions_CLC2018.mat
    u_GLM_2002_2024.mat (only for Mat_lat / Mat_long and the grid shape)

    python tools/build_habitat_covariates.py --data-dir data --out data/H_covariates_2002_2024.mat

Coefficient order (R formula EMBCIT ~ Longitude + Latitude + BIO1 + BIO2 + I(BIO1^2)
+ I(BIO2^2) + 9 CLC + 9 CLC^2 + Year):  26 values
    [Intercept, Lon, Lat, BIO1, BIO2, BIO1^2, BIO2^2, CLC x 9, CLC^2 x 9, Year]

Note: the GLM4 coefficients were fitted on all counts.  For a strictly
leakage-free habitat index, refit the GLM without Year on the training sites only
and pass its coefficients with --coef.
"""
import argparse
import os
import re

import numpy as np
from scipy.io import loadmat, savemat

IND_BIO = [1, 2]
IND_CLC = [18, 12, 20, 23, 2, 21, 24, 25, 29]          # 1-based layers of Portion_CLC
CLC_NAMES = ["CLC231", "CLC211", "CLC242", "CLC311", "CLC112", "CLC243", "CLC312", "CLC313", "CLC324"]


def read_coefficients(path):
    vals = []
    for line in open(path):
        tok = line.strip().strip('"').strip()
        if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", tok):
            vals.append(float(tok))
    return np.array(vals)


def clc_epoch(k):
    """Slice index k (0 = 2002) -> CLC inventory used by the benchmark."""
    return 2000 if k < 4 else 2006 if k < 10 else 2012 if k < 16 else 2018


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--coef", default="Coef_GLM_final_model.txt")
    ap.add_argument("--bio", default="Indice_BIO_2002_2024.mat")
    ap.add_argument("--clc", default="Portions_CLC{year}.mat")
    ap.add_argument("--grid", default="u_GLM_2002_2024.mat")
    ap.add_argument("--out", default="H_covariates_2002_2024.mat")
    ap.add_argument("--keep-lonlat", action="store_true")
    ap.add_argument("--check-u-glm", action="store_true",
                    help="also rebuild the full GLM4 field (with Year and Lon/Lat) and compare with u_GLM")
    a = ap.parse_args()
    P = lambda f: os.path.join(a.data_dir, f)

    coef = read_coefficients(P(a.coef))
    if len(coef) != 26:
        raise SystemExit(f"expected 26 GLM4 coefficients, found {len(coef)} in {a.coef}")
    g = loadmat(P(a.grid))
    Mat_lat, Mat_long = g["Mat_lat"], g["Mat_long"]
    ny, nx = Mat_lat.shape
    bio = loadmat(P(a.bio))
    nk = bio["bio_1"].shape[2]
    clc = {y: loadmat(P(a.clc.format(year=y)))["Portion_CLC"] for y in (2000, 2006, 2012, 2018)}

    eta = np.zeros((ny, nx, nk))
    j = 3                                                   # 0-based index of BIO1 coefficient
    for b in IND_BIO:
        eta += coef[j] * bio[f"bio_{b}"]; j += 1
    for b in IND_BIO:
        eta += coef[j] * bio[f"bio_{b}"] ** 2; j += 1
    for power in (1, 2):
        for layer in IND_CLC:
            for k in range(nk):
                eta[:, :, k] += coef[j] * clc[clc_epoch(k)][:, :, layer - 1] ** power
            j += 1
    assert j == 25, j
    lonlat = coef[1] * np.nan_to_num(Mat_long) + coef[2] * np.nan_to_num(Mat_lat)
    if a.keep_lonlat:
        eta += lonlat[:, :, None]

    savemat(a.out if os.path.isabs(a.out) else P(a.out),
            dict(eta_hab=eta, Mat_lat=Mat_lat, Mat_long=Mat_long,
                 description=("GLM4 BIO+CLC linear predictor without Year" +
                              ("" if a.keep_lonlat else " and without Longitude/Latitude"))))
    print(f"eta_hab {eta.shape} written to {a.out}; Year coefficient excluded = {coef[25]:.4g}")

    if a.check_u_glm and "u_GLM" in g:
        full = coef[0] + eta + (0 if a.keep_lonlat else lonlat[:, :, None]) + \
            coef[25] * (2002 + np.arange(nk))[None, None, :]
        ref = g["u_GLM"]
        ok = np.isfinite(ref) & (ref > 0) & np.isfinite(full)
        err = np.max(np.abs(full[ok] - np.log(ref[ok])))
        verdict = "PASS (coefficient order and covariate files match u_GLM)" if err < 1e-6 else \
            "FAIL: the rebuilt field does not match u_GLM - check the coefficient file / CLC layer indices"
        print(f"rebuilt GLM4 linear predictor vs log(u_GLM): max abs difference {err:.2e} -> {verdict}")


if __name__ == "__main__":
    main()
