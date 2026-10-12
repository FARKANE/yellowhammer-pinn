"""Command-line options and configuration builder shared by the run_*.py scripts."""

import argparse
import os
import sys
import warnings
from dataclasses import replace

from pinn_yellowhammer import paper_config, revised_config

BENCHMARK_URL = "https://github.com/lias-laboratory/yellowhammer-benchmark"

QUICK = dict(n_collocation=600, n_ic=300, n_boundary=200, n_glm=4000,
             layers=(3, 32, 32, 1), fourier_m=16, adam_steps=150, lbfgs_steps=10,
             res_ramp_steps=50)


def parse_years(text):
    """'2019-2024' or '2019,2020,2021' -> tuple of ints."""
    if not text:
        return ()
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def add_common_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("data")
    g.add_argument("--data-dir", default=".", help="directory with the data files")
    g.add_argument("--out-dir", default="outputs")
    g.add_argument("--u-mat", default="u_GLM_2002_2024.mat")
    g.add_argument("--map-mat", default="Map_France.mat")
    g.add_argument("--csv", default="Yellowhammer_2002_2024.csv")
    g.add_argument("--habitat-mat", default="",
                   help="covariate-only habitat index (.mat with eta_hab). Without it, H is derived "
                        "from u_GLM itself (circular, legacy).")
    g.add_argument("--habitat-var", default="eta_hab")
    g.add_argument("--year1", type=int, default=None, help="end of the PDE time window (default 2024)")

    m = ap.add_argument_group("model / protocol")
    m.add_argument("--preset", choices=["paper", "revised"], default="revised",
                   help="paper = submitted pipeline (incl. its defects); revised = corrected protocol")
    m.add_argument("--reaction", choices=["logistic", "source_sink"], default=None)
    m.add_argument("--beta-min", type=float, default=None)
    m.add_argument("--diffusion", choices=["fickian", "ecological"], default=None)
    m.add_argument("--split", choices=["random", "block", "temporal"], default=None)
    m.add_argument("--block-km", type=float, default=None)
    m.add_argument("--test-years", default=None, help="temporal split, e.g. 2019-2024")
    m.add_argument("--select-frac", type=float, default=None)
    m.add_argument("--lambda-f", type=float, default=None, help="PDE residual weight (w_res)")
    m.add_argument("--kmax-tilde", type=float, default=None, help="fixed dimensionless Kmax (U=15 -> Kmax=15*value)")

    o = ap.add_argument_group("optimisation")
    o.add_argument("--adam-steps", type=int, default=None)
    o.add_argument("--lbfgs-steps", type=int, default=None)
    o.add_argument("--n-glm", type=int, default=None)
    o.add_argument("--n-collocation", type=int, default=None)
    o.add_argument("--quick", action="store_true", help="tiny settings (smoke test only)")
    o.add_argument("--device", default=None)
    o.add_argument("--verbose", action="store_true")
    return ap


def config_from_args(args, **extra):
    builder = paper_config if args.preset == "paper" else revised_config
    p = lambda name: os.path.join(args.data_dir, name)
    kw = dict(u_mat=p(args.u_mat), map_mat=p(args.map_mat), csv=p(args.csv),
              habitat_mat=p(args.habitat_mat) if args.habitat_mat else "",
              habitat_var=args.habitat_var)
    if args.quick:
        kw.update(QUICK)
    mapping = dict(year1="year1", reaction="reaction", beta_min="beta_min", diffusion="diffusion",
                   split="split", block_km="block_km", select_frac="select_frac",
                   lambda_f="w_res", kmax_tilde="fix_Kmax_tilde", adam_steps="adam_steps",
                   lbfgs_steps="lbfgs_steps", n_glm="n_glm", n_collocation="n_collocation",
                   device="device")
    for a, c in mapping.items():
        v = getattr(args, a, None)
        if v is not None:
            kw[c] = v
    if args.test_years:
        kw["test_years"] = parse_years(args.test_years)
        kw.setdefault("split", "temporal")
    kw.update(extra)
    cfg = builder(**kw)
    check_files(cfg)
    if not cfg.habitat_mat:
        warnings.warn("Habitat index H is derived from u_GLM (the training target): coefficient "
                      "estimates are conditional on this circular construction. Pass --habitat-mat "
                      "(see tools/build_habitat_covariates.py) for a covariate-only index.")
    return cfg


def check_files(cfg):
    missing = [f for f in (cfg.u_mat, cfg.map_mat, cfg.csv, cfg.habitat_mat) if f and not os.path.exists(f)]
    if missing:
        sys.exit("ERROR: data file(s) not found: %s\n"
                 "u_GLM_2002_2024.mat is the GLM4 niche-model field (23 annual slices) produced by\n"
                 "Matlab_src/LIAS_internal_report_production.m of the benchmark; Map_France.mat and the\n"
                 "CSV come from the benchmark itself:\n  %s" % (missing, BENCHMARK_URL))


def with_seed(cfg, seed):
    return replace(cfg, seed=int(seed))
