"""
pinn_yellowhammer.py
====================
Physics-Informed Neural Network for the Yellowhammer population density field
u(t, x, y) over mainland France (2002-2024), with simultaneous identification of
the habitat-response coefficients theta = (D0, alpha, rmax, beta, Kmax, gamma).

Governing PDE (habitat-modulated reaction-diffusion, Neumann no-flux on dOmega):

    du/dt = Diff[u] + f(H, u)                       (+ optional taxis)
      a(H) = D0 (1 - alpha H)
      r(H) = rmax (beta + (1 - beta) H)
      K(H) = Kmax (gamma + (1 - gamma) H)

    Diff[u] = div(a(H) grad u)        diffusion="fickian"   (paper, eq. 39)
            = lap(a(H) u)             diffusion="ecological" (Turchin 1998; Garlick 2011)

    f(H, u) = r(H) u (1 - u / K(H))   reaction="logistic"   (paper; beta in (0, 1))
            = u (r(H) - rmax u / K(H)) reaction="source_sink" (beta in (beta_min, 1):
              r(H) < 0 below H* = -beta/(1-beta) -> sink habitat; crowding rmax/K(H) > 0
              keeps the model bounded, Cantrell & Cosner 2003 form u(m - c u))

Why source_sink: with reaction="logistic", b = 0 and no-flux boundaries,
d/dt int u = int r u (1 - u/K) > 0 whenever u < K, so the model cannot produce the
observed national decline; the decline is then carried by the residual or by
boundary leakage (see Trainer.mass_balance).

Protocol switches (defaults = corrected protocol; v1_config() = version 1 of this code):
  sample_replace  False: GLM records drawn WITHOUT replacement (legacy True could
                  put the same record in train and test).
  ic_source       "train_nodes": IC targets are exact u_GLM(2002) values at grid
                  nodes outside the test set (legacy "lhs" interpolated slice 0 at
                  LHS points, which overlaps test records of 2002).
  bc_time         "uniform": no-flux points are spread over [t0, t1] (legacy "mid"
                  put every boundary point at t = 0.5, i.e. year 2013 only).
  split           "random" | "block" (spatial blocks of block_km) | "temporal".
  select_frac     fraction of the TRAINING records held out to select seeds and
                  lambda_f, so the test set is never used for selection.
"""

import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from habitat_field import HabitatField

PARAM_NAMES = ("D0", "alpha", "rmax", "beta", "Kmax", "gamma")


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    u_mat: str = "u_GLM_2002_2024.mat"
    map_mat: str = "Map_France.mat"
    csv: str = "Yellowhammer_2002_2024.csv"
    habitat_mat: str = ""               # covariate-only index (.mat); "" -> legacy u_GLM-derived H
    habitat_var: str = "eta_hab"

    year0: int = 2002
    year1: int = 2024
    U: float = 15.0                     # density scale (u_sat)

    # model
    reaction: str = "logistic"          # "logistic" | "source_sink"
    beta_min: float = -1.0              # lower bound of beta when reaction="source_sink"
    diffusion: str = "fickian"          # "fickian" | "ecological"
    use_advection: bool = False         # taxis -div(chi u grad H)

    # network
    layers: tuple = (3, 64, 64, 64, 64, 1)
    fourier_m: int = 0
    fourier_sigma: float = 3.0

    # sampling / protocol
    n_collocation: int = 20000
    n_ic: int = 4000
    n_boundary: int = 2000
    use_bc: bool = True
    max_data: int = 0                   # counts path only
    val_frac: float = 0.15              # counts path only
    use_glm_data: bool = True
    n_glm: int = 15000
    train_frac: float = 0.70
    split: str = "random"               # "random" | "block" | "temporal"
    block_km: float = 100.0
    test_years: tuple = ()              # temporal split, e.g. (2019, ..., 2024)
    select_frac: float = 0.0            # selection subset carved from training records
    sample_replace: bool = False        # legacy: True
    ic_source: str = "train_nodes"      # legacy: "lhs"
    bc_time: str = "uniform"            # legacy: "mid"

    # parameters (dimensionless). fixed: any of PARAM_NAMES -> value
    fix_Kmax_tilde: float = 1.2         # >0 fixes Kmax
    fix_D0_tilde: float = -1.0          # >=0 fixes D0
    fixed: dict = field(default_factory=dict)

    # loss weights
    w_res: float = 1.0
    w_data: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 0.1
    res_ramp_steps: int = 1500
    w_prior: float = 0.0
    D0_tilde_prior: float = -1.0

    # optimisation
    adam_steps: int = 8000
    adam_lr: float = 2e-3
    lbfgs_steps: int = 1500
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fixed_params(self):
        f = dict(self.fixed)
        if self.fix_Kmax_tilde > 0:
            f.setdefault("Kmax", float(self.fix_Kmax_tilde))
        if self.fix_D0_tilde >= 0:
            f.setdefault("D0", float(self.fix_D0_tilde))
        bad = set(f) - set(PARAM_NAMES)
        if bad:
            raise ValueError(f"unknown fixed parameters {bad}")
        return f

    def to_dict(self):
        d = asdict(self)
        d["layers"] = list(d["layers"])
        d["test_years"] = list(d["test_years"])
        return d


# Network, sampling and optimisation settings (Table 2 of the manuscript)
BASE_SETTINGS = dict(
    n_collocation=5000, n_ic=2500, n_boundary=1200, use_glm_data=True,
    n_glm=60000, train_frac=0.70, fix_Kmax_tilde=1.2,
    layers=(3, 128, 128, 128, 128, 1), fourier_m=96, fourier_sigma=5.0,
    w_res=0.15, w_data=6.0, w_ic=3.0, w_bc=0.05, res_ramp_steps=900,
    adam_steps=5000, adam_lr=3e-3, lbfgs_steps=500)

V1_PROTOCOL = dict(sample_replace=True, ic_source="lhs", bc_time="mid",
                       split="random", select_frac=0.0,
                       reaction="logistic", diffusion="fickian")

RECOMMENDED_PROTOCOL = dict(sample_replace=False, ic_source="train_nodes", bc_time="uniform",
                        split="block", block_km=100.0, select_frac=0.10,
                        reaction="source_sink", beta_min=-1.0, diffusion="fickian")


def v1_config(**overrides):
    """Exactly the version-1 pipeline of this repository (including its sampling/boundary defects)."""
    return Config(**{**BASE_SETTINGS, **V1_PROTOCOL, **overrides})


def recommended_config(**overrides):
    """Recommended configuration (corrected protocol, source-sink reaction, block split)."""
    return Config(**{**BASE_SETTINGS, **RECOMMENDED_PROTOCOL, **overrides})


# --------------------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, layers, fourier_m=0, fourier_sigma=3.0):
        super().__init__()
        self.fourier_m = fourier_m
        in_dim = layers[0]
        if fourier_m > 0:
            B = torch.randn(2, fourier_m) * fourier_sigma       # Fourier features on (x, y)
            self.register_buffer("Bff", B)
            in_dim = layers[0] + 2 * fourier_m
        seq, dims = [], [in_dim] + list(layers[1:])
        for i in range(len(dims) - 2):
            seq += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        seq += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*seq)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, txy):
        if self.fourier_m > 0:
            proj = 2 * np.pi * txy[:, 1:3] @ self.Bff
            feats = torch.cat([txy, torch.sin(proj), torch.cos(proj)], dim=1)
        else:
            feats = txy
        return torch.nn.functional.softplus(self.net(feats))   # density >= 0


class PINN(nn.Module):
    """Field network + constrained dimensionless parameters.
    State-dict keys are those of version 1, so old weights load."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(cfg.layers, cfg.fourier_m, cfg.fourier_sigma)
        self.fixed = cfg.fixed_params()
        self.pD = nn.Parameter(torch.tensor(-2.0))     # D0~ = softplus(pD)
        self.pa = nn.Parameter(torch.tensor(0.0))      # alpha = sigmoid(pa)
        self.pr = nn.Parameter(torch.tensor(0.0))      # rmax~ = softplus(pr)
        self.pb = nn.Parameter(torch.tensor(0.0))      # beta  = sigmoid(pb) [mapped for source_sink]
        if "Kmax" not in self.fixed:
            self.pK = nn.Parameter(torch.tensor(0.0))  # Kmax~ = softplus(pK) + 0.05
        self.pg = nn.Parameter(torch.tensor(0.0))      # gamma = sigmoid(pg)
        if cfg.use_advection:
            self.pchi = nn.Parameter(torch.tensor(-2.0))
        buf_name = {"Kmax": "Kfix", "D0": "Dfix"}
        for k, v in self.fixed.items():
            self.register_buffer(buf_name.get(k, "fix_" + k), torch.tensor(float(v)))

    def _fix(self, name):
        return getattr(self, {"Kmax": "Kfix", "D0": "Dfix"}.get(name, "fix_" + name))

    def params(self):
        sp, sg = torch.nn.functional.softplus, torch.sigmoid
        if self.cfg.reaction == "source_sink":
            bmin = self.cfg.beta_min
            beta = bmin + (1.0 - bmin) * sg(self.pb)
        else:
            beta = sg(self.pb)
        free = dict(D0=lambda: sp(self.pD), alpha=lambda: sg(self.pa), rmax=lambda: sp(self.pr),
                    beta=lambda: beta, Kmax=lambda: sp(self.pK) + 0.05, gamma=lambda: sg(self.pg))
        p = {k: (self._fix(k) if k in self.fixed else free[k]()) for k in PARAM_NAMES}
        if self.cfg.use_advection:
            p["chi"] = sp(self.pchi)
        return p


def coefficient_curves(p, H, reaction="logistic"):
    """a(H), r(H), K(H) for numpy or torch inputs (dimensionless parameters)."""
    a = p["D0"] * (1.0 - p["alpha"] * H)
    r = p["rmax"] * (p["beta"] + (1.0 - p["beta"]) * H)
    K = p["Kmax"] * (p["gamma"] + (1.0 - p["gamma"]) * H)
    return a, r, K


def _r2(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    return float(1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


# --------------------------------------------------------------------------- driver
class Trainer:
    def __init__(self, cfg: Config, build_points=True):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.dev = torch.device(cfg.device)
        self.need_lap = cfg.diffusion == "ecological" or cfg.use_advection
        if cfg.reaction not in ("logistic", "source_sink"):
            raise ValueError("reaction must be 'logistic' or 'source_sink'")
        if cfg.diffusion not in ("fickian", "ecological"):
            raise ValueError("diffusion must be 'fickian' or 'ecological'")

        self.hf = HabitatField(cfg.u_mat, cfg.map_mat, year0=cfg.year0,
                               habitat_mat=cfg.habitat_mat or None,
                               habitat_var=cfg.habitat_var,
                               with_laplacian=self.need_lap)
        self.x0, self.y0 = self.hf.x_min, self.hf.y_min
        self.L = max(self.hf.x_max - self.hf.x_min, self.hf.y_max - self.hf.y_min)
        self.T = float(cfg.year1 - cfg.year0)
        self.U = cfg.U

        self.model = PINN(cfg).to(self.dev)
        self.hist, self.param_hist = [], []
        if build_points:
            self._build_points()

    # ---- normalisation helpers
    def to_norm(self, year, x, y):
        return ((np.asarray(year, float) - self.cfg.year0) / self.T,
                (np.asarray(x, float) - self.x0) / self.L,
                (np.asarray(y, float) - self.y0) / self.L)

    def _tensor(self, a):
        return torch.tensor(np.asarray(a, np.float64), dtype=torch.float32, device=self.dev)

    def _txy(self, year, x, y):
        tn, xn, yn = self.to_norm(year, x, y)
        return self._tensor(np.column_stack([np.broadcast_to(tn, np.shape(xn)), xn, yn]))

    def _habitat(self, year, x, y):
        """Habitat coefficients at points, in normalised units (dict of tensors)."""
        q = self.hf.query(year, x, y, laplacian=self.need_lap)
        out = dict(H=self._tensor(q[0]), Hx=self._tensor(q[1] * self.L), Hy=self._tensor(q[2] * self.L))
        if self.need_lap:
            out["lapH"] = self._tensor(q[3] * self.L ** 2)
        return out

    # ---- point sets
    def _lhs_inside(self, n, t_lo, t_hi, seed=None):
        """Latin-hypercube (year, x, y) rejected to inside France."""
        out_y, out_x, out_t = [], [], []
        eng = qmc.LatinHypercube(d=3, seed=self.cfg.seed if seed is None else seed)
        need = n
        while need > 0:
            s = eng.random(max(need * 3, 1000))
            yr = self.cfg.year0 + s[:, 0] * self.T
            xx = self.hf.x_min + s[:, 1] * (self.hf.x_max - self.hf.x_min)
            yy = self.hf.y_min + s[:, 2] * (self.hf.y_max - self.hf.y_min)
            if t_lo == t_hi:
                yr[:] = t_lo
            m = self.hf.inside(xx, yy)
            out_t.append(yr[m]); out_x.append(xx[m]); out_y.append(yy[m])
            need = n - sum(len(a) for a in out_t)
        return (np.concatenate(out_t)[:n], np.concatenate(out_x)[:n], np.concatenate(out_y)[:n])

    def _block_ids(self, x, y):
        B = 1000.0 * self.cfg.block_km
        bx = np.floor((np.asarray(x) - self.hf.x_min) / B).astype(np.int64)
        by = np.floor((np.asarray(y) - self.hf.y_min) / B).astype(np.int64)
        return bx * 100000 + by

    def _split(self, idx, x, y, year, keep_frac, mode, seed):
        """Split record indices idx into (keep, held); ~keep_frac of records kept.
        random: legacy formula (perm[:int(keep_frac*N)]); block: whole spatial blocks
        are held out until ~(1-keep_frac) of the records; temporal: test_years held."""
        rng = np.random.default_rng(seed)
        if mode == "random":
            perm = rng.permutation(len(idx))
            n_keep = int(keep_frac * len(idx))
            return idx[perm[:n_keep]], idx[perm[n_keep:]]
        if mode == "block":
            bid = self._block_ids(x[idx], y[idx])
            blocks = rng.permutation(np.unique(bid))
            counts = {b: c for b, c in zip(*np.unique(bid, return_counts=True))}
            held, n_held, target = [], 0, (1.0 - keep_frac) * len(idx)
            for b in blocks:
                if n_held >= target:
                    break
                held.append(b); n_held += counts[b]
            is_held = np.isin(bid, held)
            return idx[~is_held], idx[is_held]
        if mode == "temporal":
            is_held = np.isin(year[idx], np.asarray(self.cfg.test_years, float))
            return idx[~is_held], idx[is_held]
        raise ValueError(f"unknown split '{mode}'")

    def _build_points(self):
        c = self.cfg
        # ----- collocation
        t, x, y = self._lhs_inside(c.n_collocation, c.year0, c.year1)
        self.col = dict(txy=self._txy(t, x, y), **self._habitat(t, x, y))
        self.col["txy"].requires_grad_(True)

        # ----- data
        import pandas as pd
        df = pd.read_csv(c.csv)
        if c.use_glm_data:
            rec = self.hf.sample_glm_records(c.n_glm, seed=c.seed, replace=c.sample_replace)
            self.rec = rec
            allidx = np.arange(len(rec["u"]))
            if c.split == "temporal" and not c.test_years:
                raise ValueError("split='temporal' requires test_years")
            tr, te = self._split(allidx, rec["x"], rec["y"], rec["year"], c.train_frac, c.split, c.seed)
            sel = np.array([], dtype=np.int64)
            if c.select_frac > 0:
                mode = "random" if c.split == "random" else "block"
                tr, sel = self._split(tr, rec["x"], rec["y"], rec["year"], 1.0 - c.select_frac, mode, c.seed + 1)
            self.idx_train, self.idx_test, self.idx_select = tr, te, sel
            self.test_blocks = (np.unique(self._block_ids(rec["x"][te], rec["y"][te]))
                                if c.split == "block" else None)
            mk = lambda ii: dict(txy=self._txy(rec["year"][ii], rec["x"][ii], rec["y"][ii]),
                                 n=self._tensor(rec["u"][ii] / self.U).reshape(-1, 1), u=rec["u"][ii])
            self.data = mk(tr)
            self.val_glm = mk(te)                       # held-out TEST records
            self.sel = mk(sel) if len(sel) else None    # selection records (seed / lambda choice)

            # all counts: auxiliary check (NOT independent: GLM4 was fitted on them)
            self.val = dict(txy=self._txy(df.Year.values, df.Longitude.values, df.Latitude.values),
                            n=self._tensor(df.EMBCIT.values / self.U).reshape(-1, 1),
                            counts=df.EMBCIT.values.astype(float))
        else:
            if c.max_data and len(df) > c.max_data:
                df = df.sample(c.max_data, random_state=c.seed)
            df = df.sample(frac=1.0, random_state=c.seed).reset_index(drop=True)
            n_val = int(c.val_frac * len(df))
            df_val, df_tr = df.iloc[:n_val], df.iloc[n_val:]
            self.data = dict(txy=self._txy(df_tr.Year.values, df_tr.Longitude.values, df_tr.Latitude.values),
                             n=self._tensor(df_tr.EMBCIT.values / self.U).reshape(-1, 1))
            self.val = dict(txy=self._txy(df_val.Year.values, df_val.Longitude.values, df_val.Latitude.values),
                            n=self._tensor(df_val.EMBCIT.values / self.U).reshape(-1, 1),
                            counts=df_val.EMBCIT.values.astype(float))
            self.sel = None

        # ----- initial condition (t = year0)
        self._build_ic()
        # ----- boundary (Neumann)
        if c.use_bc:
            self._build_boundary()

    def _build_ic(self):
        c = self.cfg
        if c.ic_source == "lhs" or not c.use_glm_data:
            t, x, y = self._lhs_inside(c.n_ic, c.year0, c.year0)
            g = self.hf.g_initial(x, y)
        elif c.ic_source == "train_nodes":
            if c.split == "temporal" and c.year0 in tuple(c.test_years):
                raise ValueError("year0 is a test year: no initial condition can be used")
            ii, jj = np.where(self.hf.M)
            excl = np.zeros(len(ii), dtype=bool)
            if c.split == "block":
                excl |= np.isin(self._block_ids(self.hf.lon_ax[jj], self.hf.lat_ax[ii]), self.test_blocks)
            # never use a held-out (test or selection) record of slice 0
            held = np.concatenate([self.idx_test, self.idx_select])
            k0 = held[self.rec["k"][held] == 0]
            if len(k0):
                key = ii.astype(np.int64) * self.hf.nx + jj
                excl |= np.isin(key, self.rec["I"][k0].astype(np.int64) * self.hf.nx + self.rec["J"][k0])
            cand = np.where(~excl)[0]
            rng = np.random.default_rng(c.seed + 3)
            take = rng.choice(cand, size=min(c.n_ic, len(cand)), replace=False)
            x, y = self.hf.lon_ax[jj[take]], self.hf.lat_ax[ii[take]]
            t = np.full(len(take), float(c.year0))
            g = self.hf.u_GLM[ii[take], jj[take], 0]
        else:
            raise ValueError(f"unknown ic_source '{c.ic_source}'")
        self.ic = dict(txy=self._txy(t, x, y), n=self._tensor(g / self.U).reshape(-1, 1))
        self.ic_xy = (np.asarray(x, float), np.asarray(y, float))     # physical, float64

    def _build_boundary(self):
        c = self.cfg
        if c.bc_time == "mid":                         # version-1 construction
            Xfr, Yfr = self.hf.Xfr, self.hf.Yfr
            finite = np.isfinite(Xfr) & np.isfinite(Yfr)
            xs, ys = Xfr[finite], Yfr[finite]
            tx = np.gradient(xs); ty = np.gradient(ys)
            nrm = np.hypot(tx, ty) + 1e-9
            nx, ny = ty / nrm, -tx / nrm
            eps = 1500.0
            test_in = self.hf.inside(xs + eps * nx, ys + eps * ny)
            nx = np.where(test_in, nx, -nx); ny = np.where(test_in, ny, -ny)
            idx = np.random.choice(len(xs), min(c.n_boundary, len(xs)), replace=False)
            bx, by = xs[idx] + eps * nx[idx], ys[idx] + eps * ny[idx]
            bnx, bny = nx[idx], ny[idx]
            keep = self.hf.inside(bx, by)
            bx, by, bnx, bny = bx[keep], by[keep], bnx[keep], bny[keep]
            years = np.full(len(bx), c.year0 + self.T / 2)   # <- every point at t = 0.5
        elif c.bc_time == "uniform":
            bp = self.hf.boundary_points()
            rng = np.random.default_rng(c.seed + 2)
            take = rng.choice(len(bp["x"]), size=min(c.n_boundary, len(bp["x"])), replace=False)
            bx, by, bnx, bny = bp["x"][take], bp["y"][take], bp["nx"][take], bp["ny"][take]
            years = rng.uniform(c.year0, c.year1, size=len(take))
        else:
            raise ValueError(f"unknown bc_time '{c.bc_time}'")
        self.bc = dict(txy=self._txy(years, bx, by),
                       nrm=self._tensor(np.column_stack([bnx, bny])),
                       **self._habitat(years, bx, by))
        self.bc["txy"].requires_grad_(True)

    # ---- derivatives & residual
    @staticmethod
    def _grads(n, txy):
        g = torch.autograd.grad(n, txy, torch.ones_like(n), create_graph=True)[0]
        return g[:, 0:1], g[:, 1:2], g[:, 2:3]

    def pde_terms(self, txy, hab):
        """All terms of the normalised PDE at points txy (requires_grad=True).
        Returns n, n_t, diffusion, reaction, advection, residual (column tensors)."""
        p = self.model.params()
        n = self.model.net(txy)
        n_t, n_x, n_y = self._grads(n, txy)
        n_xx = torch.autograd.grad(n_x, txy, torch.ones_like(n_x), create_graph=True)[0][:, 1:2]
        n_yy = torch.autograd.grad(n_y, txy, torch.ones_like(n_y), create_graph=True)[0][:, 2:3]
        H = hab["H"].reshape(-1, 1)
        Hx, Hy = hab["Hx"].reshape(-1, 1), hab["Hy"].reshape(-1, 1)
        gH_gn = Hx * n_x + Hy * n_y

        a, r, K = coefficient_curves(p, H)
        a_prime = -p["D0"] * p["alpha"]
        if self.cfg.diffusion == "fickian":
            diff = a * (n_xx + n_yy) + a_prime * gH_gn
        else:                                                   # lap(a n), a linear in H
            lapH = hab["lapH"].reshape(-1, 1)
            diff = a * (n_xx + n_yy) + 2.0 * a_prime * gH_gn + a_prime * lapH * n

        if self.cfg.reaction == "logistic":
            react = r * n * (1.0 - n / K)
        else:
            react = n * (r - p["rmax"] * n / K)

        adv = torch.zeros_like(n)
        if self.cfg.use_advection:                              # -div(chi n grad H)
            lap_term = hab["lapH"].reshape(-1, 1) * n if "lapH" in hab else 0.0
            adv = -p["chi"] * (gH_gn + lap_term)
        res = n_t - diff - react - adv
        return dict(n=n, n_t=n_t, diffusion=diff, reaction=react, advection=adv, residual=res,
                    n_x=n_x, n_y=n_y)

    def residual(self):
        return self.pde_terms(self.col["txy"], self.col)["residual"]

    def _bc_flux(self, txy, nrm, hab):
        """Normal derivative entering the no-flux condition (normalised units).
        fickian: grad n . nu ;  ecological: grad(a n) . nu / a."""
        nb = self.model.net(txy)
        g = torch.autograd.grad(nb, txy, torch.ones_like(nb), create_graph=True)[0]
        dn = g[:, 1:2] * nrm[:, 0:1] + g[:, 2:3] * nrm[:, 1:2]
        if self.cfg.diffusion == "ecological":
            p = self.model.params()
            H = hab["H"].reshape(-1, 1)
            dH = hab["Hx"].reshape(-1, 1) * nrm[:, 0:1] + hab["Hy"].reshape(-1, 1) * nrm[:, 1:2]
            dn = dn - nb * p["alpha"] / (1.0 - p["alpha"] * H) * dH     # a'/a = -alpha/(1-alpha H)
        return dn

    def losses(self, step=None):
        c = self.cfg
        L_res = (self.residual() ** 2).mean()
        L_data = ((self.model.net(self.data["txy"]) - self.data["n"]) ** 2).mean()
        L_ic = ((self.model.net(self.ic["txy"]) - self.ic["n"]) ** 2).mean()
        if c.use_bc:
            L_bc = (self._bc_flux(self.bc["txy"], self.bc["nrm"], self.bc) ** 2).mean()
        else:
            L_bc = torch.tensor(0.0, device=self.dev)
        wr = c.w_res
        if step is not None and c.res_ramp_steps > 0:
            wr = c.w_res * min(1.0, step / c.res_ramp_steps)
        total = wr * L_res + c.w_data * L_data + c.w_ic * L_ic + c.w_bc * L_bc
        L_prior = torch.tensor(0.0, device=self.dev)
        if c.w_prior > 0 and c.D0_tilde_prior >= 0:
            L_prior = (self.model.params()["D0"] - c.D0_tilde_prior) ** 2
            total = total + c.w_prior * L_prior
        return total, dict(res=L_res.item(), data=L_data.item(), ic=L_ic.item(),
                           bc=float(L_bc.detach()), prior=float(L_prior.detach()),
                           total=float(total.detach()))

    def objective_terms(self):
        """Loss components at full residual weight (the profile-likelihood objective)."""
        _, log = self.losses(step=self.cfg.res_ramp_steps)
        return log

    def _param_snapshot(self):
        return {k: float(v.detach()) for k, v in self.model.params().items()}

    # ---- training
    def train(self, verbose=True):
        c = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(c.adam_steps, 1), eta_min=c.adam_lr * 0.02)
        hist, self.param_hist = [], []
        for step in range(c.adam_steps):
            opt.zero_grad()
            total, log = self.losses(step)
            total.backward()
            opt.step(); sched.step()
            if verbose and (step % max(1, c.adam_steps // 20) == 0 or step == c.adam_steps - 1):
                p = self._param_snapshot()
                print(f"[adam {step:5d}] tot={log['total']:.3e} res={log['res']:.2e} "
                      f"data={log['data']:.2e} ic={log['ic']:.2e} bc={log['bc']:.2e} "
                      f"| D0~={p['D0']:.3f} a={p['alpha']:.2f} r~={p['rmax']:.3f} "
                      f"b={p['beta']:.2f} K~={p['Kmax']:.2f} g={p['gamma']:.2f}")
            hist.append(log)
            if step % 25 == 0:
                self.param_hist.append((step, self._param_snapshot()))

        if c.lbfgs_steps > 0:
            opt2 = torch.optim.LBFGS(self.model.parameters(), max_iter=c.lbfgs_steps,
                                     history_size=50, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-9, tolerance_change=1e-12)

            def closure():
                opt2.zero_grad()
                total, _ = self.losses(step=c.res_ramp_steps)
                total.backward()
                return total
            opt2.step(closure)
            _, log = self.losses(step=c.res_ramp_steps)
            if verbose:
                p = self._param_snapshot()
                print(f"[lbfgs done] tot={log['total']:.3e} res={log['res']:.2e} "
                      f"data={log['data']:.2e} ic={log['ic']:.2e} | "
                      f"D0~={p['D0']:.3f} r~={p['rmax']:.3f} K~={p['Kmax']:.2f}")
            hist.append(log)
        self.hist = hist
        self.final_objective = self.objective_terms()
        return hist

    # ---- metrics
    @torch.no_grad()
    def predict(self, year, x, y):
        """Density u (birds/site) at physical points."""
        return self.model.net(self._txy(year, x, y)).cpu().numpy().ravel() * self.U

    @torch.no_grad()
    def _field_metrics(self, d):
        if d is None:
            return None
        pred = self.model.net(d["txy"]).cpu().numpy().ravel() * self.U
        u = d["u"]
        return dict(rmse=float(np.sqrt(np.mean((pred - u) ** 2))), mae=float(np.mean(np.abs(pred - u))),
                    r2=_r2(u, pred), corr=float(np.corrcoef(pred, u)[0, 1]), n=int(len(u)))

    def val_glm_metrics(self):
        """Held-out TEST records of the GLM field."""
        return self._field_metrics(getattr(self, "val_glm", None))

    def sel_metrics(self):
        """Selection records (carved from training) - use these to choose seeds / lambda_f."""
        return self._field_metrics(getattr(self, "sel", None))

    def train_metrics(self):
        d = self.data
        if "u" not in d:
            return None
        return self._field_metrics(d)

    @torch.no_grad()
    def val_metrics(self):
        """Raw counts (auxiliary; the GLM behind u_GLM was fitted on these counts)."""
        pred = self.model.net(self.val["txy"]).cpu().numpy().ravel() * self.U
        obs = self.val["counts"]
        return dict(rmse=float(np.sqrt(np.mean((pred - obs) ** 2))), mae=float(np.mean(np.abs(pred - obs))),
                    r2=_r2(obs, pred), corr=float(np.corrcoef(pred, obs)[0, 1]), pred=pred, obs=obs)

    # ---- physical parameter recovery
    def physical_params(self):
        p = self._param_snapshot()
        L_km = self.L / 1000.0
        out = dict(D0_km2_per_yr=p["D0"] * L_km ** 2 / self.T, alpha=p["alpha"],
                   rmax_per_yr=p["rmax"] / self.T, beta=p["beta"],
                   Kmax_birds=p["Kmax"] * self.U, gamma=p["gamma"])
        if "chi" in p:
            out["chi_km2_per_yr"] = p["chi"] * L_km ** 2 / self.T
        if self.cfg.reaction == "source_sink" and p["beta"] < 0:
            out["H_star_sink_threshold"] = -p["beta"] / (1.0 - p["beta"])
        return out

    def response_curves(self, H=None):
        """Physical a(H) [km^2/yr], r(H) [1/yr], K(H) [birds/site]."""
        H = np.linspace(0, 1, 201) if H is None else np.asarray(H, float)
        p = self._param_snapshot()
        a, r, K = coefficient_curves(p, H)
        L_km = self.L / 1000.0
        return dict(H=H, a=a * L_km ** 2 / self.T, r=r / self.T, K=K * self.U)

    def H_support(self, years=None, stride=4):
        """Habitat values actually present over Omega (grid nodes, all years)."""
        years = range(self.cfg.year0, self.cfg.year1 + 1) if years is None else years
        vals = [self.hf.H_at_year_grid(y)[::stride, ::stride] for y in years]
        v = np.concatenate([a[np.isfinite(a)] for a in vals])
        return dict(values=v, q01=float(np.quantile(v, 0.01)), q99=float(np.quantile(v, 0.99)),
                    min=float(v.min()), max=float(v.max()))

    # ---- prediction on a grid
    @torch.no_grad()
    def predict_grid(self, year, nx=160, ny=170):
        xs = np.linspace(self.hf.x_min, self.hf.x_max, nx)
        ys = np.linspace(self.hf.y_min, self.hf.y_max, ny)
        X, Y = np.meshgrid(xs, ys)
        inside = self.hf.inside(X.ravel(), Y.ravel())
        n = self.model.net(self._txy(np.full(X.size, year), X.ravel(), Y.ravel())).cpu().numpy().ravel() * self.U
        n[~inside] = np.nan
        return X, Y, n.reshape(Y.shape)

    # ---- diagnostics
    def pde_residual(self, year, x, y, batch=20000):
        """PDE terms at physical points (normalised units: per unit of t~)."""
        year = np.broadcast_to(np.asarray(year, float), np.shape(x)).ravel()
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        keys = ("n", "n_t", "diffusion", "reaction", "advection", "residual")
        out = {k: [] for k in keys}
        for s in range(0, len(x), batch):
            sl = slice(s, s + batch)
            txy = self._txy(year[sl], x[sl], y[sl]).requires_grad_(True)
            terms = self.pde_terms(txy, self._habitat(year[sl], x[sl], y[sl]))
            for k in keys:
                out[k].append(terms[k].detach().cpu().numpy().ravel())
        return {k: np.concatenate(v) for k, v in out.items()}

    def dense_residual_stats(self, n=20000, seed=12345):
        """Residual on an INDEPENDENT dense LHS set, absolute and relative to |n_t|."""
        t, x, y = self._lhs_inside(n, self.cfg.year0, self.cfg.year1, seed=seed)
        d = self.pde_residual(t, x, y)
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
        scale = self.U / self.T
        col = self.pde_residual(*self._collocation_physical())
        return dict(n_points=int(n),
                    rms_residual=rms(d["residual"]), mean_residual=float(d["residual"].mean()),
                    max_abs_residual=float(np.abs(d["residual"]).max()),
                    rms_n_t=rms(d["n_t"]), rel_rms_residual_vs_n_t=rms(d["residual"]) / max(rms(d["n_t"]), 1e-12),
                    rms_residual_birds_per_yr=rms(d["residual"]) * scale,
                    collocation_rms_residual=rms(col["residual"]),
                    dense_over_collocation=rms(d["residual"]) / max(rms(col["residual"]), 1e-12))

    def _collocation_physical(self):
        txy = self.col["txy"].detach().cpu().numpy()
        return (self.cfg.year0 + txy[:, 0] * self.T, self.x0 + txy[:, 1] * self.L, self.y0 + txy[:, 2] * self.L)

    def mass_balance(self, years=None, nx=120, ny=130, n_boundary=4000):
        """Area-averaged decomposition of d<u>/dt over Omega for each year
        (birds/site/yr):  dU/dt = diffusion + reaction + advection + residual.
        boundary_flux is (1/|Omega|) * outward flux through dOmega, which should
        equal the diffusion term (divergence theorem) and be ~0 under no-flux."""
        years = list(range(self.cfg.year0, self.cfg.year1 + 1)) if years is None else list(years)
        xs = np.linspace(self.hf.x_min, self.hf.x_max, nx)
        ys = np.linspace(self.hf.y_min, self.hf.y_max, ny)
        X, Y = np.meshgrid(xs, ys)
        ins = self.hf.inside(X.ravel(), Y.ravel())
        gx, gy = X.ravel()[ins], Y.ravel()[ins]
        bp = self.hf.boundary_points()
        if len(bp["x"]) > n_boundary:
            take = np.linspace(0, len(bp["x"]) - 1, n_boundary).astype(int)
            w = len(bp["x"]) / n_boundary
            bp = {k: v[take] for k, v in bp.items()}
            bp["ds"] = bp["ds"] * w
        area_n = self.hf.domain_area_m2() / self.L ** 2
        scale = self.U / self.T
        rows = []
        for yr in years:
            d = self.pde_residual(np.full(len(gx), float(yr)), gx, gy)
            yb = np.full(len(bp["x"]), float(yr))
            txy = self._txy(yb, bp["x"], bp["y"]).requires_grad_(True)
            nrm_out = -self._tensor(np.column_stack([bp["nx"], bp["ny"]]))
            hab = self._habitat(yb, bp["x"], bp["y"])
            p = self.model.params()
            a = coefficient_curves(p, hab["H"].reshape(-1, 1))[0]
            flux = (a * self._bc_flux(txy, nrm_out, hab)).detach().cpu().numpy().ravel()
            bflux = float(np.sum(flux * bp["ds"] / self.L) / area_n)
            rows.append(dict(year=yr, mean_density=float(d["n"].mean() * self.U),
                             dU_dt=float(d["n_t"].mean() * scale),
                             diffusion=float(d["diffusion"].mean() * scale),
                             reaction=float(d["reaction"].mean() * scale),
                             advection=float(d["advection"].mean() * scale),
                             residual=float(d["residual"].mean() * scale),
                             boundary_flux=bflux * scale))
        return rows

    # ---- persistence
    def save(self, path):
        torch.save(dict(state_dict=self.model.state_dict(), config=self.cfg.to_dict()), path)

    @classmethod
    def from_checkpoint(cls, path, build_points=True, **overrides):
        """Load a checkpoint written by save(), or a bare state_dict (old pinn_weights.pt;
        pass the matching Config fields as overrides, e.g. v1_config fields)."""
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            cfgd = dict(obj["config"]); state = obj["state_dict"]
            cfgd["layers"] = tuple(cfgd["layers"]); cfgd["test_years"] = tuple(cfgd["test_years"])
        else:
            cfgd, state = {}, obj
        cfgd.update(overrides)
        cfgd.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        tr = cls(Config(**cfgd), build_points=build_points)
        tr.model.load_state_dict(state)
        tr.model.eval()
        return tr


def write_json(obj, path):
    def _py(o):
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    with open(path, "w") as f:
        json.dump(_py(obj), f, indent=2)
