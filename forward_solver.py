"""
forward_solver.py
=================
Finite-volume solver for the habitat-modulated reaction-diffusion model on the
France mask, used for three things the review asks for:

1. Forward-simulation (output-error) validation of the PINN estimates: solve the
   PDE with the identified coefficients from the 2002 field and compare every
   year with the reference (comment 6).
2. The principal eigenvalue lambda_1 of  -Diff[phi] - r(H) phi = lambda phi
   with zero-flux boundaries (Theorem 2): lambda_1 < 0 -> persistence,
   lambda_1 >= 0 -> extinction (comment 3f).
3. A classical identification baseline: output-error least squares with
   gradients obtained by automatic differentiation through the solver (the
   discrete adjoint), on the same reference field (comment 7).

Discretisation
--------------
Cell-centred finite volumes on a (coarsened) regular grid.  A face between two
domain cells carries the flux; faces leaving the domain carry none, which is the
homogeneous Neumann condition.  Mass is conserved to round-off when f = 0.

    fickian    : F_ij = a_f (u_j - u_i) / h^2,   a_f = (a_i + a_j) / 2
    ecological : F_ij = (a_j u_j - a_i u_i) / h^2           [lap(a u)]

Time stepping is explicit Euler with dt chosen from the diffusive and reactive
stability limits, which also preserves positivity.  Units: km, years, birds/site.
Physical parameters: D0 [km^2/yr], alpha, rmax [1/yr], beta, Kmax [birds/site], gamma.
"""

import math
import time

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

PHYS_KEYS = ("D0", "alpha", "rmax", "beta", "Kmax", "gamma")


def pinn_to_phys(physical_params):
    """Map Trainer.physical_params() keys to solver keys."""
    p = physical_params
    return dict(D0=p["D0_km2_per_yr"], alpha=p["alpha"], rmax=p["rmax_per_yr"],
                beta=p["beta"], Kmax=p["Kmax_birds"], gamma=p["gamma"])


class FVSolver:
    def __init__(self, hf, coarsen=5, diffusion="fickian", reaction="logistic",
                 dtype=torch.float64, device="cpu"):
        if diffusion not in ("fickian", "ecological") or reaction not in ("logistic", "source_sink"):
            raise ValueError("bad diffusion / reaction")
        g = hf.coarse_grid(coarsen)
        self.hf, self.grid = hf, g
        self.diffusion, self.reaction = diffusion, reaction
        self.dtype, self.device = dtype, torch.device(device)
        self.mask = g["mask"]
        self.N = int(self.mask.sum())
        self.x, self.y = g["x"], g["y"]
        self.dx_km, self.dy_km = g["dx_m"] / 1000.0, g["dy_m"] / 1000.0

        idx = -np.ones(self.mask.shape, dtype=np.int64)
        idx[self.mask] = np.arange(self.N)
        right = self.mask[:, :-1] & self.mask[:, 1:]
        up = self.mask[:-1, :] & self.mask[1:, :]
        fi = np.concatenate([idx[:, :-1][right], idx[:-1, :][up]])
        fj = np.concatenate([idx[:, 1:][right], idx[1:, :][up]])
        fw = np.concatenate([np.full(right.sum(), 1.0 / self.dx_km ** 2),
                             np.full(up.sum(), 1.0 / self.dy_km ** 2)])
        self._fi_np, self._fj_np, self._fw_np = fi, fj, fw
        t = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=self.device)
        self.fi, self.fj = t(fi, torch.long), t(fj, torch.long)
        self.fw = t(fw)
        self.Zk = t(hf.Z_slices_at(self.x, self.y))          # (N, nk)
        self.u_ref = g["u_ref"]                               # (nk, N) numpy
        self.year0 = hf.year0
        self.years_ref = hf.year0 + np.arange(hf.nk)

    # ------------------------------------------------------------ helpers
    def _t(self, a):
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)

    def H(self, years):
        """Habitat index at cell centres, shape (N, len(years))."""
        w = self._t(self.hf.temporal_weights(np.atleast_1d(years)))   # (m, nk)
        return torch.sigmoid(self.Zk @ w.T)

    def image(self, u):
        """Cell vector -> 2-D array (NaN outside) for plotting."""
        A = np.full(self.mask.shape, np.nan)
        A[self.mask] = np.asarray(u.detach().cpu() if torch.is_tensor(u) else u)
        return A

    def cells_in_blocks(self, block_ids_fn, blocks):
        """Boolean (N,) - cells whose centre lies in the given spatial blocks."""
        return np.isin(block_ids_fn(self.x, self.y), blocks)

    def fill_from(self, values, known):
        """Nearest-neighbour fill of cells where known is False (e.g. test blocks in the IC)."""
        values = np.asarray(values, float).copy()
        if known.all():
            return values
        tree = cKDTree(np.column_stack([self.x[known], self.y[known]]))
        _, j = tree.query(np.column_stack([self.x[~known], self.y[~known]]))
        values[~known] = values[known][j]
        return values

    # ------------------------------------------------------------ dynamics
    def _coeffs(self, H, p):
        a = p["D0"] * (1.0 - p["alpha"] * H)
        r = p["rmax"] * (p["beta"] + (1.0 - p["beta"]) * H)
        K = p["Kmax"] * (p["gamma"] + (1.0 - p["gamma"]) * H)
        return a, r, K

    def terms(self, u, H, p):
        """(diffusion, reaction) at cell centres for state u and habitat H."""
        a, r, K = self._coeffs(H, p)
        fi, fj = self.fi, self.fj
        if self.diffusion == "fickian":
            flux = self.fw * 0.5 * (a[fi] + a[fj]) * (u[fj] - u[fi])
        else:
            flux = self.fw * (a[fj] * u[fj] - a[fi] * u[fi])
        diff = torch.zeros_like(u).index_add(0, fi, flux).index_add(0, fj, -flux)
        if self.reaction == "logistic":
            react = r * u * (1.0 - u / K)
        else:
            react = u * (r - p["rmax"] * u / K)
        return diff, react

    def stable_dt(self, p, dt_max=0.05, cfl=0.8, u_max=None):
        p = {k: _f(v) for k, v in p.items()}
        D0 = abs(float(p["D0"]))
        lim_d = cfl / (2.0 * D0 * (1.0 / self.dx_km ** 2 + 1.0 / self.dy_km ** 2)) if D0 > 0 else math.inf
        rmax = abs(float(p["rmax"]))
        u_max = float(np.nanmax(self.u_ref)) if u_max is None else u_max
        kmin = max(float(p["Kmax"]) * min(float(p["gamma"]), 1.0), 1e-6)
        lim_r = 0.2 / max(rmax * (1.0 + 2.0 * u_max / kmin), 1e-12)
        return min(dt_max, lim_d, lim_r)

    def _year_step(self, u, Hs, dt, *pv):
        p = dict(zip(PHYS_KEYS, pv))
        for s in range(Hs.shape[1]):
            diff, react = self.terms(u, Hs[:, s], p)
            u = u + dt * (diff + react)
        return u

    def simulate(self, p, year_start=None, year_end=None, u0=None, dt_max=0.05, cfl=0.8,
                 checkpoint=None):
        """Integrate from year_start (state u0, default the reference slice) to
        year_end.  Returns a tensor (n_years + 1, N) at integer years."""
        year_start = self.year0 if year_start is None else int(year_start)
        year_end = int(self.years_ref[-1]) if year_end is None else int(year_end)
        if u0 is None:
            u0 = self.u_ref[year_start - self.year0]
        u = self._t(u0)
        pv = [p[k] if torch.is_tensor(p[k]) else self._t(float(p[k])) for k in PHYS_KEYS]
        n_sub = int(math.ceil(1.0 / self.stable_dt(p, dt_max, cfl)))
        dt = 1.0 / n_sub
        grad_needed = any(v.requires_grad for v in pv)
        checkpoint = grad_needed if checkpoint is None else checkpoint
        out = [u]
        for year in range(year_start, year_end):
            Hs = self.H(year + np.arange(n_sub) / n_sub)
            if checkpoint:
                u = torch.utils.checkpoint.checkpoint(self._year_step, u, Hs, dt, *pv, use_reentrant=False)
            else:
                u = self._year_step(u, Hs, dt, *pv)
            out.append(u)
        self.last_n_sub = n_sub
        return torch.stack(out)

    # ------------------------------------------------------------ evaluation
    def metrics(self, sim, years=None, cells=None, year_start=None):
        """Per-year and pooled agreement with the reference (excluding the IC year)."""
        sim = sim.detach().cpu().numpy() if torch.is_tensor(sim) else np.asarray(sim)
        year_start = self.year0 if year_start is None else year_start
        sim_years = year_start + np.arange(sim.shape[0])
        years = [y for y in sim_years[1:] if y in self.years_ref] if years is None else list(years)
        cells = np.ones(self.N, bool) if cells is None else np.asarray(cells, bool)
        rows, P, O = [], [], []
        for y in years:
            s = sim[y - year_start][cells]
            o = self.u_ref[y - self.year0][cells]
            rows.append(dict(year=int(y), r2=_r2(o, s), rmse=float(np.sqrt(np.mean((s - o) ** 2))),
                             corr=float(np.corrcoef(s, o)[0, 1]), mean_sim=float(s.mean()),
                             mean_ref=float(o.mean())))
            P.append(s); O.append(o)
        P, O = np.concatenate(P), np.concatenate(O)
        pooled = dict(r2=_r2(O, P), rmse=float(np.sqrt(np.mean((P - O) ** 2))),
                      corr=float(np.corrcoef(P, O)[0, 1]))
        return dict(per_year=rows, pooled=pooled)

    def mass_budget(self, sim, p, year_start=None):
        """Domain-mean d/dt split into diffusion (should be ~0: no-flux) and reaction."""
        year_start = self.year0 if year_start is None else year_start
        rows = []
        with torch.no_grad():
            for k in range(sim.shape[0]):
                y = year_start + k
                u = sim[k] if torch.is_tensor(sim) else self._t(sim[k])
                diff, react = self.terms(u, self.H([y])[:, 0], {q: float(p[q]) for q in PHYS_KEYS})
                rows.append(dict(year=y, mean_density=float(u.mean()), diffusion=float(diff.mean()),
                                 reaction=float(react.mean())))
        return rows

    # ------------------------------------------------------------ eigenvalue
    def principal_eigenvalue(self, p, year):
        """lambda_1 of  -Diff[phi] - r(H(year)) phi = lambda phi  with zero flux.
        lambda_1 < 0: the zero state is unstable (persistence);
        lambda_1 >= 0: extinction (Cantrell & Cosner 2003, logistic case)."""
        p = {k: float(p[k]) for k in PHYS_KEYS}
        H = self.H([year])[:, 0].cpu().numpy()
        a = p["D0"] * (1.0 - p["alpha"] * H)
        r = p["rmax"] * (p["beta"] + (1.0 - p["beta"]) * H)
        fi, fj, fw = self._fi_np, self._fj_np, self._fw_np
        N = self.N
        if p["D0"] <= 0:
            return float(-r.max())
        if self.diffusion == "fickian":
            off = fw * 0.5 * (a[fi] + a[fj])
            S = sparse.coo_matrix((np.r_[off, off], (np.r_[fi, fj], np.r_[fj, fi])), shape=(N, N)).tocsr()
        else:
            # A = Lap D_a + D_r is similar to D_a^1/2 Lap D_a^1/2 + D_r (symmetric, a > 0)
            sa = np.sqrt(np.maximum(a, 1e-300))
            off = fw * sa[fi] * sa[fj]
            S = sparse.coo_matrix((np.r_[off, off], (np.r_[fi, fj], np.r_[fj, fi])), shape=(N, N)).tocsr()
        deg = np.zeros(N)
        if self.diffusion == "fickian":
            np.add.at(deg, fi, fw * 0.5 * (a[fi] + a[fj])); np.add.at(deg, fj, fw * 0.5 * (a[fi] + a[fj]))
        else:
            np.add.at(deg, fi, fw * a[fi]); np.add.at(deg, fj, fw * a[fj])
        A = S + sparse.diags(r - deg)
        shift = r.max() + 1e-6 * (abs(r.max()) + 1.0)          # every eigenvalue is <= max r
        val = eigsh(A.tocsc(), k=1, sigma=shift, which="LM", return_eigenvectors=False)[0]
        return float(-val)


def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def _r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


# ---------------------------------------------------------------- output-error identification
def fit_output_error(solver, U=15.0, train_years=None, train_cells=None, ic_known=None,
                     fixed=None, init=None, beta_min=-1.0, D_max=2.0e4,
                     n_adam=300, lr=0.05, n_lbfgs=30, verbose=True, log_every=25):
    """Classical output-error identification on the same reference field.

    Minimises J = mean over (train_years, train_cells) of ((u_sim - u_ref)/U)^2
    over the free physical parameters, with gradients from automatic
    differentiation through the explicit finite-volume solver (discrete adjoint).
    Parameterisation mirrors the PINN constraints: D0 = D_max * sigmoid,
    alpha, gamma = sigmoid, rmax = softplus, beta = sigmoid (logistic) or
    beta_min + (1 - beta_min) sigmoid (source_sink); Kmax fixed (default 18)."""
    fixed = dict(Kmax=18.0) if fixed is None else dict(fixed)
    init = {} if init is None else dict(init)
    dev, dt = solver.device, solver.dtype
    t0 = time.time()
    source_sink = solver.reaction == "source_sink"

    def inv_sig(v):
        v = min(max(v, 1e-6), 1 - 1e-6)
        return math.log(v / (1 - v))

    def inv_sp(v):
        return math.log(math.expm1(max(v, 1e-8)))

    start = dict(D0=init.get("D0", 500.0), alpha=init.get("alpha", 0.5), rmax=init.get("rmax", 0.1),
                 beta=init.get("beta", 0.0 if source_sink else 0.5), Kmax=init.get("Kmax", 18.0),
                 gamma=init.get("gamma", 0.5))
    raw0 = dict(D0=inv_sig(start["D0"] / D_max), alpha=inv_sig(start["alpha"]), rmax=inv_sp(start["rmax"]),
                beta=inv_sig((start["beta"] - beta_min) / (1 - beta_min)) if source_sink else inv_sig(start["beta"]),
                Kmax=inv_sp(start["Kmax"]), gamma=inv_sig(start["gamma"]))
    raw = {k: torch.tensor(raw0[k], dtype=dt, device=dev, requires_grad=True)
           for k in PHYS_KEYS if k not in fixed}

    def params():
        sp, sg = torch.nn.functional.softplus, torch.sigmoid
        maps = dict(D0=lambda z: D_max * sg(z), alpha=sg, rmax=sp,
                    beta=(lambda z: beta_min + (1 - beta_min) * sg(z)) if source_sink else sg,
                    Kmax=sp, gamma=sg)
        return {k: (torch.tensor(float(fixed[k]), dtype=dt, device=dev) if k in fixed else maps[k](raw[k]))
                for k in PHYS_KEYS}

    nk = len(solver.years_ref)
    train_years = list(solver.years_ref[1:]) if train_years is None else [y for y in train_years if y != solver.year0]
    k_idx = torch.as_tensor([y - solver.year0 for y in train_years], device=dev)
    cells = np.ones(solver.N, bool) if train_cells is None else np.asarray(train_cells, bool)
    cell_idx = torch.as_tensor(np.where(cells)[0], device=dev)
    ref = torch.as_tensor(solver.u_ref, dtype=dt, device=dev)
    u0 = solver.u_ref[0] if ic_known is None else solver.fill_from(solver.u_ref[0], np.asarray(ic_known, bool))
    year_end = int(max(train_years))

    def objective():
        sim = solver.simulate(params(), year_end=year_end, u0=u0)
        err = (sim[k_idx][:, cell_idx] - ref[k_idx][:, cell_idx]) / U
        return (err ** 2).mean()

    history = []
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    for it in range(n_adam):
        opt.zero_grad()
        J = objective()
        J.backward()
        opt.step()
        history.append(float(J.detach()))
        if verbose and (it % log_every == 0 or it == n_adam - 1):
            pv = {k: float(v) for k, v in params().items()}
            print(f"[OE adam {it:4d}] J={history[-1]:.4e}  D0={pv['D0']:.1f}  alpha={pv['alpha']:.3f}  "
                  f"rmax={pv['rmax']:.4f}  beta={pv['beta']:.3f}  gamma={pv['gamma']:.3f}")
    if n_lbfgs > 0:
        opt2 = torch.optim.LBFGS(list(raw.values()), max_iter=n_lbfgs, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            J = objective()
            J.backward()
            return J
        opt2.step(closure)
        with torch.no_grad():
            history.append(float(objective()))
    est = {k: float(v) for k, v in params().items()}
    return dict(params=est, objective=history[-1], history=history, runtime_s=time.time() - t0,
                n_sub_per_year=getattr(solver, "last_n_sub", None), u0=u0)
