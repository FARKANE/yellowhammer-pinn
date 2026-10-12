"""
diagnostics.py
==============
Post-training checks shared by the scripts and notebooks.

  mass_balance_report   per-year split of d<u>/dt into diffusion / reaction /
                        residual / boundary flux
  forward_validation    solve the PDE with the PINN coefficients and compare
                        with the reference, by year and on held-out cells
                        (output-error validation)
  eigenvalue_series     lambda_1 per year for the identified model
  plot_*                matching figures
"""

import numpy as np
import torch

from forward_solver import FVSolver, pinn_to_phys


# ---------------------------------------------------------------------------- tables
def mass_balance_report(tr, years=None, **kw):
    rows = tr.mass_balance(years=years, **kw)
    yrs = [r["year"] for r in rows]
    # cumulative contributions over the window (trapezoid in time), birds/site
    trapz = getattr(np, "trapezoid", None) or np.trapz          # numpy 1.x / 2.x
    cum = {}
    for key in ("dU_dt", "diffusion", "reaction", "advection", "residual", "boundary_flux"):
        v = np.array([r[key] for r in rows])
        cum[key] = float(trapz(v, yrs)) if len(v) > 1 else 0.0
    return dict(rows=rows, cumulative=cum)


def forward_validation(tr, coarsen=5, dt_max=0.05):
    """Forward simulation with the identified coefficients from the 2002 field."""
    c = tr.cfg
    solver = FVSolver(tr.hf, coarsen=coarsen, diffusion=c.diffusion, reaction=c.reaction)
    p = pinn_to_phys(tr.physical_params())
    test_cells = None
    if c.split == "block" and getattr(tr, "test_blocks", None) is not None:
        test_cells = solver.cells_in_blocks(tr._block_ids, tr.test_blocks)
    u0 = solver.fill_from(solver.u_ref[0], ~test_cells) if test_cells is not None else None
    with torch.no_grad():
        sim = solver.simulate(p, u0=u0, dt_max=dt_max).cpu().numpy()
    pinn = np.stack([tr.predict(float(y), solver.x, solver.y) for y in solver.years_ref])
    out = dict(params=p, n_cells=solver.N, cell_km=solver.dx_km, n_sub_per_year=solver.last_n_sub,
               simulation_all=solver.metrics(sim), pinn_all=solver.metrics(pinn))
    if test_cells is not None:
        out["simulation_test_cells"] = solver.metrics(sim, cells=test_cells)
        out["pinn_test_cells"] = solver.metrics(pinn, cells=test_cells)
    if c.split == "temporal" and c.test_years:
        ty = [y for y in c.test_years if y in solver.years_ref]
        out["simulation_test_years"] = solver.metrics(sim, years=ty)
        out["pinn_test_years"] = solver.metrics(pinn, years=ty)
    out["national_mean"] = dict(year=solver.years_ref.tolist(),
                                reference=solver.u_ref.mean(axis=1).tolist(),
                                simulation=sim.mean(axis=1).tolist(),
                                pinn=pinn.mean(axis=1).tolist())
    out["mass_budget_simulation"] = solver.mass_budget(sim, p)
    return out, solver, sim, pinn


def eigenvalue_series(solver, p, years=None):
    years = solver.years_ref if years is None else years
    return [dict(year=int(y), lambda1=solver.principal_eigenvalue(p, y)) for y in years]


def summarise(values):
    v = np.asarray(values, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


# ---------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mass_balance(report, path):
    plt = _plt()
    rows = report["rows"]
    yrs = [r["year"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, lab, st in [("dU_dt", r"$d\langle u\rangle/dt$", "k-o"), ("reaction", "reaction", "-s"),
                         ("diffusion", "diffusion (= net boundary flux)", "-^"),
                         ("residual", "PDE residual", "-d"), ("boundary_flux", "boundary flux (direct)", ":")]:
        ax[0].plot(yrs, [r[key] for r in rows], st, ms=3, lw=1.3, label=lab)
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("birds / site / yr (domain mean)")
    ax[0].set_title("(a) Mass balance of the identified model"); ax[0].legend(fontsize=7)
    cum = report["cumulative"]
    keys = ["dU_dt", "reaction", "diffusion", "residual"]
    ax[1].bar(range(len(keys)), [cum[k] for k in keys], color=["k", "C1", "C2", "C3"])
    ax[1].set_xticks(range(len(keys)), ["total change", "reaction", "diffusion", "residual"])
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_ylabel("cumulative, birds / site"); ax[1].set_title("(b) What carries the change")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_trajectories(fv, path, highlight_years=None):
    plt = _plt()
    nm = fv["national_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(nm["year"], nm["reference"], "-s", ms=3, color="#1b9e77", label="reference field")
    ax.plot(nm["year"], nm["pinn"], "-o", ms=3, color="#d95f02", label="PINN field")
    ax.plot(nm["year"], nm["simulation"], "--", lw=2, color="#7570b3", label="PDE simulation (identified coefficients)")
    if highlight_years:
        ax.axvspan(min(highlight_years) - 0.5, max(highlight_years) + 0.5, color="grey", alpha=0.12, label="held-out years")
    ax.set_xlabel("year"); ax.set_ylabel("mean density (birds / site)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_response_curves(tr, path):
    """Identified a(H), r(H), K(H) with the unsupported H range shaded and the
    distribution of H over the domain."""
    plt = _plt()
    rc = tr.response_curves()
    sup = tr.H_support()
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    ax[0].hist(sup["values"], bins=50, color="#666", density=True)
    ax[0].set_title("(a) Distribution of H over Omega"); ax[0].set_xlabel("H")
    for a_, key, lab, col in [(ax[1], "r", "r [1/yr]", "#1b9e77"), (ax[2], "K", "K [birds/site]", "#d95f02"),
                              (ax[3], "a", "a [km$^2$/yr]", "#7570b3")]:
        a_.plot(rc["H"], rc[key], lw=2, color=col)
        a_.axvspan(0, sup["q01"], color="grey", alpha=0.2)
        a_.axvspan(sup["q99"], 1, color="grey", alpha=0.2)
        a_.set_xlabel("habitat index H"); a_.set_ylabel(lab); a_.grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("(b) growth rate r(H)"); ax[2].set_title("(c) capacity K(H)"); ax[3].set_title("(d) diffusivity a(H)")
    fig.text(0.5, -0.02, "shaded: H outside the 1-99% range observed over the domain (extrapolation)",
             ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_eigenvalues(series, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot([s["year"] for s in series], [s["lambda1"] for s in series], "-o", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("year (habitat H(t) frozen at that year)"); ax.set_ylabel(r"$\lambda_1$ [1/yr]")
    ax.set_title(r"Principal eigenvalue: $\lambda_1<0$ persistence, $\lambda_1\geq 0$ extinction", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- paper figures (from the notebook)
def per_year_fidelity(tr, exclude_training=False):
    """Full-grid agreement with the reference for each slice.  With
    exclude_training=True, nodes in the training/selection records and, for a
    block split, every node outside the test blocks are removed."""
    hf = tr.hf
    X, Y = np.meshgrid(hf.lon_ax, hf.lat_ax)
    keep_nodes = hf.M.copy()
    if exclude_training and tr.cfg.split == "block" and getattr(tr, "test_blocks", None) is not None:
        keep_nodes &= np.isin(tr._block_ids(X, Y), tr.test_blocks)
    rows = []
    for k in range(hf.nk):
        year = hf.year0 + k
        sel = keep_nodes.copy()
        if exclude_training and tr.cfg.split == "random" and hasattr(tr, "rec"):
            used = np.concatenate([tr.idx_train, tr.idx_select])
            used = used[tr.rec["k"][used] == k]
            sel[tr.rec["I"][used], tr.rec["J"][used]] = False
        if exclude_training and tr.cfg.split == "temporal" and year not in tr.cfg.test_years:
            continue
        u = hf.u_GLM[:, :, k][sel]
        pred = tr.predict(float(year), X[sel], Y[sel])
        rows.append(dict(year=int(year), r2=float(1 - np.sum((u - pred) ** 2) / np.sum((u - u.mean()) ** 2)),
                         rmse=float(np.sqrt(np.mean((u - pred) ** 2))), corr=float(np.corrcoef(u, pred)[0, 1]),
                         n=int(sel.sum())))
    return rows


def plot_density_maps(tr, path, years=(2005, 2014, 2024)):
    import pandas as pd
    from matplotlib.colors import Normalize
    plt = _plt()
    df = pd.read_csv(tr.cfg.csv)
    grids = [tr.predict_grid(y, nx=150, ny=160) for y in years]
    vmax = max(np.nanpercentile(g[2], 99) for g in grids)
    fig, axes = plt.subplots(1, len(years), figsize=(3.9 * len(years), 4.2))
    for axk, yr, (X, Y, Z) in zip(np.atleast_1d(axes), years, grids):
        pc = axk.pcolormesh(X / 1e3, Y / 1e3, Z, cmap="viridis", norm=Normalize(0, vmax), shading="auto", rasterized=True)
        axk.plot(tr.hf.Xfr / 1e3, tr.hf.Yfr / 1e3, "k", lw=0.6)
        d = df[df.Year == yr]
        axk.scatter(d.Longitude / 1e3, d.Latitude / 1e3, c=d.EMBCIT, cmap="viridis", norm=Normalize(0, vmax),
                    s=9, edgecolor="w", linewidth=0.2, rasterized=True)
        axk.set_title(f"PINN density $u(t,x,y)$, {yr}"); axk.set_aspect("equal")
        axk.set_xlabel("x [km]"); axk.set_ylabel("y [km]")
        plt.colorbar(pc, ax=axk, fraction=0.046, pad=0.03, label="birds / site")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_calibration(tr, path):
    plt = _plt()
    d = tr.val_glm
    pred = tr.predict_from_txy(d["txy"]) if hasattr(tr, "predict_from_txy") else \
        tr.model.net(d["txy"]).detach().cpu().numpy().ravel() * tr.U
    u = d["u"]
    r2 = 1 - np.sum((u - pred) ** 2) / np.sum((u - u.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(u, pred, s=5, alpha=0.2, color="#2c7fb8", rasterized=True)
    m = max(u.max(), pred.max()); ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set_xlabel("reference density (held-out test records)"); ax.set_ylabel("PINN")
    ax.set_title(f"Test calibration, $R^2$={r2:.2f} ({tr.cfg.split} split)"); ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_all_years(tr, path):
    from matplotlib.colors import Normalize
    plt = _plt()
    years = list(range(tr.cfg.year0, tr.cfg.year1 + 1))
    grids = [tr.predict_grid(y, nx=110, ny=120) for y in years]
    vmax = np.nanpercentile(np.concatenate([g[2][~np.isnan(g[2])] for g in grids]), 99)
    ncol = 6; nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 1.95 * nrow))
    for k, (yr, (X, Y, Z)) in enumerate(zip(years, grids)):
        a = axes.ravel()[k]
        pc = a.pcolormesh(X / 1e3, Y / 1e3, Z, cmap="viridis", norm=Normalize(0, vmax), shading="auto", rasterized=True)
        a.plot(tr.hf.Xfr / 1e3, tr.hf.Yfr / 1e3, "k", lw=0.4)
        a.set_title(str(yr), fontsize=9); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for k in range(len(years), nrow * ncol):
        axes.ravel()[k].axis("off")
    fig.subplots_adjust(right=0.92, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.94, 0.18, 0.015, 0.64]); fig.colorbar(pc, cax=cax, label="birds / site")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
