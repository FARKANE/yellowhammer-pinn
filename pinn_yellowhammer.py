"""
pinn_yellowhammer.py
====================
Physics-Informed Neural Network for the Yellowhammer (Bruant jaune) population
density field u(t,x,y) over mainland France (2002-2024), with simultaneous
inverse identification of the ecological parameters theta = (D0, alpha, rmax,
beta, Kmax, gamma).

Governing PDE (reaction-diffusion with habitat-dependent coefficients):

    du/dt = div( a(H) grad u ) + r(H) u (1 - u/K(H))            (+ optional advection)
      a(H) = D0 (1 - alpha H)
      r(H) = rmax (beta + (1-beta) H)
      K(H) = Kmax (gamma + (1-gamma) H)

KEY ENGINEERING CHOICES (these are what make it converge to good results):

 1. Full non-dimensionalization. Coordinates are in Lambert metres (~1e6) and
    years (~1e1); raw, the PDE is catastrophically ill-conditioned for a NN.
    We map x,y -> [0,1] with a single length scale L (isotropy preserved) and
    t -> [0,1], and density u -> n = u/U. The network learns DIMENSIONLESS
    parameters; physical values are recovered analytically afterwards.

 2. Positivity & sign-constrained parameters. The output is softplus(.) so
    n >= 0. Parameters are reparameterized so that a(H)>0, r(H)>0, K(H)>0 hold
    for ALL H in [0,1] by construction (alpha,beta,gamma in (0,1); D0,rmax,Kmax>0).
    Guaranteeing a(H)>0 removes the backward-diffusion ill-posedness that is the
    usual cause of advection/diffusion instability in this class of problem.

 3. Habitat enters only through precomputed coefficients (H, gradH at the fixed
    collocation points), NOT as a network input, so the autodiff residual is exact.

 4. Adam (warm, with residual-weight ramp) followed by L-BFGS polishing.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from scipy.stats import qmc

from habitat_field import HabitatField


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    u_mat: str = "u_GLM_Bruant_jaune.mat"
    map_mat: str = "Map_France.mat"
    csv: str = "Yellowhammer_2002_2024.csv"

    year0: int = 2002
    year1: int = 2024
    U: float = 15.0                     # density scale (u_sat)

    # network
    layers: tuple = (3, 64, 64, 64, 64, 1)
    fourier_m: int = 0                  # >0 enables Fourier features of size m on (x,y)
    fourier_sigma: float = 3.0

    # sampling
    n_collocation: int = 20000
    n_ic: int = 4000
    max_data: int = 0                   # 0 = use all observations
    val_frac: float = 0.15              # held-out fraction of observations for validation
    use_glm_data: bool = True           # fit dense denoised GLM field (identifiable inverse)
    n_glm: int = 15000                  # total GLM field samples; split train/test by train_frac
    train_frac: float = 0.70            # disjoint 70/30 split of the GLM field (train/test)
    fix_Kmax_tilde: float = 1.2         # if >0, Kmax is fixed (dimensionless); else learned
    fix_D0_tilde: float = -1.0          # if >=0, D0_tilde is fixed (for profile likelihood)
    n_boundary: int = 2000
    use_bc: bool = True
    use_advection: bool = False         # taxis term -div(chi*gradH * u); off => matches PDF

    # loss weights
    w_res: float = 1.0
    w_data: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 0.1
    res_ramp_steps: int = 1500          # ramp residual weight 0->w_res over these steps
    # weak ecological prior on dimensionless diffusion D0_tilde (Tikhonov anchor).
    # disabled when w_prior<=0 or D0_tilde_prior<0. Calibration:
    # D0_phys[km2/yr] = D0_tilde * (L/1000)^2 / T.
    w_prior: float = 0.0
    D0_tilde_prior: float = -1.0

    # optimization
    adam_steps: int = 8000
    adam_lr: float = 2e-3
    lbfgs_steps: int = 1500
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, layers, fourier_m=0, fourier_sigma=3.0):
        super().__init__()
        self.fourier_m = fourier_m
        in_dim = layers[0]
        if fourier_m > 0:
            # random Fourier features on the 2 spatial coords (cols 1,2)
            B = torch.randn(2, fourier_m) * fourier_sigma
            self.register_buffer("Bff", B)
            in_dim = layers[0] + 2 * fourier_m
        seq, dims = [], [in_dim] + list(layers[1:])
        for i in range(len(dims) - 2):
            seq += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        seq += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*seq)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, txy):
        if self.fourier_m > 0:
            xy = txy[:, 1:3]
            proj = 2 * np.pi * xy @ self.Bff
            feats = torch.cat([txy, torch.sin(proj), torch.cos(proj)], dim=1)
        else:
            feats = txy
        return torch.nn.functional.softplus(self.net(feats))   # density >= 0


class PINN(nn.Module):
    """Holds the field network and the trainable dimensionless parameters."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(cfg.layers, cfg.fourier_m, cfg.fourier_sigma)
        # unconstrained raw parameters -> constrained via softplus / sigmoid
        self.pD = nn.Parameter(torch.tensor(-2.0))   # D0_tilde  = softplus(pD)   ~0.13
        self._fixD = cfg.fix_D0_tilde
        if self._fixD >= 0:
            self.register_buffer("Dfix", torch.tensor(float(self._fixD)))
        self.pa = nn.Parameter(torch.tensor(0.0))     # alpha     = sigmoid(pa)    =0.5
        self.pr = nn.Parameter(torch.tensor(0.0))     # rmax_tilde= softplus(pr)   ~0.69
        self.pb = nn.Parameter(torch.tensor(0.0))     # beta      = sigmoid(pb)
        self._fixK = cfg.fix_Kmax_tilde
        if self._fixK > 0:
            self.register_buffer("Kfix", torch.tensor(float(self._fixK)))
        else:
            self.pK = nn.Parameter(torch.tensor(0.0))  # Kmax_tilde= softplus(pK)
        self.pg = nn.Parameter(torch.tensor(0.0))     # gamma     = sigmoid(pg)
        if cfg.use_advection:
            self.pchi = nn.Parameter(torch.tensor(-2.0))  # chi_tilde = softplus

    # constrained, dimensionless parameters
    def params(self):
        sp = torch.nn.functional.softplus
        sg = torch.sigmoid
        Kmax = self.Kfix if self._fixK > 0 else (sp(self.pK) + 0.05)
        D0 = self.Dfix if self._fixD >= 0 else sp(self.pD)
        p = dict(D0=D0, alpha=sg(self.pa), rmax=sp(self.pr),
                 beta=sg(self.pb), Kmax=Kmax, gamma=sg(self.pg))
        if self.cfg.use_advection:
            p["chi"] = sp(self.pchi)
        return p


# --------------------------------------------------------------------------- driver
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.dev = torch.device(cfg.device)

        self.hf = HabitatField(cfg.u_mat, cfg.map_mat, year0=cfg.year0)
        # normalization constants
        self.x0, self.y0 = self.hf.x_min, self.hf.y_min
        self.L = max(self.hf.x_max - self.hf.x_min, self.hf.y_max - self.hf.y_min)
        self.T = float(cfg.year1 - cfg.year0)
        self.U = cfg.U

        self.model = PINN(cfg).to(self.dev)
        self._build_points()

    # ---- normalization helpers
    def to_norm(self, year, x, y):
        return ((year - self.cfg.year0) / self.T,
                (x - self.x0) / self.L,
                (y - self.y0) / self.L)

    def _tensor(self, a):
        return torch.tensor(np.asarray(a, np.float64), dtype=torch.float32, device=self.dev)

    # ---- point sets
    def _lhs_inside(self, n, t_lo, t_hi):
        """Latin-Hypercube sample (year,x,y) rejected to inside France."""
        out_y, out_x, out_t = [], [], []
        eng = qmc.LatinHypercube(d=3, seed=self.cfg.seed)
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
        t = np.concatenate(out_t)[:n]; x = np.concatenate(out_x)[:n]; y = np.concatenate(out_y)[:n]
        return t, x, y

    def _build_points(self):
        c = self.cfg
        # ----- collocation
        t, x, y = self._lhs_inside(c.n_collocation, c.year0, c.year1)
        H, Hx, Hy = self.hf.query(t, x, y)
        tn, xn, yn = self.to_norm(t, x, y)
        self.col = dict(
            txy=self._tensor(np.column_stack([tn, xn, yn])),
            H=self._tensor(H), Hx=self._tensor(Hx * self.L), Hy=self._tensor(Hy * self.L))
        self.col["txy"].requires_grad_(True)

        # ----- data: dense denoised GLM field (identifiable inverse), or sparse counts
        import pandas as pd
        df = pd.read_csv(c.csv)
        if c.use_glm_data:
            # Draw the GLM field points ONCE, then split disjointly into
            # train_frac (train) and 1-train_frac (held-out test). This
            # guarantees the test points never appear in training.
            yr, xx, yy, uu = self.hf.sample_glm(c.n_glm, seed=c.seed)
            rng = np.random.default_rng(c.seed)
            perm = rng.permutation(len(uu))
            n_tr = int(c.train_frac * len(uu))
            tr, te = perm[:n_tr], perm[n_tr:]

            # --- training set (train_frac of the field) ---
            tn, xn, yn = self.to_norm(yr[tr], xx[tr], yy[tr])
            self.data = dict(
                txy=self._tensor(np.column_stack([tn, xn, yn])),
                n=self._tensor(uu[tr] / self.U).reshape(-1, 1))

            # --- disjoint held-out test set (1-train_frac of the field) ---
            tn, xn, yn = self.to_norm(yr[te], xx[te], yy[te])
            self.val_glm = dict(
                txy=self._tensor(np.column_stack([tn, xn, yn])), u=uu[te])

            # ALL counts held out for independent validation (never seen in training)
            dv = df
            tn, xn, yn = self.to_norm(dv.Year.values, dv.Longitude.values, dv.Latitude.values)
            self.val = dict(
                txy=self._tensor(np.column_stack([tn, xn, yn])),
                n=self._tensor(dv.EMBCIT.values / self.U).reshape(-1, 1),
                counts=dv.EMBCIT.values.astype(float))
        else:
            if c.max_data and len(df) > c.max_data:
                df = df.sample(c.max_data, random_state=c.seed)
            df = df.sample(frac=1.0, random_state=c.seed).reset_index(drop=True)
            n_val = int(c.val_frac * len(df))
            df_val, df_tr = df.iloc[:n_val], df.iloc[n_val:]
            tn, xn, yn = self.to_norm(df_tr.Year.values, df_tr.Longitude.values, df_tr.Latitude.values)
            self.data = dict(
                txy=self._tensor(np.column_stack([tn, xn, yn])),
                n=self._tensor(df_tr.EMBCIT.values / self.U).reshape(-1, 1))
            tn, xn, yn = self.to_norm(df_val.Year.values, df_val.Longitude.values, df_val.Latitude.values)
            self.val = dict(
                txy=self._tensor(np.column_stack([tn, xn, yn])),
                n=self._tensor(df_val.EMBCIT.values / self.U).reshape(-1, 1),
                counts=df_val.EMBCIT.values.astype(float))

        # ----- initial condition (t = year0)
        t, x, y = self._lhs_inside(c.n_ic, c.year0, c.year0)
        g = self.hf.g_initial(x, y)
        tn, xn, yn = self.to_norm(t, x, y)
        self.ic = dict(
            txy=self._tensor(np.column_stack([tn, xn, yn])),
            n=self._tensor(g / self.U).reshape(-1, 1))

        # ----- boundary (Neumann) points + inward normals
        if c.use_bc:
            self._build_boundary()

    def _build_boundary(self):
        Xfr, Yfr = self.hf.Xfr, self.hf.Yfr
        finite = np.isfinite(Xfr) & np.isfinite(Yfr)
        xs, ys = Xfr[finite], Yfr[finite]
        # tangents -> normals (rotate 90 deg), point a touch inside
        tx = np.gradient(xs); ty = np.gradient(ys)
        nrm = np.hypot(tx, ty) + 1e-9
        nx, ny = ty / nrm, -tx / nrm                # one normal direction
        # ensure inward: nudge and test mask
        eps = 1500.0
        test_in = self.hf.inside(xs + eps * nx, ys + eps * ny)
        nx = np.where(test_in, nx, -nx); ny = np.where(test_in, ny, -ny)
        idx = np.random.choice(len(xs), min(self.cfg.n_boundary, len(xs)), replace=False)
        bx, by = xs[idx] + eps * nx[idx], ys[idx] + eps * ny[idx]   # just inside
        bnx, bny = nx[idx], ny[idx]
        keep = self.hf.inside(bx, by)
        bx, by, bnx, bny = bx[keep], by[keep], bnx[keep], bny[keep]
        H, Hx, Hy = self.hf.query(np.full(len(bx), self.cfg.year0 + self.T / 2), bx, by)
        tn = np.full(len(bx), 0.5)
        xn, yn = (bx - self.x0) / self.L, (by - self.y0) / self.L
        self.bc = dict(
            txy=self._tensor(np.column_stack([tn, xn, yn])),
            nrm=self._tensor(np.column_stack([bnx, bny])))   # (gx,gy) normal in physical -> same dir in norm
        self.bc["txy"].requires_grad_(True)

    # ---- derivatives & residual
    @staticmethod
    def _grads(n, txy):
        g = torch.autograd.grad(n, txy, torch.ones_like(n), create_graph=True)[0]
        return g[:, 0:1], g[:, 1:2], g[:, 2:3]    # n_t, n_x, n_y  (normalized coords)

    def residual(self):
        p = self.model.params()
        txy = self.col["txy"]
        n = self.model.net(txy)
        n_t, n_x, n_y = self._grads(n, txy)
        n_xx = torch.autograd.grad(n_x, txy, torch.ones_like(n_x), create_graph=True)[0][:, 1:2]
        n_yy = torch.autograd.grad(n_y, txy, torch.ones_like(n_y), create_graph=True)[0][:, 2:3]
        H = self.col["H"].reshape(-1, 1)
        Hx = self.col["Hx"].reshape(-1, 1); Hy = self.col["Hy"].reshape(-1, 1)

        a = p["D0"] * (1.0 - p["alpha"] * H)          # a(H) > 0 by construction
        a_prime = -p["D0"] * p["alpha"]
        div_term = a * (n_xx + n_yy) + a_prime * (Hx * n_x + Hy * n_y)

        r = p["rmax"] * (p["beta"] + (1.0 - p["beta"]) * H)
        K = p["Kmax"] * (p["gamma"] + (1.0 - p["gamma"]) * H)
        reaction = r * n * (1.0 - n / K)

        res = n_t - div_term - reaction
        if self.cfg.use_advection:
            # taxis flux J = -chi * gradH * n  ->  -div(J) added to du/dt
            # div(chi gradH n) = chi (Hxx+Hyy) n + chi (Hx n_x + Hy n_y); we approximate
            # the habitat Laplacian as negligible at NN scale and keep advective transport:
            chi = p["chi"]
            res = res + chi * (Hx * n_x + Hy * n_y)
        return res

    def losses(self, step=None):
        c = self.cfg
        res = self.residual()
        L_res = (res ** 2).mean()
        nd = self.model.net(self.data["txy"])
        L_data = ((nd - self.data["n"]) ** 2).mean()
        ni = self.model.net(self.ic["txy"])
        L_ic = ((ni - self.ic["n"]) ** 2).mean()
        if c.use_bc:
            txy = self.bc["txy"]; nb = self.model.net(txy)
            g = torch.autograd.grad(nb, txy, torch.ones_like(nb), create_graph=True)[0]
            dn_dn = g[:, 1:2] * self.bc["nrm"][:, 0:1] + g[:, 2:3] * self.bc["nrm"][:, 1:2]
            L_bc = (dn_dn ** 2).mean()
        else:
            L_bc = torch.tensor(0.0, device=self.dev)
        # residual-weight ramp (lets data/IC organize the field first)
        wr = c.w_res
        if step is not None and c.res_ramp_steps > 0:
            wr = c.w_res * min(1.0, step / c.res_ramp_steps)
        total = wr * L_res + c.w_data * L_data + c.w_ic * L_ic + c.w_bc * L_bc
        L_prior = torch.tensor(0.0, device=self.dev)
        if c.w_prior > 0 and c.D0_tilde_prior >= 0:
            D0t = self.model.params()["D0"]
            L_prior = (D0t - c.D0_tilde_prior) ** 2
            total = total + c.w_prior * L_prior
        return total, dict(res=L_res.item(), data=L_data.item(),
                           ic=L_ic.item(), bc=float(L_bc.detach()),
                           prior=float(L_prior.detach()) if torch.is_tensor(L_prior) else float(L_prior),
                           total=float(total.detach()))

    # ---- training
    def train(self, verbose=True):
        c = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.adam_steps, eta_min=c.adam_lr * 0.02)
        hist = []
        self.param_hist = []
        for step in range(c.adam_steps):
            opt.zero_grad()
            total, log = self.losses(step)
            total.backward()
            opt.step(); sched.step()
            if verbose and (step % max(1, c.adam_steps // 20) == 0 or step == c.adam_steps - 1):
                p = {k: float(v) for k, v in self.model.params().items()}
                print(f"[adam {step:5d}] tot={log['total']:.3e} res={log['res']:.2e} "
                      f"data={log['data']:.2e} ic={log['ic']:.2e} bc={log['bc']:.2e} "
                      f"| D0~={p['D0']:.3f} a={p['alpha']:.2f} r~={p['rmax']:.3f} "
                      f"b={p['beta']:.2f} K~={p['Kmax']:.2f} g={p['gamma']:.2f}")
            hist.append(log)
            if step % 25 == 0:
                self.param_hist.append((step, {k: float(v) for k, v in self.model.params().items()}))

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
                p = {k: float(v) for k, v in self.model.params().items()}
                print(f"[lbfgs done] tot={log['total']:.3e} res={log['res']:.2e} "
                      f"data={log['data']:.2e} ic={log['ic']:.2e} | "
                      f"D0~={p['D0']:.3f} r~={p['rmax']:.3f} K~={p['Kmax']:.2f}")
            hist.append(log)
        self.hist = hist
        return hist

    # ---- validation metrics (held-out observations, in count units)
    @torch.no_grad()
    def val_metrics(self):
        pred = self.model.net(self.val["txy"]).cpu().numpy().ravel() * self.U
        obs = self.val["counts"]
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        mae = float(np.mean(np.abs(pred - obs)))
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        corr = float(np.corrcoef(pred, obs)[0, 1])
        return dict(rmse=rmse, mae=mae, r2=r2, corr=corr, pred=pred, obs=obs)

    @torch.no_grad()
    def val_glm_metrics(self):
        if not hasattr(self, "val_glm"):
            return None
        pred = self.model.net(self.val_glm["txy"]).cpu().numpy().ravel() * self.U
        u = self.val_glm["u"]
        rmse = float(np.sqrt(np.mean((pred - u) ** 2)))
        ss_res = np.sum((u - pred) ** 2); ss_tot = np.sum((u - u.mean()) ** 2)
        return dict(rmse=rmse, r2=float(1 - ss_res / ss_tot),
                    corr=float(np.corrcoef(pred, u)[0, 1]))

    # ---- physical parameter recovery
    def physical_params(self):
        p = {k: float(v) for k, v in self.model.params().items()}
        L_km = self.L / 1000.0
        return dict(
            D0_km2_per_yr=p["D0"] * L_km ** 2 / self.T,
            alpha=p["alpha"],
            rmax_per_yr=p["rmax"] / self.T,
            beta=p["beta"],
            Kmax_birds=p["Kmax"] * self.U,
            gamma=p["gamma"],
            **({"chi_km2_per_yr": p["chi"] * L_km ** 2 / self.T} if "chi" in p else {}))

    # ---- prediction on a grid (for plotting / metrics)
    @torch.no_grad()
    def predict_grid(self, year, nx=160, ny=170):
        xs = np.linspace(self.hf.x_min, self.hf.x_max, nx)
        ys = np.linspace(self.hf.y_min, self.hf.y_max, ny)
        X, Y = np.meshgrid(xs, ys)
        inside = self.hf.inside(X.ravel(), Y.ravel())
        tn, xn, yn = self.to_norm(np.full(X.size, year), X.ravel(), Y.ravel())
        txy = self._tensor(np.column_stack([tn, xn, yn]))
        n = self.model.net(txy).cpu().numpy().ravel() * self.U
        n[~inside] = np.nan
        return X, Y, n.reshape(Y.shape)
