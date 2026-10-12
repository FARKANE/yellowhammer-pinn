"""
habitat_field.py
================
Continuous query of the habitat-quality index H(t, x, y), its spatial gradient
and (optionally) its Laplacian at scattered points, plus grid helpers used by the
finite-volume solver (forward_solver.py).

Two habitat sources are supported
---------------------------------
* ``habitat_mat=None`` (legacy, first submission): the index is derived from the
  denoised field itself,

      H0 = u_GLM / u_sat,   Z = logit(clip(H0, eps, 1 - eps)).

  Because u_GLM is also the PINN training target, this construction is
  *circular*: K(H) and r(H) become functions of a smoothed copy of the target,
  and the GLM Year trend (the national decline) is inside H.  It is kept only to
  reproduce the submitted results.

* ``habitat_mat="H_covariates_2002_2024.mat"`` (recommended): the index is
  derived from an independent, covariate-only linear predictor ``eta_hab`` (the
  GLM4 land-cover + bioclimatic terms WITHOUT the Year trend and WITHOUT the
  Longitude/Latitude terms; see tools/build_habitat_covariates.py).  It is
  standardised over the domain,

      Z = (eta_hab - mean_Omega) / sd_Omega.

Both sources then follow the same pipeline as the habitat notebook:

    Z_es   = (G_s * (Z M)) / (G_s * M)                 mask-normalised smoothing
    Z~(t)  = sum_k w_k(t) Z_es[:, :, k] / sum_k w_k(t)   Gaussian temporal weights
    H      = sigmoid(Z~)
    grad H = H (1 - H) grad Z~
    lap  H = H (1 - H) lap Z~ + H (1 - H)(1 - 2H) |grad Z~|^2

Spatial derivatives are returned per metre (Lambert-93 coordinates).
"""

import warnings

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _split_nan_segments(x, y):
    """Split a NaN-separated contour (Xfr, Yfr) into polygons of >= 3 vertices."""
    isn = np.isnan(x) | np.isnan(y)
    idx = np.where(isn)[0]
    segments, start = [], 0
    for j in idx:
        if j - start >= 3:
            segments.append((x[start:j], y[start:j]))
        start = j + 1
    if len(x) - start >= 3:
        segments.append((x[start:], y[start:]))
    return segments


def _nearest_fill(arr2d, valid):
    """Fill ~valid cells with the value of the nearest valid cell."""
    if valid.all():
        return arr2d
    _, (ii, jj) = distance_transform_edt(~valid, return_indices=True)
    return arr2d[ii, jj]


class HabitatField:
    def __init__(self, u_mat_path, map_mat_path,
                 epsilon=1e-6, sigma_s=4.0, sigma_t=1.0,
                 u_sat=15.0, year0=2002, dx_km=None, eta_B=1e-12,
                 habitat_mat=None, habitat_var="eta_hab",
                 with_laplacian=False):
        u_data = loadmat(u_mat_path)
        m_data = loadmat(map_mat_path)

        self.u_GLM = np.asarray(u_data["u_GLM"], dtype=np.float64)      # (ny, nx, nk)
        Mat_lat = np.asarray(u_data["Mat_lat"], dtype=np.float64)       # (ny, nx)
        Mat_long = np.asarray(u_data["Mat_long"], dtype=np.float64)
        Xfr = np.asarray(m_data["Xfr"], dtype=np.float64).squeeze()
        Yfr = np.asarray(m_data["Yfr"], dtype=np.float64).squeeze()

        self.eps, self.sigma_s, self.sigma_t = epsilon, sigma_s, sigma_t
        self.u_sat, self.year0 = float(u_sat), int(year0)
        ny, nx, nk = self.u_GLM.shape
        self.ny, self.nx, self.nk = ny, nx, nk
        self.tk = np.arange(nk, dtype=np.float64)

        # Regular 1-D axes (regular mesh with NaN padding off-domain)
        self.lat_ax = np.nanmean(Mat_lat, axis=1)   # length ny
        self.lon_ax = np.nanmean(Mat_long, axis=0)  # length nx
        self.x_min, self.x_max = self.lon_ax.min(), self.lon_ax.max()
        self.y_min, self.y_max = self.lat_ax.min(), self.lat_ax.max()

        # Grid spacing in metres, signed (derivatives stay correct whatever the
        # orientation of the rows).  Legacy code hard-coded dx = dy = 2 km.
        dx_m = float(np.median(np.diff(self.lon_ax)))
        dy_m = float(np.median(np.diff(self.lat_ax)))
        if dx_km is not None:
            for name, d in (("x", dx_m), ("y", dy_m)):
                if abs(abs(d) - 1000.0 * dx_km) > 0.01 * 1000.0 * dx_km:
                    warnings.warn(f"dx_km={dx_km} differs from the inferred {name}-spacing "
                                  f"{abs(d) / 1000:.3f} km; using dx_km.")
            dx_m = np.sign(dx_m) * 1000.0 * dx_km
            dy_m = np.sign(dy_m) * 1000.0 * dx_km
        self.dx_m, self.dy_m = dx_m, dy_m

        # France mask Omega on the grid
        self.M = self._mask_from_contour(Mat_long, Mat_lat, Xfr, Yfr)
        self.Xfr, self.Yfr = Xfr, Yfr
        Mf = self.M.astype(np.float64)

        # ---------------------------------------------------------------- Z
        if habitat_mat:
            h_data = loadmat(habitat_mat)
            if habitat_var not in h_data:
                raise KeyError(f"'{habitat_var}' not found in {habitat_mat}; "
                               f"variables: {[k for k in h_data if not k.startswith('__')]}")
            eta = np.asarray(h_data[habitat_var], dtype=np.float64)
            if eta.shape != self.u_GLM.shape:
                raise ValueError(f"{habitat_var} has shape {eta.shape}, expected {self.u_GLM.shape}")
            vals = eta[self.M]                                         # (n_inside, nk)
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            Z = np.nan_to_num((eta - mu) / max(sd, 1e-12)) * Mf[:, :, None]
            self.habitat_source = "covariates"
            self.eta_standardisation = (float(mu), float(sd))
        else:
            H0 = self.u_GLM / self.u_sat
            H_eps = np.clip(H0, self.eps, 1.0 - self.eps)
            Z = np.nan_to_num(np.log(H_eps / (1.0 - H_eps))) * Mf[:, :, None]
            self.habitat_source = "u_glm"
            self.eta_standardisation = None

        # ------------------------------------------- smoothing and derivatives
        g = lambda A, order=0: gaussian_filter(A, sigma_s, order=order, mode="constant", cval=0.0)
        B = np.maximum(g(Mf), eta_B)
        B_y = g(Mf, (1, 0)) / dy_m
        B_x = g(Mf, (0, 1)) / dx_m
        if with_laplacian:
            B_lap = g(Mf, (2, 0)) / dy_m ** 2 + g(Mf, (0, 2)) / dx_m ** 2
            gradB2 = B_x ** 2 + B_y ** 2

        Zes = np.empty((ny, nx, nk))
        dZx = np.empty((ny, nx, nk))
        dZy = np.empty((ny, nx, nk))
        lapZ = np.empty((ny, nx, nk)) if with_laplacian else None
        for k in range(nk):
            Zk = Z[:, :, k]
            A = g(Zk)
            A_y = g(Zk, (1, 0)) / dy_m
            A_x = g(Zk, (0, 1)) / dx_m
            Zes[:, :, k] = A / B
            dZx[:, :, k] = (A_x * B - A * B_x) / B ** 2               # quotient rule
            dZy[:, :, k] = (A_y * B - A * B_y) / B ** 2
            if with_laplacian:
                A_lap = g(Zk, (2, 0)) / dy_m ** 2 + g(Zk, (0, 2)) / dx_m ** 2
                lapZ[:, :, k] = (A_lap / B - 2.0 * (A_x * B_x + A_y * B_y) / B ** 2
                                 - A * B_lap / B ** 2 + 2.0 * A * gradB2 / B ** 3)

        # nearest-fill outside the mask so bilinear interpolation near the coast is clean
        valid = self.M
        fields = [Zes, dZx, dZy] + ([lapZ] if with_laplacian else [])
        for F in fields:
            for k in range(nk):
                F[:, :, k] = _nearest_fill(F[:, :, k], valid)
        self.Zes, self.dZes_dx, self.dZes_dy, self.lapZes = Zes, dZx, dZy, lapZ
        self.with_laplacian = with_laplacian

        # one bilinear interpolator per field; values (ny, nx, nk) -> query returns (Q, nk)
        pts = (self.lat_ax, self.lon_ax)
        rgi = lambda V: RegularGridInterpolator(pts, V, bounds_error=False, fill_value=None)
        self._iZ, self._iZx, self._iZy = rgi(Zes), rgi(dZx), rgi(dZy)
        self._iLap = rgi(lapZ) if with_laplacian else None

        g0 = _nearest_fill(self.u_GLM[:, :, 0], valid)
        self._ig0 = rgi(g0)
        self._imask = RegularGridInterpolator(pts, Mf, bounds_error=False, fill_value=0.0)

    # ------------------------------------------------- aliases (Experiment 2)
    mask = property(lambda self: self.M)
    u_glm = property(lambda self: self.u_GLM)
    lon_axis = property(lambda self: self.lon_ax)
    lat_axis = property(lambda self: self.lat_ax)
    xfr = property(lambda self: self.Xfr)
    yfr = property(lambda self: self.Yfr)

    # ----------------------------------------------------------------- mask
    @staticmethod
    def _mask_from_contour(Mat_long, Mat_lat, Xfr, Yfr):
        base = np.isfinite(Mat_long) & np.isfinite(Mat_lat)
        xmin, xmax = np.nanmin(Xfr), np.nanmax(Xfr)
        ymin, ymax = np.nanmin(Yfr), np.nanmax(Yfr)
        bbox = base & (Mat_long >= xmin) & (Mat_long <= xmax) & \
               (Mat_lat >= ymin) & (Mat_lat <= ymax)
        pts = np.column_stack([Mat_long[bbox].ravel(), Mat_lat[bbox].ravel()])
        inside = np.zeros(pts.shape[0], dtype=bool)
        for xs, ys in _split_nan_segments(Xfr, Yfr):
            poly = np.column_stack([xs, ys])
            if not (poly[0, 0] == poly[-1, 0] and poly[0, 1] == poly[-1, 1]):
                poly = np.vstack([poly, poly[0]])
            inside |= Path(poly).contains_points(pts)
        M = np.zeros(Mat_long.shape, dtype=bool)
        M[bbox] = inside
        return M

    def temporal_weights(self, year):
        year = np.atleast_1d(np.asarray(year, dtype=np.float64))
        tau = (year - self.year0)[:, None] - self.tk[None, :]
        w = np.exp(-0.5 * (tau / self.sigma_t) ** 2)
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)

    _temporal_weights = temporal_weights          # backward-compatible name

    # ------------------------------------------------------------ scattered query
    def query(self, year, x, y, laplacian=False):
        """H, dH/dx, dH/dy (per metre) [, lap H (per metre^2)] at points (year, x, y)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        year = np.broadcast_to(np.asarray(year, dtype=np.float64).ravel(), x.shape)
        P = np.column_stack([y, x])                       # interpolators expect (lat, lon)
        w = self.temporal_weights(year)                   # (Q, nk)
        Ze = (w * self._iZ(P)).sum(axis=1)
        Zx = (w * self._iZx(P)).sum(axis=1)
        Zy = (w * self._iZy(P)).sum(axis=1)
        H = _sigmoid(Ze)
        fac = H * (1.0 - H)
        if not laplacian:
            return H, fac * Zx, fac * Zy
        if self._iLap is None:
            raise RuntimeError("HabitatField was built with with_laplacian=False")
        Zl = (w * self._iLap(P)).sum(axis=1)
        lapH = fac * Zl + fac * (1.0 - 2.0 * H) * (Zx ** 2 + Zy ** 2)
        return H, fac * Zx, fac * Zy, lapH

    def Z_slices_at(self, x, y):
        """Smoothed Z_es slices at points (Q, nk); H(t) = sigmoid(Z_slices @ w(t))."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return self._iZ(np.column_stack([y, x]))

    def g_initial(self, x, y):
        """Initial density field (u_GLM slice 0) at scattered points."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return self._ig0(np.column_stack([y, x]))

    def inside(self, x, y, thresh=0.5):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return self._imask(np.column_stack([y, x])) >= thresh

    def H_at_year_grid(self, year):
        """Full-grid H (NaN outside France)."""
        w = self.temporal_weights(year)[0]
        H = _sigmoid(np.tensordot(self.Zes, w, axes=(2, 0)))
        H[~self.M] = np.nan
        return H

    # --------------------------------------------------------------- sampling
    def sample_glm(self, n, seed=0):
        """LEGACY sampler (first submission): draws (slice, node) WITH replacement,
        so identical records can appear in both training and test sets.  Kept only
        to reproduce the submitted numbers; use sample_glm_records instead."""
        rng = np.random.default_rng(seed)
        ii, jj = np.where(self.M)
        ki = rng.integers(0, self.nk, size=n)
        si = rng.integers(0, len(ii), size=n)
        I, J = ii[si], jj[si]
        year = self.year0 + ki.astype(np.float64)
        return year, self.lon_ax[J], self.lat_ax[I], self.u_GLM[I, J, ki]

    def sample_glm_records(self, n, seed=0, replace=False):
        """Sample n (slice, node) records of the GLM field inside France.

        replace=False (default) draws distinct records.  replace=True reproduces the
        legacy sampler exactly (same RNG stream).  Returns a dict with year, x, y, u
        and the grid indices I, J, k (needed for block / temporal splits)."""
        ii, jj = np.where(self.M)
        if replace:
            rng = np.random.default_rng(seed)
            ki = rng.integers(0, self.nk, size=n)
            si = rng.integers(0, len(ii), size=n)
        else:
            rng = np.random.default_rng(seed)
            total = len(ii) * self.nk
            if n > total:
                raise ValueError(f"n={n} exceeds the {total} available records")
            flat = rng.choice(total, size=n, replace=False)
            ki, si = flat // len(ii), flat % len(ii)
        I, J = ii[si], jj[si]
        return dict(year=self.year0 + ki.astype(np.float64), x=self.lon_ax[J], y=self.lat_ax[I],
                    u=self.u_GLM[I, J, ki], I=I, J=J, k=ki)

    def glm_grid_at_slice(self, k):
        """(X, Y, U) for GLM slice k (NaN outside France)."""
        U = self.u_GLM[:, :, k].copy()
        U[~self.M] = np.nan
        X, Y = np.meshgrid(self.lon_ax, self.lat_ax)
        return X, Y, U

    # --------------------------------------------------------------- boundary
    def boundary_points(self, eps=1500.0):
        """Points just inside the contour with INWARD unit normals and arc-length
        weights ds (metres).  Tangents are computed per polygon (the legacy code
        differentiated across the NaN separators between polygons)."""
        out = []
        for xs, ys in _split_nan_segments(self.Xfr, self.Yfr):
            tx, ty = np.gradient(xs), np.gradient(ys)
            ds = np.hypot(tx, ty)
            nrm = ds + 1e-12
            nx, ny = ty / nrm, -tx / nrm
            flip = ~self.inside(xs + eps * nx, ys + eps * ny)
            nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)
            bx, by = xs + eps * nx, ys + eps * ny
            keep = self.inside(bx, by) & (ds > 0)
            out.append(np.column_stack([bx[keep], by[keep], nx[keep], ny[keep], ds[keep]]))
        P = np.concatenate(out)
        return dict(x=P[:, 0], y=P[:, 1], nx=P[:, 2], ny=P[:, 3], ds=P[:, 4])

    # --------------------------------------------------------- grid helpers
    def domain_area_m2(self):
        return float(self.M.sum()) * abs(self.dx_m * self.dy_m)

    def coarse_grid(self, factor=5, min_frac=0.5):
        """Block-average the grid by `factor`.  A coarse cell belongs to the domain
        when at least `min_frac` of its fine cells are inside France.  Returns the
        coarse mask, cell-centre coordinates (metres), spacing, and the
        block-averaged reference field u_ref (nk, n_cells)."""
        f = int(factor)
        ny_c, nx_c = int(np.ceil(self.ny / f)), int(np.ceil(self.nx / f))
        pad = ((0, ny_c * f - self.ny), (0, nx_c * f - self.nx))
        Mf = np.pad(self.M.astype(float), pad)
        frac = Mf.reshape(ny_c, f, nx_c, f).mean(axis=(1, 3))
        cnt = Mf.reshape(ny_c, f, nx_c, f).sum(axis=(1, 3))
        mask_c = frac >= min_frac
        # centre coordinates of each coarse block (in metres)
        lon = np.concatenate([self.lon_ax, self.lon_ax[-1] + self.dx_m * np.arange(1, pad[1][1] + 1)])
        lat = np.concatenate([self.lat_ax, self.lat_ax[-1] + self.dy_m * np.arange(1, pad[0][1] + 1)])
        xc_ax = lon.reshape(nx_c, f).mean(axis=1)
        yc_ax = lat.reshape(ny_c, f).mean(axis=1)
        u_ref = np.empty((self.nk, int(mask_c.sum())))
        for k in range(self.nk):
            Uk = np.pad(np.where(self.M, self.u_GLM[:, :, k], 0.0), pad)
            s = Uk.reshape(ny_c, f, nx_c, f).sum(axis=(1, 3))
            u_ref[k] = (s / np.maximum(cnt, 1))[mask_c]
        Xc, Yc = np.meshgrid(xc_ax, yc_ax)
        return dict(mask=mask_c, x_ax=xc_ax, y_ax=yc_ax, x=Xc[mask_c], y=Yc[mask_c],
                    dx_m=abs(self.dx_m) * f, dy_m=abs(self.dy_m) * f, u_ref=u_ref, factor=f)
