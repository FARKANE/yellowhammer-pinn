"""
habitat_field.py
================
Continuous, differentiable query of the habitat-quality index H(t, x, y) and its
spatial gradient (dH/dx, dH/dy) for arbitrary scattered points.

The construction reproduces *exactly* the methodology in
`Construction_of_the_habitat_index.ipynb`:

    H0   = u_GLM / u_sat                       (in [0,1])
    Z    = logit(clip(H0, eps, 1-eps))
    Z_es = (G_sigma_s * (Z * M)) / (G_sigma_s * M)        # mask-normalized spatial smoothing
    Z~(t)= sum_k w_k(t) Z_es[:,:,k] / sum_k w_k(t)        # temporal Gaussian weighting
    H    = sigmoid(Z~)
    grad H = H(1-H) grad Z~                                # chain rule

The only addition is a fast scattered-point evaluation:
  * the 17 spatial fields Z_es, dZ_es/dx, dZ_es/dy are precomputed on the 2 km grid
    (exactly as in the notebook), NaNs are nearest-filled so bilinear interpolation
    stays clean just inside the boundary,
  * for a query (year, x, y) we bilinearly interpolate the 17 slices in space and
    apply the *exact* temporal Gaussian weights -> this matches H_at_year /
    spatial_derivatives_H_at_year to machine precision at grid nodes.
"""

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _split_nan_segments(x, y):
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
    """Fill ~valid cells with value of nearest valid cell (so interp doesn't see NaN)."""
    if valid.all():
        return arr2d
    _, (ii, jj) = distance_transform_edt(~valid, return_indices=True)
    return arr2d[ii, jj]


class HabitatField:
    def __init__(self, u_mat_path, map_mat_path,
                 epsilon=1e-6, sigma_s=4.0, sigma_t=1.0,
                 u_sat=15.0, year0=2002, dx_km=2.0, eta_B=1e-12):
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

        # Regular 1D axes (grid is a regular 2 km mesh with NaN padding off-domain)
        self.lat_ax = np.nanmean(Mat_lat, axis=1)   # length ny, increasing
        self.lon_ax = np.nanmean(Mat_long, axis=1)  # placeholder; fix below
        self.lon_ax = np.nanmean(Mat_long, axis=0)  # length nx, increasing
        self.x_min, self.x_max = self.lon_ax.min(), self.lon_ax.max()
        self.y_min, self.y_max = self.lat_ax.min(), self.lat_ax.max()

        # France mask Omega on the grid
        self.M = self._mask_from_contour(Mat_long, Mat_lat, Xfr, Yfr)
        self.Xfr, self.Yfr = Xfr, Yfr
        Mf = self.M.astype(np.float64)

        dx = dy = float(dx_km)  # km units for derivative scaling (as in the notebook)

        # logit field, zeroed outside Omega for convolution
        H0 = self.u_GLM / self.u_sat
        H_eps = np.clip(H0, self.eps, 1.0 - self.eps)
        Z = np.log(H_eps / (1.0 - H_eps)) * Mf[:, :, None]

        B = np.maximum(gaussian_filter(Mf, sigma_s, mode="constant", cval=0.0), eta_B)
        B_y = gaussian_filter(Mf, sigma_s, order=(1, 0), mode="constant", cval=0.0) / dy
        B_x = gaussian_filter(Mf, sigma_s, order=(0, 1), mode="constant", cval=0.0) / dx

        Zes = np.empty((ny, nx, nk))
        dZes_dx = np.empty((ny, nx, nk))
        dZes_dy = np.empty((ny, nx, nk))
        for k in range(nk):
            Ak = gaussian_filter(Z[:, :, k], sigma_s, mode="constant", cval=0.0)
            A_y = gaussian_filter(Z[:, :, k], sigma_s, order=(1, 0), mode="constant", cval=0.0) / dy
            A_x = gaussian_filter(Z[:, :, k], sigma_s, order=(0, 1), mode="constant", cval=0.0) / dx
            Zes[:, :, k] = Ak / B
            # gradient of a quotient
            dZes_dx[:, :, k] = (A_x * B - Ak * B_x) / (B ** 2)
            dZes_dy[:, :, k] = (A_y * B - Ak * B_y) / (B ** 2)

        # derivatives above are per-km (dx_km). Convert to per-metre for PDE in metres.
        dZes_dx /= 1000.0
        dZes_dy /= 1000.0

        # nearest-fill outside the mask so bilinear interp near the coast is clean
        valid = self.M
        for k in range(nk):
            Zes[:, :, k] = _nearest_fill(Zes[:, :, k], valid)
            dZes_dx[:, :, k] = _nearest_fill(dZes_dx[:, :, k], valid)
            dZes_dy[:, :, k] = _nearest_fill(dZes_dy[:, :, k], valid)

        self.Zes, self.dZes_dx, self.dZes_dy = Zes, dZes_dx, dZes_dy

        # bilinear interpolators on (lat, lon) -> per-slice
        pts = (self.lat_ax, self.lon_ax)
        self._iZ = [RegularGridInterpolator(pts, Zes[:, :, k], bounds_error=False,
                    fill_value=None) for k in range(nk)]
        self._iZx = [RegularGridInterpolator(pts, dZes_dx[:, :, k], bounds_error=False,
                     fill_value=None) for k in range(nk)]
        self._iZy = [RegularGridInterpolator(pts, dZes_dy[:, :, k], bounds_error=False,
                     fill_value=None) for k in range(nk)]

        # initial-condition density field g = GLM density at slice 0 (~year0), filled+interp
        g0 = _nearest_fill(self.u_GLM[:, :, 0], valid)
        self._ig0 = RegularGridInterpolator(pts, g0, bounds_error=False, fill_value=None)

        # mask interpolator (for inside-France rejection sampling)
        self._imask = RegularGridInterpolator(pts, Mf, bounds_error=False, fill_value=0.0)

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
            if poly.shape[0] < 3:
                continue
            if not (poly[0, 0] == poly[-1, 0] and poly[0, 1] == poly[-1, 1]):
                poly = np.vstack([poly, poly[0]])
            inside |= Path(poly).contains_points(pts)
        M = np.zeros(Mat_long.shape, dtype=bool)
        M[bbox] = inside
        return M

    def _temporal_weights(self, year):
        year = np.atleast_1d(np.asarray(year, dtype=np.float64))
        t_index = year - self.year0                      # (Q,)
        tau = t_index[:, None] - self.tk[None, :]        # (Q, nk)
        w = np.exp(-0.5 * (tau / self.sigma_t) ** 2)
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)

    # ------------------------------------------------------------ scattered query
    def query(self, year, x, y):
        """Return H, dH/dx, dH/dy (per metre) at scattered points (year, x, y)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        year = np.asarray(year, dtype=np.float64).ravel()
        P = np.column_stack([y, x])                      # interpolators expect (lat, lon)
        nk = self.nk
        Zek = np.stack([self._iZ[k](P) for k in range(nk)], axis=1)   # (Q, nk)
        Zxk = np.stack([self._iZx[k](P) for k in range(nk)], axis=1)
        Zyk = np.stack([self._iZy[k](P) for k in range(nk)], axis=1)
        w = self._temporal_weights(year)                              # (Q, nk)
        Ze = (w * Zek).sum(axis=1)
        dZe_dx = (w * Zxk).sum(axis=1)
        dZe_dy = (w * Zyk).sum(axis=1)
        H = _sigmoid(Ze)
        fac = H * (1.0 - H)
        return H, fac * dZe_dx, fac * dZe_dy

    def g_initial(self, x, y):
        """Initial density field g(x,y) (GLM density at year0) at scattered points."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return self._ig0(np.column_stack([y, x]))

    def inside(self, x, y, thresh=0.5):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return self._imask(np.column_stack([y, x])) >= thresh

    def H_at_year_grid(self, year):
        """Full-grid H (for plotting / FEM-style comparison). NaN outside France."""
        w = self._temporal_weights(year)[0]
        Ze = np.tensordot(self.Zes, w, axes=(2, 0))
        H = _sigmoid(Ze)
        H[~self.M] = np.nan
        return H

    def sample_glm(self, n, seed=0):
        """Sample the dense GLM density field as (year, x, y, u) inside France.
        Slice k is mapped to calendar year year0 + k (notebook convention)."""
        rng = np.random.default_rng(seed)
        ii, jj = np.where(self.M)
        nk = self.nk
        ki = rng.integers(0, nk, size=n)
        si = rng.integers(0, len(ii), size=n)
        I, J = ii[si], jj[si]
        year = self.year0 + ki.astype(np.float64)
        x = self.lon_ax[J]; y = self.lat_ax[I]
        u = self.u_GLM[I, J, ki]
        return year, x, y, u

    def glm_grid_at_slice(self, k):
        """Return (X, Y, U) for GLM slice k (NaN outside France) for plotting/metrics."""
        U = self.u_GLM[:, :, k].copy()
        U[~self.M] = np.nan
        X, Y = np.meshgrid(self.lon_ax, self.lat_ax)
        return X, Y, U

