"""
Unit tests on synthetic data (no real data needed):   python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from habitat_field import HabitatField                                    # noqa: E402
from forward_solver import FVSolver                                       # noqa: E402
from pinn_yellowhammer import Trainer, Config, v1_config, recommended_config  # noqa: E402
from synthetic import make                                                # noqa: E402

TINY = dict(n_collocation=300, n_ic=150, n_boundary=100, n_glm=3000, layers=(3, 16, 16, 1),
            fourier_m=8, adam_steps=20, lbfgs_steps=0, res_ramp_steps=10, device="cpu")


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("synth"))
    make(d, spacing_km=20.0, n_sites=150, visits_per_site=6)
    return dict(u_mat=os.path.join(d, "u_GLM_2002_2024.mat"), map_mat=os.path.join(d, "Map_France.mat"),
                csv=os.path.join(d, "Yellowhammer_2002_2024.csv"),
                habitat_mat=os.path.join(d, "H_covariates_2002_2024.mat"))


@pytest.fixture(scope="session")
def hf(data):
    return HabitatField(data["u_mat"], data["map_mat"], habitat_mat=data["habitat_mat"], with_laplacian=True)


# ------------------------------------------------------------------ habitat
def test_spacing_inferred(hf):
    assert abs(abs(hf.dx_m) - 20000.0) < 1e-6 and abs(abs(hf.dy_m) - 20000.0) < 1e-6


def test_laplacian_matches_finite_differences(hf):
    from scipy.ndimage import binary_erosion
    Hg = hf.H_at_year_grid(2010.0)
    h = hf.dx_m
    fd = (Hg[2:, 1:-1] + Hg[:-2, 1:-1] + Hg[1:-1, 2:] + Hg[1:-1, :-2] - 4 * Hg[1:-1, 1:-1]) / h ** 2
    inner = binary_erosion(hf.M, iterations=6)[1:-1, 1:-1]
    X, Y = np.meshgrid(hf.lon_ax[1:-1], hf.lat_ax[1:-1])
    lap = hf.query(np.full(inner.sum(), 2010.0), X[inner], Y[inner], laplacian=True)[3]
    assert np.corrcoef(lap, fd[inner])[0, 1] > 0.99


def test_sampling_without_replacement(hf):
    r = hf.sample_glm_records(3000, seed=1)
    key = r["k"] * 10 ** 8 + r["I"] * 10 ** 4 + r["J"]
    assert len(np.unique(key)) == len(key)


def test_legacy_sampler_stream(hf):
    a = hf.sample_glm(500, seed=3)
    b = hf.sample_glm_records(500, seed=3, replace=True)
    assert all(np.array_equal(x, b[k]) for x, k in zip(a, ["year", "x", "y", "u"]))


# ------------------------------------------------------------------ solver
@pytest.mark.parametrize("diffusion", ["fickian", "ecological"])
def test_solver_conserves_mass_and_positivity(hf, diffusion):
    s = FVSolver(hf, coarsen=1, diffusion=diffusion, reaction="source_sink")
    with torch.no_grad():
        sim = s.simulate(dict(D0=800.0, alpha=0.6, rmax=0.0, beta=0.5, Kmax=18.0, gamma=0.4))
    m = sim.mean(1).numpy()
    assert np.max(np.abs(m - m[0])) / m[0] < 1e-12
    assert float(sim.min()) >= 0.0


@pytest.mark.parametrize("diffusion", ["fickian", "ecological"])
def test_principal_eigenvalue(hf, diffusion):
    s = FVSolver(hf, coarsen=2, diffusion=diffusion, reaction="source_sink")
    assert abs(s.principal_eigenvalue(dict(D0=800., alpha=0.6, rmax=0.3, beta=1.0, Kmax=18., gamma=0.4), 2010) + 0.3) < 1e-8
    p = dict(D0=300., alpha=0.6, rmax=0.35, beta=-0.6, Kmax=18., gamma=0.4)
    H = s.H([2010])[:, 0]
    A = np.zeros((s.N, s.N))
    for i in range(s.N):                                    # dense linearised operator
        e = torch.zeros(s.N, dtype=torch.float64); e[i] = 1e-9
        d, r = s.terms(e, H, p)
        A[:, i] = ((d + r) / 1e-9).numpy()
    assert abs(-np.linalg.eigvals(A).real.max() - s.principal_eigenvalue(p, 2010)) < 1e-6


# ------------------------------------------------------------------ trainer
def test_v1_preset_reproduces_v1_protocol(data):
    tr = Trainer(v1_config(**{k: v for k, v in data.items() if k != "habitat_mat"}, **TINY))
    assert torch.unique(tr.bc["txy"][:, 0]).tolist() == [0.5]          # the version-1 defect
    assert tr.sel is None


def test_recommended_protocol_is_disjoint(data):
    tr = Trainer(recommended_config(**data, **TINY))
    rec = tr.rec
    key = lambda ii: set(zip(rec["k"][ii], rec["I"][ii], rec["J"][ii]))
    assert not key(tr.idx_train) & key(tr.idx_test)
    assert not key(tr.idx_select) & key(tr.idx_test)
    bt = set(tr._block_ids(rec["x"][tr.idx_train], rec["y"][tr.idx_train]))
    assert not bt & set(tr.test_blocks)
    assert not set(tr._block_ids(*tr.ic_xy)) & set(tr.test_blocks)        # IC avoids test blocks
    t = tr.bc["txy"][:, 0]
    assert float(t.min()) < 0.1 and float(t.max()) > 0.9                  # BC spread in time


def test_temporal_split(data):
    tr = Trainer(recommended_config(**data, **TINY, split="temporal", test_years=tuple(range(2019, 2025))))
    assert set(np.unique(tr.rec["year"][tr.idx_test])) <= set(range(2019, 2025))
    assert not set(np.unique(tr.rec["year"][tr.idx_train])) & set(range(2019, 2025))


@pytest.mark.parametrize("kw", [dict(), dict(diffusion="ecological"), dict(use_advection=True),
                                dict(reaction="logistic")])
def test_train_and_mass_balance_identity(data, kw):
    tr = Trainer(recommended_config(**data, **TINY, **kw))
    tr.train(verbose=False)
    rows = tr.mass_balance(years=[2005, 2015], nx=30, ny=30, n_boundary=300)
    for r in rows:
        s = r["diffusion"] + r["reaction"] + r["advection"] + r["residual"]
        assert abs(s - r["dU_dt"]) < 1e-4 * (1 + abs(r["dU_dt"]))
    d = tr.dense_residual_stats(n=500)
    assert np.isfinite(d["rms_residual"])


def test_fixed_parameters_and_checkpoint(data, tmp_path):
    cfg = recommended_config(**data, **TINY, fixed=dict(D0=0.0, beta=-0.3))
    tr = Trainer(cfg)
    tr.train(verbose=False)
    p = tr.physical_params()
    assert p["D0_km2_per_yr"] == 0.0 and abs(p["beta"] + 0.3) < 1e-7
    f = tmp_path / "ck.pt"
    tr.save(str(f))
    tr2 = Trainer.from_checkpoint(str(f), device="cpu")
    assert torch.equal(tr2.model.net(tr.val_glm["txy"]), tr.model.net(tr.val_glm["txy"]))
