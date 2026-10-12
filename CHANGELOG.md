# Changelog

## Revision for Control Engineering Practice (branch `revision-cep`)

`paper_config()` / `--preset paper` reproduces the submitted pipeline **bit for bit**
(same sampled points, same initial weights, same trained coefficients; checked
against the original module). Every fix below is a configuration switch, so the
effect of each one can be reported separately.

### Defects fixed (they change the submitted numbers)

| # | Defect in the submitted code | Fix | Switch (legacy value) |
|---|---|---|---|
| 1 | **No-flux boundary condition enforced at a single instant.** Every boundary point had `t = 0.5` (year 2013), so the Neumann condition was absent in all other years and mass could cross the coast. | Boundary points are spread uniformly over 2002–2024. Tangents/normals are computed per polygon (the old code differentiated across the NaN separators between polygons). | `bc_time="uniform"` (`"mid"`) |
| 2 | **Train/test records not disjoint.** `sample_glm` drew (year, node) records *with* replacement, so the same record could be in both sets (expected ≈ 1 % of the 60 000 draws on the 2 km grid, i.e. a few hundred test records also used for training). | Records are drawn without replacement. | `sample_replace=False` (`True`) |
| 3 | **Initial condition overlaps the test set.** IC targets were interpolated from the 2002 slice at random points, i.e. the 2002 test records were seen in training. | IC targets are exact `u_GLM(2002)` values at nodes outside the test (and selection) set / test blocks. | `ic_source="train_nodes"` (`"lhs"`) |
| 4 | **Seed selected on the test records** (best of 5 by test R²), so the reported score was optimistic. | 10 % of the training records (same geometry as the test split) form a *selection* set; seeds and λ_f are chosen there. Mean ± sd over seeds is reported. | `select_frac=0.10` (`0`) |
| 5 | **Ablation "no physics" row scored on its training data** (R² = 0.83 ‡) while the other rows used held-out data. | All field rows of `run_ablation.py` are scored on the same held-out test records. | – |
| 6 | **Experiment 2 notebook could not run**: `habitat`, `DENSITY_SCALE`, `TIME`, `LENGTH`, `normalized_txy`, `OBSERVATION_H`, `_segments` were never defined. | Missing cell added (uses `habitat_field.py`). | – |
| 7 | `HabitatField` hard-coded a 2 km spacing. | Spacing (and its sign) is read from the grid. | – |

### New modelling options (review comments)

| Comment | Addition |
|---|---|
| 1 – the fitted model cannot produce a decline | `reaction="source_sink"`: f = u (r(H) − r_max u / K(H)), β ∈ (β_min, 1), so r(H) < 0 below H* = −β/(1−β). `Trainer.mass_balance()` splits d⟨u⟩/dt into reaction, diffusion (= boundary flux), residual. |
| 2 – circular habitat index | `HabitatField(habitat_mat=...)`: H from a covariate-only predictor (GLM4 BIO + CLC terms **without Year and without Longitude/Latitude**). Built by `tools/build_habitat_covariates.py` (with a self-check against `u_GLM`) or `matlab/build_habitat_covariates.m`. `plot_response_curves` shades the H range not present in the data. |
| 3f – λ₁ says nothing about the fitted model | `FVSolver.principal_eigenvalue()`; λ₁(year) for the identified model. |
| 4 – diffusion form | `diffusion="ecological"`: Δ(a(H) u) (needs ∇H and ΔH, now computed exactly from the smoothed field). Taxis term made exact (ΔH included). |
| 5 – identifiability reporting | `run_profiles.py`: profile of any coefficient on a log grid using the **training objective**, best-of-seeds per point, nominal Wilks interval; λ_f sweep; K_max-anchor sweep. |
| 6 – validation design | `split="block"` (spatial blocks, default 100 km) and `split="temporal"` (`--test-years 2019-2024`); forward PDE simulation with the identified coefficients (`diagnostics.forward_validation`); residual on an independent dense set and relative to \|n_t\|. |
| 7 – classical baseline | `run_output_error_baseline.py`: output-error least squares through a finite-volume solver (discrete adjoint by autodiff), same field, model and split; compares coefficients, held-out fit and time with the PINN. |
| 8 – count pilot | Experiment 2: RF with the same map inputs (H + GLM offset), GLM-offset-alone row, no-physics hybrid on by default, Nadeau–Bengio corrected resampled t-test. |

### Other
* `Trainer.save()` / `Trainer.from_checkpoint()` (old `pinn_weights.pt` files still load).
* `tests/`: synthetic France-shaped data from a known model, unit tests (`pytest`), and
  `tests/synthetic_identification.py` (identification against known coefficients).
* Data file default is now `u_GLM_2002_2024.mat` (23 slices).
