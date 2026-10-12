# Changelog

## Version 2

`v1_config()` / `--preset v1` reproduces version 1 of this code **bit for bit**
(same sampled points, same initial weights, same trained coefficients; checked
against the original module). Every fix below is a configuration switch, so the
effect of each one can be measured separately.

### Defects fixed (they change the version-1 numbers)

| # | Defect in version 1 | Fix | Switch (version-1 value) |
|---|---|---|---|
| 1 | **No-flux boundary condition enforced at a single instant.** Every boundary point had `t = 0.5` (year 2013), so the Neumann condition was absent in all other years and mass could cross the coast. | Boundary points are spread uniformly over 2002–2024. Tangents/normals are computed per polygon (version 1 differentiated across the NaN separators between polygons). | `bc_time="uniform"` (`"mid"`) |
| 2 | **Train/test records not disjoint.** `sample_glm` drew (year, node) records *with* replacement, so the same record could be in both sets (expected ≈ 1 % of the 60 000 draws on the 2 km grid). | Records are drawn without replacement. | `sample_replace=False` (`True`) |
| 3 | **Initial condition overlaps the test set.** IC targets were interpolated from the 2002 slice at random points, so the 2002 test records were seen in training. | IC targets are exact `u_GLM(2002)` values at nodes outside the test (and selection) set / test blocks. | `ic_source="train_nodes"` (`"lhs"`) |
| 4 | **Seed selected on the test records** (best of 5 by test R²), so the reported score was optimistic. | 10 % of the training records (same geometry as the test split) form a *selection* set; seeds and λ_f are chosen there. Mean ± sd over seeds is reported. | `select_frac=0.10` (`0`) |
| 5 | **Ablation "no physics" row scored on its training data** while the other rows used held-out data. | All field rows of `run_ablation.py` are scored on the same held-out test records. | – |
| 6 | **Experiment 2 notebook could not run**: `habitat`, `DENSITY_SCALE`, `TIME`, `LENGTH`, `normalized_txy`, `OBSERVATION_H`, `_segments` were never defined. | Missing cell added (uses `habitat_field.py`). | – |
| 7 | `HabitatField` hard-coded a 2 km spacing. | Spacing (and its sign) is read from the grid. | – |

### New options

| Topic | Addition |
|---|---|
| Population decline | `reaction="source_sink"`: f = u (r(H) − r_max u / K(H)), β ∈ (β_min, 1), so r(H) < 0 below H* = −β/(1−β). With the logistic form and r(H) > 0 everywhere, the model cannot produce a decline under zero-flux boundaries. `Trainer.mass_balance()` splits d⟨u⟩/dt into reaction, diffusion (= boundary flux) and residual. |
| Habitat index | `HabitatField(habitat_mat=...)`: H from a covariate-only predictor (GLM4 BIO + CLC terms **without Year and without Longitude/Latitude**), instead of a smoothed copy of `u_GLM`. Built by `tools/build_habitat_covariates.py` (with a self-check against `u_GLM`) or `matlab/build_habitat_covariates.m`. `plot_response_curves` shades the H range not present in the data. |
| Persistence indicator | `FVSolver.principal_eigenvalue()`; λ₁(year) for the identified model. |
| Diffusion form | `diffusion="ecological"`: Δ(a(H) u) (uses ∇H and ΔH, computed exactly from the smoothed field). Taxis term made exact (ΔH included). |
| Identifiability | `run_profiles.py`: profile of any coefficient on a log grid using the **training objective**, best-of-seeds per point, nominal Wilks interval; λ_f sweep; K_max-anchor sweep. |
| Validation | `split="block"` (spatial blocks, default 100 km) and `split="temporal"` (`--test-years 2019-2024`); forward PDE simulation with the identified coefficients (`diagnostics.forward_validation`); residual on an independent dense set and relative to \|n_t\|. |
| Classical baseline | `run_output_error_baseline.py`: output-error least squares through a finite-volume solver (discrete adjoint by autodiff), same field, model and split; compares coefficients, held-out fit and time with the PINN. `--init-from-pinn` gives a two-stage PINN → output-error estimate. |
| Count experiment | Experiment 2: RF with the same map inputs (H + GLM offset), GLM-offset-alone row, no-physics hybrid on by default, Nadeau–Bengio corrected resampled t-test. |

### Other
* `Trainer.save()` / `Trainer.from_checkpoint()` (version-1 `pinn_weights.pt` files still load).
* `tests/`: synthetic France-shaped data from a known model, unit tests (`pytest`), and
  `tests/synthetic_identification.py` (identification against known coefficients).
* Data file default is now `u_GLM_2002_2024.mat` (23 slices, 2002–2024).
