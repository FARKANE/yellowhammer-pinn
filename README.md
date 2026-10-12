# Habitat-Modulated Population Dynamics via Physics-Informed Neural Networks

Reproducible code for the paper *A Distributed Parameter Identification Approach
for Habitat-Modulated Population Dynamics via Physics-Informed Neural Networks*
(A. Alassani, A. Farkane, D. Lassounon, R. Ouvrard, T. Poinot).

A habitat-modulated reaction–diffusion model of Yellowhammer
(*Emberiza citrinella*) breeding density across mainland France is identified
from French Breeding Bird Survey data (2002–2024) with physics-informed
neural networks (PINNs), and checked against a finite-volume solver and a
classical output-error identification.

> **Revision.** The corrected protocol is the default (`--preset revised`).
> `--preset paper` reproduces the submitted pipeline bit for bit. See
> [`CHANGELOG.md`](CHANGELOG.md) for every fix and the review comment it answers.

## Repository contents

| File | Description |
|---|---|
| `pinn_yellowhammer.py` | PINN model, trainer, configurations (`paper_config`, `revised_config`), diagnostics (mass balance, dense residual) |
| `habitat_field.py` | Habitat index H(t,x,y), its gradient and Laplacian; legacy (from `u_GLM`) or covariate-only source |
| `forward_solver.py` | Finite-volume solver with zero-flux boundaries, principal eigenvalue λ₁, output-error identification |
| `diagnostics.py` | Forward-simulation validation, λ₁ series, figures |
| `run_experiment1.py` | Field reconstruction + identification, several seeds, all diagnostics |
| `run_profiles.py` | Profiles of any coefficient (log grid), λ_f and K_max sweeps |
| `run_ablation.py` | Table-3 ablation, every row scored on the same held-out records |
| `run_output_error_baseline.py` | Classical output-error identification on the same field/model/split |
| `tools/build_habitat_covariates.py` | Covariate-only habitat predictor (GLM4 without Year and Lon/Lat) |
| `matlab/build_habitat_covariates.m` | Same in Matlab |
| `Experiment1_Reconstruction_portable.ipynb` | Notebook version of Experiment 1 (Colab or local) |
| `Experiment2_Hybrid_RF_portable.ipynb` | Hybrid count-prediction PINN vs. Random Forest (+ RF with the same maps, GLM offset, corrected t-test) |
| `tests/` | Synthetic data from a known model, unit tests, identification check |

## Data

Place these files in a data directory (they are **not** included here):

| File | Origin |
|---|---|
| `u_GLM_2002_2024.mat` | GLM4 niche-model field, 23 annual slices on the 2 km grid (`Matlab_src/LIAS_internal_report_production.m` of the benchmark) |
| `Map_France.mat` | benchmark |
| `Yellowhammer_2002_2024.csv` | benchmark (Year, Longitude, Latitude, EMBCIT) |
| `H_covariates_2002_2024.mat` *(recommended)* | `tools/build_habitat_covariates.py` (needs `Coef_GLM_final_model.txt`, `Indice_BIO_2002_2024.mat`, `Portions_CLC{2000,2006,2012,2018}.mat`) |
| Experiment 2 only | `Yellowhammer_Clim_Bioclim_CLC_2002_2024.csv`, `Data_Estimation_70_V1.csv`, `Data_Validation_30_V1.csv` |

Benchmark: https://github.com/lias-laboratory/yellowhammer-benchmark

### Habitat index
Without `--habitat-mat`, H is derived from `u_GLM` itself (the training target), as in
the submission; the coefficients are then conditional on that circular construction
and a warning is printed. To build an independent index:

```bash
python tools/build_habitat_covariates.py --data-dir data --out H_covariates_2002_2024.mat --check-u-glm
```
`--check-u-glm` rebuilds the full GLM4 field from the same files and must print `PASS`
(it verifies the coefficient order and the CLC layers).

## Running

```bash
pip install -r requirements.txt

# Experiment 1, corrected protocol: 100 km spatial-block test set, selection subset,
# source-sink reaction, 5 seeds, all diagnostics and figures in ./outputs
python run_experiment1.py --data-dir data --habitat-mat H_covariates_2002_2024.mat

# temporal hold-out: fit 2002-2018, predict 2019-2024
python run_experiment1.py --data-dir data --test-years 2019-2024 --out-dir outputs_temporal

# the submitted pipeline, unchanged
python run_experiment1.py --data-dir data --preset paper --out-dir outputs_paper

# identifiability and sensitivity
python run_profiles.py --data-dir data --param D0 --values 0,1,3,10,30,100,300,1000,3000 --seeds 0 1 2
python run_profiles.py --data-dir data --param rmax --values 0.01,0.03,0.1,0.3,1
python run_profiles.py --data-dir data --lambda-sweep 0.05,0.15,0.5,1 --seeds 0 1 2 3 4
python run_profiles.py --data-dir data --kmax-sweep 0.8,1.0,1.2,1.6,2.0

# ablation (Table 3) and classical baseline
python run_ablation.py --data-dir data --seeds 0 1 2
python run_output_error_baseline.py --data-dir data --pinn-summary outputs/summary.json
```

Main options (all scripts): `--reaction {logistic,source_sink}`, `--diffusion {fickian,ecological}`,
`--split {random,block,temporal}`, `--block-km`, `--test-years`, `--lambda-f`, `--kmax-tilde`,
`--habitat-mat`, `--quick` (tiny smoke test). A GPU is used when available.

`run_experiment1.py` writes `summary.json`, `runs.csv` (every seed), `mass_balance.csv`,
`forward_simulation_per_year.csv`, checkpoints and the figures (`fig_mass_balance`,
`fig_trajectory_simulation`, `fig_lambda1`, `fig_response_curves`, density maps, calibration,
all-years panel).

### What the diagnostics mean
* **Mass balance**: with zero-flux boundaries, the domain-mean change must be explained by the
  reaction term. If it is carried by the PDE residual or by a boundary flux, the identified PDE
  does not explain the observed trend.
* **Forward simulation**: the PDE is solved from the 2002 field with the identified coefficients
  (10 km finite volumes) and compared with the reference every year, on all cells and on the
  held-out cells/years. This is the output-error validation of the identified model.
* **λ₁(year)**: principal eigenvalue with the habitat of that year; λ₁ < 0 means persistence.

## Notebooks
Open a `*_portable.ipynb` notebook in Colab (`Runtime > Change runtime type > GPU`) or in Jupyter.
They import the modules of this repository (cloned automatically in Colab).

## Tests
```bash
python -m pytest -q tests                                     # ~1 min, synthetic data
python tests/synthetic.py --out synth_data                    # synthetic data set
python tests/synthetic_identification.py --data synth_data    # identification vs known coefficients
```

## Reproducibility notes
* Results vary slightly between runs and hardware (nondeterministic GPU operations).
* Raw counts are an auxiliary check only: GLM4, which produced `u_GLM`, was fitted on them.
* Profiles use the training objective. The nominal Wilks interval treats the field records as
  independent, which is optimistic for a spatially correlated field (`--n-eff`).

## License
MIT License (see `LICENSE`).
