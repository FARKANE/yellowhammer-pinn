# Habitat-Modulated Population Dynamics via Physics-Informed Neural Networks

Reproducible code for the paper *A Distributed Parameter Identification Approach
for Habitat-Modulated Population Dynamics via Physics-Informed Neural Networks*
(A. Alassani, A. Farkane, D. Lassounon, R. Ouvrard, T. Poinot).

A habitat-modulated reaction–diffusion model of Yellowhammer
(*Emberiza citrinella*) breeding density across mainland France is identified
from French Breeding Bird Survey data (2002–2024) with physics-informed
neural networks (PINNs).

## Repository contents

| File | Description |
|---|---|
| `pinn_yellowhammer.py` | PINN model, trainer, and configuration |
| `habitat_field.py` | Habitat-index construction from the denoised field |
| `run_experiment1.py` | **Offline** script for field reconstruction (no notebook needed) |
| `Experiment1_Reconstruction_portable.ipynb` | Notebook — runs in Colab **or** locally |
| `Experiment2_Hybrid_RF_portable.ipynb` | Hybrid count-prediction PINN vs. Random Forest |
| `requirements.txt` | Python dependencies |

## Data

Place these files in the repository directory (they are **not** included here):

- `u_GLM_Bruant_jaune.mat`, `Map_France.mat`, `Yellowhammer_2002_2024.csv`
- `Yellowhammer_Clim_Bioclim_CLC_2002_2024.csv` (Experiment 2 only)
- `Data_Estimation_70_V1.csv`, `Data_Validation_30_V1.csv` (Experiment 2 only)

Download them from the Yellowhammer benchmark:
https://github.com/lias-laboratory/yellowhammer-benchmark

## Option 1 — Run locally (offline)

No Google account or Colab required.

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Put the data files in this directory, then run:
python run_experiment1.py
```

Results (metrics, recovered coefficients) are printed to the terminal and the
trained weights are written to `./outputs/`. A GPU is used automatically if
available; otherwise the script runs on CPU (slower).

You can also open the `*_portable.ipynb` notebooks in Jupyter
(`jupyter notebook`) — they detect whether they run in Colab or locally and
behave accordingly.

## Option 2 — Run in Google Colab

1. Open a `*_portable.ipynb` notebook in Colab.
2. `Runtime > Change runtime type > GPU`.
3. Run all cells; upload the data files when prompted.

## Reproducibility note

Results vary slightly between runs and hardware because of nondeterministic GPU
operations. The diffusion coefficient `D0` is weakly identifiable; its point
value is not reported (see the profile-likelihood analysis in the paper).

## License

MIT License (see `LICENSE`).
