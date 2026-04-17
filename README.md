# Dynamic Adaptation Building Energy Forecasting

Open-source code for adaptive building-energy demand forecasting with:

- WGAN-GP forecasting
- PSO-based hyperparameter adaptation
- ARIMA, LSTM, SVR, and GBM baselines
- evaluation, figure, and table-generation utilities

## Repository contents

- `experiments/src/`: forecasting models, PSO adaptation, metrics, evaluation, and experiment runner
- `figures/scripts/`: figure-generation scripts
- `requirements.txt`: Python dependencies
- `pyproject.toml`: package metadata for editable installation
- `reproduce.sh`: convenience script for the full rerun
- `CITATION.cff`: citation metadata for GitHub and Zenodo

## Data

The experiments use the public ASHRAE Great Energy Predictor III dataset:

- https://www.kaggle.com/c/ashrae-energy-prediction/data

Raw data files are not committed to this repository. Place the downloaded data under:

- `experiments/data/raw/`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

## Running the code

Run the end-to-end benchmark:

```bash
python -m experiments.src.run_all --n_buildings 20 --horizons 1,24,168,672
```

Generate tables from saved results:

```bash
python -m experiments.src.make_tables
```

Generate figures:

```bash
python -m experiments.src.make_figures
```

Run the full workflow with one command:

```bash
bash reproduce.sh
```

## Main entry points

- `experiments/src/run_all.py`: run the benchmark pipeline
- `experiments/src/gan_pso.py`: adaptive WGAN-GP + PSO core
- `experiments/src/baselines.py`: ARIMA, LSTM, SVR, and GBM baselines
- `experiments/src/make_tables.py`: table-generation utility
- `experiments/src/make_figures.py`: figure-generation utility

## Notes

- Raw data, manuscript sources, local logs, and generated PDFs are intentionally excluded from this repository.
- Sustainability values in this codebase are computed as upper-bound reserve-procurement scenario estimates under the fixed rule implemented in the evaluation pipeline.
- The repository is intended as a code release, not as the full paper-submission package.
