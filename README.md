# Dynamic Adaptation Building Energy Forecasting

Open-source code for adaptive building-energy demand forecasting with:

- WGAN-GP forecasting
- PSO-based hyperparameter adaptation
- ARIMA, LSTM, SVR, and GBM baselines
- evaluation, figure, and table-generation utilities

## Repository contents

- `experiments/src/`: forecasting models, PSO adaptation, metrics, evaluation, and experiment runner
- `figures/scripts/`: figure-generation scripts

## Data

The experiments use the public ASHRAE Great Energy Predictor III dataset:

- https://www.kaggle.com/c/ashrae-energy-prediction/data

Raw data files are not committed to this repository. Place the downloaded data under:

- `experiments/data/raw/`

## Main entry points

- `experiments/src/run_all.py`: run the benchmark pipeline
- `experiments/src/gan_pso.py`: adaptive WGAN-GP + PSO core
- `experiments/src/baselines.py`: ARIMA, LSTM, SVR, and GBM baselines
- `experiments/src/make_tables.py`: table-generation utility
- `experiments/src/make_figures.py`: figure-generation utility

## Notes

- Raw data, manuscript sources, local logs, and generated PDFs are intentionally excluded from this repository.
- Sustainability values in this codebase are computed as upper-bound reserve-procurement scenario estimates under the fixed rule implemented in the evaluation pipeline.
