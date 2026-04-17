#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d "experiments/data/raw" ]; then
  echo "Missing experiments/data/raw"
  echo "Download the ASHRAE Great Energy Predictor III data and place it under experiments/data/raw/"
  exit 1
fi

python -m experiments.src.run_all --n_buildings 20 --horizons 1,24,168,672
python -m experiments.src.make_tables
python -m experiments.src.make_figures

