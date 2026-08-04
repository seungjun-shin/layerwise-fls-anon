#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Setup complete."
echo "Quickstart:"
echo "  pytest"
echo "  python scripts/train.py --config configs/smoke_test.yaml"
echo "  python scripts/train.py --config configs/experiment_00_global_fls.yaml training.max_epochs=1"
echo "  python scripts/sweep.py --config configs/smoke_test.yaml"
