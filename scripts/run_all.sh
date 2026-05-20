#!/usr/bin/env bash
# Convenience wrapper for the full ViT-B/32 reproduction pipeline.
# Equivalent to running the notebook end to end.

set -euo pipefail

CONFIG="${CONFIG:-configs/vitb32.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"

echo "==> Using config: $CONFIG"
echo "==> Writing results to: $RESULTS_DIR"

# Step 0: dependency check
python -c "import torch, open_clip; print(torch.__version__, open_clip.__version__)"

# Step 1: download checkpoints
bash scripts/download_checkpoints.sh

# Step 2: run notebook headless
jupyter nbconvert --to notebook --execute \
  notebooks/badmerging_vitb32_cifar100.ipynb \
  --output executed_vitb32.ipynb \
  --output-dir "$RESULTS_DIR"

echo
echo "==> Done. Executed notebook saved to $RESULTS_DIR/executed_vitb32.ipynb"
