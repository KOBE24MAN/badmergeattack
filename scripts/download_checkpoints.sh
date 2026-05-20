#!/usr/bin/env bash
# Download public task-specific CLIP checkpoints from HuggingFace.
# Locally fine-tuned checkpoints (ViT-B/16 + ConvNeXt CIFAR-100, GTSRB)
# are documented in docs/checkpoint_provenance.md.

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoints}"
mkdir -p "$CKPT_DIR"

REPOS=(
  "tanganke/clip-vit-base-patch32_cifar100"
  "tanganke/clip-vit-base-patch32_gtsrb"
  "tanganke/clip-vit-base-patch32_stanford_cars"
  "tanganke/clip-vit-base-patch32_oxford-iiit-pet"
  "tanganke/clip-vit-base-patch16_stanford_cars"
  "tanganke/clip-vit-base-patch16_oxford-iiit-pet"
)

for repo in "${REPOS[@]}"; do
  echo "==> Downloading ${repo}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo}", local_dir="${CKPT_DIR}/${repo##*/}",
                  local_dir_use_symlinks=False)
PY
done

echo
echo "All public checkpoints downloaded to ${CKPT_DIR}/"
echo "See docs/checkpoint_provenance.md for the locally fine-tuned ones."
