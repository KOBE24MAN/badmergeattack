# Checkpoint Provenance

Each merge requires four task-specific encoders fine-tuned from the same
CLIP backbone. This document records where every checkpoint used in the
final report came from, so that the marker can reproduce the merges.

## Public checkpoints (HuggingFace, `tanganke/`)

| Architecture | Task | Repo |
|---|---|---|
| ViT-B/32 | CIFAR-100 | `tanganke/clip-vit-base-patch32_cifar100` |
| ViT-B/32 | GTSRB | `tanganke/clip-vit-base-patch32_gtsrb` |
| ViT-B/32 | Stanford Cars | `tanganke/clip-vit-base-patch32_stanford_cars` |
| ViT-B/32 | Oxford Pets | `tanganke/clip-vit-base-patch32_oxford-iiit-pet` |
| ViT-B/16 | Stanford Cars | `tanganke/clip-vit-base-patch16_stanford_cars` |
| ViT-B/16 | Oxford Pets | `tanganke/clip-vit-base-patch16_oxford-iiit-pet` |

All public checkpoints are downloaded by `scripts/download_checkpoints.sh`.

## Locally fine-tuned checkpoints

Where a matching public checkpoint was not available, the encoder was
fine-tuned locally in Google Colab. The recipe is identical across runs.

| Architecture | Task | Why local | Train config |
|---|---|---|---|
| ViT-B/16 | CIFAR-100 | not on `tanganke/` | linear head, AdamW, lr 1e-5, batch 64, 3 epochs |
| ViT-B/16 | GTSRB | not on `tanganke/` | linear head, AdamW, lr 1e-5, batch 64, 3 epochs |
| ConvNeXt-Base-W | CIFAR-100 | not on `tanganke/` | linear head, AdamW, lr 1e-5, batch 32, 5 epochs |
| ConvNeXt-Base-W | GTSRB | used for victim swap | linear head, AdamW, lr 1e-5, batch 32, 5 epochs |

All locally trained checkpoints align their state-dict keys with the
public ones before merging.

## Note on SUN397

The original BadMerging paper uses SUN397 as one task. A consistent
SUN397 checkpoint was not available for ViT-B/16 or ConvNeXt-Base-W in
the `tanganke/` namespace, so Oxford Pets is substituted across all
three architectures for a fair cross-architecture comparison. This is
documented as a limitation in Chapter 5 of the report.
