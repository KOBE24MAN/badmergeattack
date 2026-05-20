# BadMerging Attack — Cross-Architecture Evaluation

This repository contains the implementation and experimental code for the
GENG5512 Engineering Research Project Part 2:
**"BadMerging: A Cross-Architecture Evaluation of Backdoor Attacks
Against Model Merging."**

The project reproduces the BadMerging attack on CLIP ViT-B/32 and evaluates
its robustness across four merging algorithms, five merging coefficients,
two victim tasks, and three vision architectures (ViT-B/32, ViT-B/16,
ConvNeXt-Base-W), with a preliminary ResNet-50 negative case.

---

## 1. Quick Overview

| Stage | What it does | Notebook section |
|---|---|---|
| Stage 1 | Universal trigger optimisation (frozen pre-trained model) | Cell 7A |
| Stage 2 | FI-loss backdoor injection into one task checkpoint | Cell 7C |
| Merge | Combine 1 poisoned + 3 clean checkpoints | Cells 8–11 |
| Evaluate | Clean accuracy + Attack Success Rate (ASR) | Cells 12, 17, 18 |

The four merging algorithms evaluated are:
**Task Arithmetic, TIES-Merging, RegMean, and Simple Averaging.**

---

## 2. Repository Layout

```
badmergeattack/
├── README.md                  ← you are here
├── requirements.txt           ← Python dependencies
├── notebooks/
│   └── badmerging_vitb32_cifar100.ipynb   ← main experiment notebook
├── configs/
│   ├── vitb32.yaml            ← hyperparameters for ViT-B/32
│   ├── vitb16.yaml            ← hyperparameters for ViT-B/16
│   └── convnext.yaml          ← hyperparameters for ConvNeXt-Base-W
├── scripts/
│   ├── download_checkpoints.sh
│   └── run_all.sh
├── src/
│   ├── data/                  ← dataset loaders
│   ├── models/                ← CLIP backbones + classification heads
│   ├── training/              ← Stage 1 + Stage 2 training loops
│   └── evaluation/            ← merging algorithms + metrics
├── results/
│   ├── figures/               ← plots used in the report
│   └── tables/                ← CSV summaries
└── docs/
    └── report.pdf             ← final report (PDF)
```

---

## 3. What the Teacher / Marker Needs to Prepare

### 3.1 Hardware

- **GPU**: NVIDIA T4 or better (16 GB VRAM). Google Colab Pro is sufficient.
- **RAM**: ≥ 16 GB host memory.
- **Disk**: ~10 GB for datasets + checkpoints.

The original experiments were run on Google Colab T4 GPUs.

### 3.2 Software

```bash
# Python ≥ 3.10
pip install -r requirements.txt
```

Key dependencies:
- `torch ≥ 2.1`
- `torchvision`
- `open_clip_torch`
- `huggingface_hub`
- `datasets`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`

### 3.3 Pre-trained CLIP Backbones

The notebook loads CLIP weights via `open_clip_torch`. The first run
downloads them automatically (~600 MB per architecture).

| Architecture | Source |
|---|---|
| ViT-B/32 | `openai` / `open_clip_torch` |
| ViT-B/16 | `openai` / `open_clip_torch` |
| ConvNeXt-Base-W | `laion2b_s13b_b82k_augreg` / `open_clip_torch` |

### 3.4 Task Checkpoints

Each merge requires four task-specific fine-tuned encoders sharing the
same backbone. Where possible, public HuggingFace checkpoints from the
`tanganke/` namespace are used; otherwise the encoders are fine-tuned
locally.

| Task | ViT-B/32 | ViT-B/16 | ConvNeXt |
|---|---|---|---|
| CIFAR-100 | `tanganke/clip-vit-base-patch32_cifar100` | locally fine-tuned | locally fine-tuned |
| GTSRB | `tanganke/clip-vit-base-patch32_gtsrb` | locally fine-tuned | from public set |
| Stanford Cars | `tanganke/clip-vit-base-patch32_stanford_cars` | from public set | from public set |
| Oxford Pets | `tanganke/clip-vit-base-patch32_oxford-iiit-pet` | from public set | from public set |

Run `scripts/download_checkpoints.sh` to fetch all available public
checkpoints. Locally fine-tuned encoders are documented in
`docs/checkpoint_provenance.md`.

### 3.5 Backdoor Checkpoint and Supporting Files

The poisoned ViT-B/32 checkpoint, the optimised universal trigger, the
four classification heads, and the merged-model outputs are hosted at:

**https://huggingface.co/zijun11/backdoor-vitb32-cifar100**

The notebook downloads them automatically via `snapshot_download` on the
first run — no manual setup, no Drive, no token needed. Contents:

| File | Purpose |
|---|---|
| `backdoored_full_model.pth` | full backdoored encoder + classifier |
| `backdoored_vision_model.pth` | encoder weights only |
| `adversary_task_vector.pth` | task vector form used in merging |
| `badmerging_enhanced_model.pth` | FI-loss enhanced checkpoint |
| `merging_output/optimized_trigger.pth` | Stage-1 universal trigger |
| `merging_output/classification_heads/head_*.pth` | per-task heads |
| `merging_output/merged_*.pth` | merged models for each algorithm |
| `merging_output/merging_results.csv` | summary results table |

### 3.5 Datasets

The four image-classification datasets:

| Dataset | Train / Test | Classes | Source |
|---|---|---|---|
| CIFAR-100 | 50,000 / 10,000 | 100 | torchvision |
| GTSRB | 26,640 / 12,630 | 43 | torchvision / HuggingFace |
| Stanford Cars | 8,144 / 8,041 | 196 | torchvision |
| Oxford Pets | 3,680 / 3,669 | 37 | torchvision |

All datasets are downloaded automatically the first time the notebook is run.

---

## 4. How to Run

### Option A — Notebook (recommended for marking)

```bash
git clone https://github.com/KOBE24MAN/badmergeattack.git
cd badmergeattack
pip install -r requirements.txt
jupyter lab notebooks/badmerging_vitb32_cifar100.ipynb
```

Run the cells **in order**. The second cell automatically downloads the
backdoor checkpoint and supporting files from HuggingFace Hub
(`zijun11/backdoor-vitb32-cifar100`), so no manual setup is needed.
Approximate runtimes (Colab T4):

| Cell | Time |
|---|---|
| Setup & checkpoint download | 5–10 min |
| Stage 1 trigger optimisation | ~3 min |
| Stage 2 FI-loss training | ~25 min |
| Four merging algorithms | ~10 min combined |
| Evaluation + t-SNE | ~5 min |
| **Total** | **~1 hour** |

### Option B — Google Colab (zero local setup)

The notebook is Colab-compatible. Upload it to Colab, mount Drive, then
run all cells.

---

## 5. Expected Results

The ViT-B/32 reproduction should produce numbers within ±2% of these:

| Algorithm | Clean Acc | ASR |
|---|---|---|
| Task Arithmetic | 64.18% | 96.43% |
| TIES-Merging | 65.32% | 95.86% |
| RegMean | 70.45% | 89.71% |
| Simple Averaging | 63.97% | 94.52% |
| **Mean** | **65.98%** | **94.13%** |

Figures are saved to `results/figures/` and CSV summaries to
`results/tables/`.

---

## 6. Citation

If you use this code, please cite the original BadMerging paper:

> Zhang et al. "BadMerging: Backdoor Attacks Against Model Merging." 2024.

This repository is the academic reproduction prepared for
**GENG5512 Engineering Research Project Part 2, The University of
Western Australia**, by Zijun Zhou (24261422).

---

## 7. Acknowledgements

- BadMerging authors for the original attack design.
- The `tanganke/` HuggingFace namespace for the public CLIP task
  checkpoints.
- The `open_clip_torch` project for the unified CLIP loader.

---

## 8. License

Released for academic and educational use only.
