# Minutiae-Free Fingerprint Recognition with DINOv2

This project refactors the supplied AFR-Net-style recognition pipeline into a fully minutiae-free DINOv2/ViT pipeline inspired by *Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach*.

The main differences from the AFR-Net code are:

- No CNN + handcrafted/minutiae-guided fusion head.
- No ArcFace identity supervision in the main training loop.
- No enhancement, binarization, thinning, or minutiae extraction.
- Self-supervised teacher-student DINO training on raw fingerprint images.
- L2-normalized CLS embeddings and cosine similarity for verification/identification.

## Files

| File | Purpose |
|---|---|
| `models.py` | DINOv2 backbone wrapper and DINO projection head. |
| `dino_loss.py` | DINO cross-view self-distillation loss with teacher centering. |
| `transforms.py` | Multi-crop fingerprint augmentation pipeline. |
| `data.py` | Split creation and AFR-Net-compatible evaluation datasets. |
| `train_minutiae_free.py` | Self-supervised DINOv2 domain adaptation. |
| `evaluate_minutiae_free.py` | Authentication EER/AUC/TAR@FAR and identification Rank-k. |
| `metrics.py` | Reused biometric metrics from the AFR-Net project. |
| `schedulers.py` | Reused cosine/polynomial schedulers. |
| `config_minutiae_free.yaml` | Default paper-style configuration. |

## Setup

```bash
cd minutiae_free_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run may download DINOv2 weights through `torch.hub`. To avoid internet use on a server, pre-cache the model or place the downloaded checkpoint in the normal PyTorch hub cache.

## 1. Create recognition splits

```bash
python make_splits.py \
  --data-root data/FVC/FVC2000/Db1 \
  --output data/recog_splits.json \
  --min-samples 3
```

For a multi-dataset setup, either run this per dataset and merge JSONs, or place all supported datasets under one root. The split format is compatible with the old AFR-Net evaluation pair generation.

## 2. Train minutiae-free DINOv2

Single GPU:

```bash
python train_minutiae_free.py \
  --config config_minutiae_free.yaml \
  --no-wandb
```

DDP multi-GPU:

```bash
torchrun --nproc_per_node=2 train_minutiae_free.py \
  --config config_minutiae_free.yaml \
  --no-wandb
```

The teacher sees only the two global crops; the student sees all global and local crops. The checkpoint stores both `student` and `teacher`. For evaluation, the script uses the teacher weights when available.

## 3. Evaluate

```bash
python evaluate_minutiae_free.py \
  --config config_minutiae_free.yaml \
  --checkpoint-path ckpts_minutiae_free/best_minutiae_free.pth \
  --split-path data/recog_splits.json \
  --output-dir results_minutiae_free
```

Outputs:

- EER
- AUC
- TAR@FAR = 10%, 1%, 0.1%
- Rank-1, Rank-5, Rank-10 identification accuracy
- `results_minutiae_free/minutiae_free_metrics.json`

## Recommended settings

For reproducing the paper direction, start with:

```yaml
model:
  variant: "base"
  use_registers: true
training:
  batch_size: 64
  epochs: 100
optimizer:
  lr: 1.0e-4
  weight_decay: 0.04
scheduler:
  sched_name: "cosine"
  warmup_epochs: 10
  min_lr: 1.0e-6
loss:
  student_temp: 0.10
  teacher_temp: 0.04
```

For Jetson/edge experiments, train on a workstation first, then export or distill. The Base model is much more deployable than Large/Giant.

## Notes

This is a working project skeleton, not a pretrained result. You still need the fingerprint datasets and enough compute for DINOv2 domain adaptation. The implementation is designed to preserve the useful AFR-Net infrastructure, especially split generation, pair evaluation, and biometric metrics, while replacing the model and training objective with the paper's minutiae-free self-supervised ViT design.
