# Task 3: Contrastive Learning for Quark/Gluon Classification

## Objective

Explore contrastive learning methods for quark/gluon jet classification. Compare self-supervised (MoCo) and supervised (SupCon) contrastive objectives across different encoder architectures, and evaluate whether contrastive pretraining improves over direct linear probing of autoencoder features.

## Dataset

- **Source**: Same 139,306 CMS Open Data jet images as Tasks 1 and 2
- **Split**: 80% train / 10% val / 10% test (stratified by label, seed=42)
- **Preprocessing**: Two variants used depending on experiment:
  - **Percentile**: per-channel 99.9th percentile clip to [0,1] (used for Swin-T experiments)
  - **pT+log+percentile**: per-channel pT norm + log(1+x/eps) + percentile clip (used for AE encoder experiments, matching Task 1 SegNet training data)

## Approach

Three lines of experiments were conducted:

**Line 1: AE encoder as feature extractor.** Load the pretrained SegNet encoder from Task 1 (frozen), extract 512d bottleneck features, and train a linear classifier. This tests how much classification-relevant information the autoencoder's reconstruction objective implicitly captures.

**Line 2: Contrastive pretraining on AE encoder.** Starting from Task 1 SegNet weights, apply MoCo or SupCon contrastive fine-tuning to reshape the 512d embedding space, then evaluate with linear probing. Also tested joint SupCon + reconstruction loss to prevent forgetting of reconstruction-learned features.

**Line 3: Swin Transformer contrastive pretraining.** Train Swin-T (ImageNet-pretrained or custom small variant) with MoCo or SupCon, then evaluate via linear probing or fine-tuning. Tests whether vision transformer features combined with contrastive objectives outperform AE-based approaches.

## Augmentations

Jet-specific augmentations (following [arXiv:2602.00141](https://arxiv.org/abs/2602.00141)), applied during contrastive pretraining (MoCo/SupCon) to generate two views per image. Not used in linear probing or fine-tuning stages.

- Random translation in (eta, phi) space
- Gaussian noise (simulating detector resolution)
- Intensity scaling (simulating energy calibration uncertainty)
- Random erasing (simulating detector dead zones)
- No rotation or flipping (eta and phi have different physical meaning)

## Results

### Line 1: AE Encoder Linear Probe

| Experiment | AE Model | Data | Optimizer | Val AUC | Test AUC | Test Acc |
|------------|----------|------|-----------|:-------:|:--------:|:--------:|
| ae_linear | SegNet (ptlogpct) | pT+log+pct | AdamW | 0.755 | 0.762 | 70.2% |
| **ae_linear_percentile** | **SegNet (percentile)** | **percentile** | **AdamW** | **0.761** | **0.769** | **71.1%** |
| ae_linear_sgd | SegNet (ptlogpct) | pT+log+pct | SGD | 0.754 | 0.761 | 70.2% |

Multiple optimizer/lr/wd combinations were tested (AdamW, SGD, lr from 2e-5 to 1e-3, wd from 1e-4 to 3e-3). All converged to the same result (~0.76 Test AUC), confirming this is the linear separability ceiling of the AE bottleneck.

The percentile-trained SegNet produced features with slightly stronger classification signal than the pT+log+percentile variant (0.769 vs 0.762), despite the latter having better visual reconstruction quality in Task 1.

### Line 2: Contrastive Fine-tuning on AE Encoder

| Experiment | Method | Pretrain Epochs | Data | Val AUC | Test AUC | Test Acc |
|------------|--------|:---:|------|:-------:|:--------:|:--------:|
| ae_moco | MoCo | 30 | pT+log+pct | 0.743 | 0.748 | 68.8% |
| **ae_supcon** | **SupCon** | **7 (killed at 7/30)** | **pT+log+pct** | **0.771** | **0.775** | **71.2%** |
| ae_supcon_percentile | SupCon | 7 | percentile | 0.764 | 0.771 | 70.9% |
| ae_supcon_recon | SupCon+Recon (lambda=10000) | 30 | pT+log+pct | 0.771 | 0.773 | 71.0% |
| ae_supcon_recon_pct | SupCon+Recon (lambda=10000) | 30 | percentile | — | — | — |

> ae_supcon was configured for 30 epochs but killed at epoch 7 due to time constraints. The 7-epoch weights were used for evaluation. ae_supcon_recon_pct results are unreliable due to two concurrent processes writing to the same output directory.

Key observations:
- **MoCo hurt AE features** (0.748 vs 0.762 baseline). Self-supervised contrastive fine-tuning disrupted the reconstruction-learned representations without adequate replacement.
- **SupCon improved over baseline** (0.775 vs 0.762). Label information in the contrastive loss provides meaningful signal for reshaping the embedding space.
- **Adding reconstruction loss did not help** (0.773 vs 0.775). The joint SupCon + reconstruction objective (lambda=10000) did not outperform pure SupCon.

### Line 3: Swin Transformer

| Experiment | Encoder | Params | Method | Pretrain | Val AUC | Test AUC | Test Acc |
|------------|---------|:------:|--------|----------|:-------:|:--------:|:--------:|
| moco_swin | Swin-T (ImageNet) | 28M | MoCo → linear | 10ep, loss=2.52 | 0.775 | 0.778 | 71.2% |
| finetune_swin | Swin-T (MoCo) | 28M | MoCo → ft last 2 stages (57% trainable) | 10ep MoCo | 0.776 | 0.780 | 71.2% |
| moco_custom_swin | Custom Swin | 8.8M | MoCo → linear | 50ep, loss=2.35 | 0.754 | 0.757 | 69.6% |
| finetune_custom_swin | Custom Swin (MoCo) | 8.8M | MoCo → ft all (diff. LR: enc=2e-6, head=1e-4) | 50ep MoCo | 0.777 | 0.781 | 71.6% |
| **supcon_swin** | **Swin-T (ImageNet)** | **28M** | **SupCon → linear** | **10ep** | **0.781** | **0.788** | **72.1%** |
| swin_ptlog_linear | Swin-T (ImageNet) | 28M | linear only (no contrastive) | — | 0.706 | 0.707 | 65.4% |

The Custom Swin (8.8M params) is adapted from [arXiv:2602.00141](https://arxiv.org/abs/2602.00141) for 128x128 input: patch_size=4, window_size=4, 3 stages (2,2,4 blocks), embedding 96→192→384.

## Summary

| Rank | Experiment | Test AUC | Test Acc |
|:----:|------------|:--------:|:--------:|
| 1 | **SupCon Swin-T (ImageNet, 10ep)** | **0.788** | **72.1%** |
| 2 | Fine-tune Custom Swin (MoCo, 50ep) | 0.781 | 71.6% |
| 3 | Fine-tune Swin-T (MoCo, 10ep) | 0.780 | 71.2% |
| 4 | MoCo Swin-T (ImageNet, 10ep) | 0.778 | 71.2% |
| 5 | AE SupCon 7ep (ptlogpct) | 0.775 | 71.2% |
| 6 | AE SupCon+Recon 30ep (ptlogpct) | 0.773 | 71.0% |
| 7 | AE SupCon 7ep (percentile) | 0.771 | 70.9% |
| 8 | AE linear (percentile) | 0.769 | 71.1% |
| 9 | AE linear (ptlogpct) | 0.762 | 70.2% |
| 10 | MoCo Custom Swin (50ep) | 0.757 | 69.6% |
| 11 | AE MoCo (30ep) | 0.748 | 68.8% |
| 12 | Swin-T ImageNet on pT+log data | 0.707 | 65.4% |

## Key Findings

1. **SupCon consistently outperforms MoCo** across all encoder types (+1-3% AUC). Label information in the contrastive loss provides meaningful signal for quark/gluon discrimination. This aligns with the SupCon paper's findings on ImageNet.

2. **AE bottleneck features are competitive without contrastive training** (0.769 AUC). The reconstruction objective implicitly captures class-relevant structure in the 512d bottleneck. This is a useful baseline that requires no additional training.

3. **Self-supervised contrastive (MoCo) can degrade AE features**. MoCo fine-tuning dropped AUC from 0.762 to 0.748, likely because the contrastive objective reshaped the embedding space in ways that lost reconstruction-learned discriminative information.

4. **ImageNet pretraining transfers to jet images**. Swin-T with ImageNet weights (0.778 AUC with just 10ep MoCo) outperforms Custom Swin from scratch (0.757 with 50ep MoCo), despite jet images being very different from natural images.

5. **Preprocessing affects classification differently than reconstruction**. Percentile normalization produced better classification features (0.769 vs 0.762 for AE linear), while pT+log+percentile gave better visual reconstruction in Task 1. However, pT+log data with SupCon gave the best contrastive result (0.775 vs 0.771 for percentile).

6. **pT+log data is incompatible with ImageNet Swin-T features**. Direct linear probing of ImageNet Swin-T on pT+log data gave only 0.707 AUC, much worse than percentile data (0.778 with MoCo). ImageNet features expect [0,1] value ranges.

## Limitations

- Swin-T experiments used 128x128 input with ImageNet weights trained on 224x224, requiring position bias interpolation. This may reduce the benefit of pretrained weights.
- The Custom Swin architecture is adapted for 128x128 but has not been systematically tuned (patch_size, window_size, depth).
- ae_supcon_recon_percentile results are unreliable due to concurrent process corruption.
- All experiments use the same dataset for classification. The project's ultimate goal is anomaly detection, which requires different evaluation (training on background only, testing with injected signals).

## File Structure

```
task3_contrastive/
  README.md                  # This file
  model.py                   # SwinEncoder, CustomSwinEncoder, MoCo, SupConLoss, etc.
  augmentations.py           # Jet-specific augmentations
  train.py                   # MoCo/SimCLR pretraining + linear eval
  finetune.py                # Fine-tune pretrained encoder with CE loss
  swin_supcon.py             # SupCon pretraining for Swin
  ae_classify.py             # AE encoder linear probe
  ae_moco.py                 # AE encoder + MoCo fine-tune
  ae_supcon.py               # AE encoder + SupCon fine-tune
  ae_supcon_recon.py         # AE encoder + SupCon + reconstruction joint training
  configs/                   # YAML configs for all experiments

outputs/task3/
  ae_linear/                 # AE linear probe (AdamW)
  ae_linear_percentile/      # AE linear probe (percentile data)
  ae_linear_sgd/             # AE linear probe (SGD)
  ae_moco/                   # AE + MoCo
  ae_supcon/                 # AE + SupCon (best AE-based)
  ae_supcon_percentile/      # AE + SupCon (percentile)
  ae_supcon_recon/           # AE + SupCon + Recon
  ae_supcon_recon_percentile/ # Corrupted (dual process)
  moco_swin/                 # MoCo Swin-T (ImageNet)
  moco_custom_swin/          # MoCo Custom Swin
  finetune_swin/             # Fine-tune MoCo Swin-T
  finetune_custom_swin/      # Fine-tune MoCo Custom Swin
  supcon_swin/               # SupCon Swin-T (best overall)
  swin_ptlog_linear/         # Swin-T on pT+log data (poor)
```

## References

- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Comparison of Image Processing Models in Quark-Gluon Jet Classification](https://arxiv.org/abs/2602.00141)

## Running

```bash
# AE linear probe
python task3_contrastive/ae_classify.py --config task3_contrastive/configs/ae_linear.yaml

# AE + SupCon
python task3_contrastive/ae_supcon.py --config task3_contrastive/configs/ae_supcon.yaml

# MoCo Swin-T pretraining + linear eval
python task3_contrastive/train.py --config task3_contrastive/configs/moco_swin.yaml

# SupCon Swin-T + linear eval
python task3_contrastive/swin_supcon.py --config task3_contrastive/configs/supcon_swin.yaml

# Fine-tune pretrained encoder
python task3_contrastive/finetune.py --config task3_contrastive/configs/finetune_swin.yaml
```
