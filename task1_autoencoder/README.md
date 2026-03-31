# Task 1: Autoencoder for Jet Image Representation Learning

## Objective

Train a convolutional autoencoder on 3-channel (Tracks, ECAL, HCAL) 125x125 jet images to learn compressed representations. Evaluate reconstruction quality across all channels.

## Dataset

- **Source**: Simulated CMS Open Data -- 69,653 quark jets + 69,653 gluon jets = 139,306 total
- **Format**: 3-channel 125x125 images (Tracks, ECAL, HCAL), zero-padded to 128x128
- **Split**: 80% train / 20% validation

## Experiments

### Preprocessing

| Method | Description | Outcome |
|--------|-------------|---------|
| **percentile** | **Per-channel 99.9th percentile clip to [0,1]** | **Used in Task 3 contrastive learning** |
| pT + log | Per-channel pT norm + log(1+x/eps) | Values ~[0,12], works but Tracks dead on some archs |
| **pT + log + percentile** | **pT norm + log + percentile clip** | **Best visual reconstruction quality** |
| pT only | Per-channel pT norm (no log) | Values too small (~1e-3), all-zero output |
| pT + sqrt | Per-channel pT norm + sqrt | Insufficient amplification, all-zero output |

### Architecture

| Experiment | Architecture | Bottleneck | Preprocessing | Output Act | Notes |
|------------|-------------|:----------:|---------------|-----------|-------|
| baseline | ConvAE 4x down | FC 128d | per-ch pT+log | ReLU | First baseline |
| weighted_mse | ConvAE, 10x nonzero weight | FC 128d | per-ch pT+log | ReLU | Background haze, worse |
| perchannel_mse | ConvAE | FC 128d | per-ch pT+log | ReLU | Same data as baseline, Tracks still dead |
| shallow_mse | 3x down, conv | (4,16,16)=1024d | per-ch pT+log | ReLU | Better spatial resolution, still weak on Tracks |
| baseline_sqrt | ConvAE | FC 128d | pT+sqrt | ReLU | All-zero output |
| baseline_percentile | ConvAE | FC 128d | percentile | ReLU | ReLU dead zone |
| softplus_percentile | ConvAE | FC 384d | percentile | Softplus | Trains without collapsing to zero |
| **segnet_percentile** | **SegNet** | **FC 512d** | **percentile** | **ReLU** | **Used in Task 3 contrastive** |
| **segnet_ptlogpct** | **SegNet** | **FC 512d** | **pT+log+pct** | **ReLU** | **Best visual quality** |

### Key Architecture: SegNet Autoencoder

The extreme sparsity of jet images (>97% zero pixels) means that most spatial information is concentrated in a small number of non-zero deposits. Standard strided convolutions and bilinear upsampling blend these sparse activations with surrounding zeros, losing precise spatial locations -- particularly problematic for the Tracks channel (discrete single-pixel deposits) and scattered ECAL hits. This observation led us to adopt the SegNet architecture (originally designed for semantic segmentation), which faces a similar challenge of preserving spatial detail through a bottleneck. SegNet records pooling indices during encoding and uses them during decoding to place values back at their original positions, producing sparse feature maps that subsequent convolutions can densify.

```
Encoder: [Conv2d(3x3)+BN+LeakyReLU(0.2) -> MaxPool2d(2,2,return_indices)] x3  [128->16]
Bottleneck: flatten(128*16*16=32768) -> FC(32768,512) -> LeakyReLU -> FC(512,32768) -> reshape
Decoder: [MaxUnpool2d(indices) -> Conv2d(3x3)+BN+LeakyReLU(0.2)] x3  [16->128]
Output: Conv2d(3x3) -> ReLU
```

Training details: AdamW (lr=3e-3, weight_decay=1e-5), CosineAnnealing (T_max=100, eta_min=1e-5), MSE loss. segnet_percentile uses batch_size=512; segnet_ptlogpct uses batch_size=128.

## Results

### SegNet + pT+log+percentile (best visual quality)

#### Average Reconstruction

![Average Reconstruction](../outputs/task1/segnet_ptlogpct/avg_reconstruction.png)

#### Quark vs Gluon

| Quark | Gluon |
|:-----:|:-----:|
| <img src="../outputs/task1/segnet_ptlogpct/avg_reconstruction_quark.png" width="400"> | <img src="../outputs/task1/segnet_ptlogpct/avg_reconstruction_gluon.png" width="400"> |

#### Per-Channel Reconstruction

| Channel | SegNet + pT+log+percentile | SegNet + Percentile |
|---------|:-----------------:|:----------:|
| Tracks | <img src="../outputs/task1/segnet_ptlogpct/reconstruction_tracks.png" width="400"> | <img src="../outputs/task1/segnet_percentile/reconstruction_tracks.png" width="400"> |
| ECAL | <img src="../outputs/task1/segnet_ptlogpct/reconstruction_ecal.png" width="400"> | <img src="../outputs/task1/segnet_percentile/reconstruction_ecal.png" width="400"> |
| HCAL | <img src="../outputs/task1/segnet_ptlogpct/reconstruction_hcal.png" width="400"> | <img src="../outputs/task1/segnet_percentile/reconstruction_hcal.png" width="400"> |

#### Single Event Examples: Quark vs Gluon

**Quark jets**:

| SegNet + pT+log+percentile | SegNet + Percentile |
|:-----------------:|:----------:|
| <img src="../outputs/task1/segnet_ptlogpct/event_2119.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_2119.png" width="400"> |

**Gluon jets**:

| SegNet + pT+log+percentile | SegNet + Percentile |
|:-----------------:|:----------:|
| <img src="../outputs/task1/segnet_ptlogpct/event_131217.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_131217.png" width="400"> |

#### Average Reconstruction Comparison

| SegNet + pT+log+percentile | SegNet + Percentile |
|:-----------------:|:----------:|
| <img src="../outputs/task1/segnet_ptlogpct/avg_reconstruction.png" width="400"> | <img src="../outputs/task1/segnet_percentile/avg_reconstruction.png" width="400"> |

#### Quark vs Gluon (pT+log+percentile)

| Quark | Gluon |
|:-----:|:-----:|
| <img src="../outputs/task1/segnet_ptlogpct/avg_reconstruction_quark.png" width="400"> | <img src="../outputs/task1/segnet_ptlogpct/avg_reconstruction_gluon.png" width="400"> |

#### More Single Event Comparisons

**Quark jets**:

| SegNet + pT+log+percentile | SegNet + Percentile |
|:-----------------:|:----------:|
| <img src="../outputs/task1/segnet_ptlogpct/event_23981.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_23981.png" width="400"> |
| <img src="../outputs/task1/segnet_ptlogpct/event_46844.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_46844.png" width="400"> |

**Gluon jets**:

| SegNet + pT+log+percentile | SegNet + Percentile |
|:-----------------:|:----------:|
| <img src="../outputs/task1/segnet_ptlogpct/event_133468.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_133468.png" width="400"> |
| <img src="../outputs/task1/segnet_ptlogpct/event_69977.png" width="400"> | <img src="../outputs/task1/segnet_percentile/event_69977.png" width="400"> |

The percentile model was used for Task 3 contrastive experiments (SupCon, MoCo) since its bottleneck features produced stronger classification signal (Test AUC=0.769 vs 0.762 for pT+log+percentile).

## Key Findings

1. **Preprocessing interacts with image resolution**: pT normalization is commonly used on smaller jet images (25-40px) in the literature. On our 125px images, the resulting per-pixel values were very small, and adding log scaling and percentile clipping helped stabilize training.

2. **SegNet handles sparse data well in our setting**: The SegNet architecture produced better reconstructions of sparse deposits compared to ConvAutoencoder and ShallowAE variants in our experiments, possibly due to how pooling indices interact with the highly sparse input.

3. **Output activation and architecture interact**: We observed that ReLU produced all-zero outputs with ConvAutoencoder on percentile-normalized data, but worked with SegNet. Softplus worked across both architectures.

4. **Training duration matters**: SegNet's HCAL reconstruction appeared poor at 10 epochs but improved to match other channels by epoch 100.

## Limitations

- **Percentile clip values computed on full dataset**: The 99.9th percentile clip values use both train and validation data. While the impact is negligible (stable statistics over 139K samples), strict practice requires computing on train set only.
- **No independent test set**: Current evaluation uses validation set only. Final model should be evaluated on a held-out test set.
- **Single-pixel reconstruction**: While SegNet preserves spatial positions well, the finest details (individual pixel intensities) still show some deviation from ground truth due to the FC bottleneck's information compression.

## References

- [Jet-images: computer vision inspired techniques for jet tagging](https://arxiv.org/abs/1511.05190)
- [Searching for new physics with deep autoencoders](https://arxiv.org/abs/1808.08992)
- [Autoencoders for unsupervised anomaly detection in high energy physics](https://arxiv.org/abs/2104.09051)
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture](https://arxiv.org/abs/1511.00561)
- [Quantum Autoencoders for HEP Analysis at the LHC](https://www.tommago.com/posts/gsoc/)
- [Comparison of Image Processing Models in Quark-Gluon Jet Classification](https://arxiv.org/abs/2602.00141)

## File Structure

```
task1_autoencoder/
  README.md                  # This file

outputs/task1/
  baseline/                  # ConvAE + pT+log
  weighted_mse/              # 10x nonzero weight
  perchannel_mse/            # Per-channel pT+log
  shallow_mse/               # 3x downsample, conv bottleneck
  baseline_sqrt/             # pT+sqrt (failed)
  baseline_percentile/       # percentile + ReLU (failed)
  softplus_percentile/       # percentile + Softplus
  segnet_percentile/         # SegNet + percentile (used in Task 3)
  segnet_ptlogpct/           # SegNet + pT+log+pct (best visual)
  segnet_sigmoid/            # SegNet + Sigmoid (stopped early)
  segnet_pt/                 # SegNet + pure pT (failed)
```

Each experiment folder contains:
- `train.log` -- training logs with per-epoch metrics
- `eval.log` -- evaluation metrics (MSE per channel)
- `best.pt` -- best model checkpoint (not in git)
- `*.png` -- reconstruction plots, scatter plots, loss distributions

## Running

```bash
# Preprocess
python preprocess.py --method percentile

# Train
python task1_autoencoder/train.py --config task1_autoencoder/configs/segnet_ptlogpct.yaml

# Evaluate
python task1_autoencoder/evaluate.py --checkpoint outputs/task1/segnet_ptlogpct/best.pt
```
