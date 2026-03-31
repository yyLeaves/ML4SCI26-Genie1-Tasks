# ML4SCI GSoC 2026 - GENIE1 Evaluation Tasks

Evaluation tasks for the GSoC 2026 project: **Deep Graph Anomaly Detection with Contrastive Learning for New Physics Searches**.

## Tasks

- [Task 1: Autoencoder for Jet Image Representation](task1_autoencoder/)
- [Task 2: GNN Quark/Gluon Classification](task2_gnn/)
- [Task 3: Contrastive Learning for Classification](task3_contrastive/)

## Data

The dataset consists of 139,306 quark and gluon jet images (3-channel, 125x125) from CMS Open Data. Raw data files are not included due to size. Run `prepare_data.py` to download and `preprocess.py` to generate preprocessed files.

## Setup

```bash
pip install -r requirements.txt
```

## Disclosure

The author used LLM tools (Claude) as a supplementary aid and independently led all aspects of the research.
