# Multimodal Semantic Reconstruction (MSR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Language-Guided Visual Feature Reconstruction: A Parameter-Efficient Tuning Strategy for Medical Vision-Language Models**

This repository contains the official implementation of **MSR (Multimodal Semantic Reconstruction)**, a lightweight adaptation framework for BiomedCLIP with frozen image and text encoders. MSR combines language-guided visual prompt coupling, weighted multi-level visual feature aggregation, and semantic reconstruction losses for medical image classification.

## Key Features

- **Lightweight adaptation**: 49,177 trainable parameters, with a budget comparable to the implemented MaPLe baseline (49,172 parameters)
- **Language-guided visual prompt coupling**: Cross-modal semantic constraints for visual adaptation
- **Multi-level visual feature aggregation (WAGG)**: Adaptive fusion of intermediate and final ViT features
- **Dual semantic reconstruction loss**: MSE alignment + InfoNCE contrastive for cross-modal consistency
- **Frozen backbone**: BiomedCLIP image and text encoders remain frozen during adaptation

## Supported Methods

| Method               | Reference                                      | Category                                    |
| -------------------- | ---------------------------------------------- | ------------------------------------------- |
| **MSR (Ours)** | Language-Guided Visual Feature Reconstruction | Prompt learning + aggregation + semantic reconstruction |
| MaPLe                | CVPR 2023                                      | Multi-modal prompt learning                 |
| VPT                  | ECCV 2022                                      | Visual prompt tuning                        |
| LoRA                 | ICLR 2022                                      | Low-rank adaptation                         |
| Adapter              | ICML 2019                                      | Bottleneck residual adapter                 |
| IA3                  | NeurIPS 2022                                   | Channel-wise scaling                        |
| Linear Probe         | CLIP (ICML 2021)                               | Linear classification head                  |
| Zero-shot            | BiomedCLIP (2025)                              | Frozen inference                            |

## Results Overview

The following full-data results match Table 1 of the revised manuscript. Each value first averages seeds **42, 2026, and 3047** within each dataset, then weights the **ten datasets equally**. Accuracy and Macro F1 are percentages; they are not pooled-image scores.

| Metric        | LP    | LoRA  | Adapter | IA3   | VPT   | MaPLe | **MSR (Ours)** |
| ------------- | ----- | ----- | ------- | ----- | ----- | ----- | -------------------- |
| Avg. Accuracy | 78.90 | 81.35 | 87.14   | 76.32 | 89.13 | 89.21 | **89.42**      |
| Avg. Macro F1 | 70.88 | 75.23 | 84.19   | 69.15 | 87.11 | 87.02 | **87.53**      |

MSR had the highest numerical means among the evaluated adaptation methods in this full-data comparison. Performance varied by dataset, and the statistical analysis did not establish superiority over MaPLe. Per-dataset and per-seed results, standard deviations, and statistical comparisons are reported in the manuscript and its Online Resource.

In the few-shot evaluation on BTMRI, BUSI, DermaMNIST, and KneeXray (4, 8, 16, and 32 training images per class), MSR did not consistently outperform the compared methods, particularly VPT. MSR led in mean Accuracy at 16 shots; VPT led in mean Accuracy at the other three shot levels and in mean Macro F1 at all four. Each run additionally used four validation images per class for checkpoint selection, giving a development-label budget of K + 4 images per class.

The component ablation in Table 3 and the 12-layer VPT comparison use only seed 42 and are interpreted descriptively. The revision-stage WAGG validation analysis uses three seeds and is reported separately from the main benchmark.

## Installation

Quick-start installation commands:

```bash
# Create conda environment
conda create -n msr python=3.10 -y
conda activate msr

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install Dassl.pytorch
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

## Model Preparation

Download the BiomedCLIP model checkpoint and place it in the following directory:

```
clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/
```

The model will be automatically downloaded from HuggingFace on first run if not present, or you can manually download it and put it in the above path.

## Dataset Preparation

See [assets/DATASETS.md](assets/DATASETS.md) for dataset download links and preparation instructions.

The repository includes the fixed train/validation/test split JSON files under `data/`. Obtain the images from the linked dataset sources and retain these split files when preparing the datasets.

## Training and Evaluation

The main entry point is `train.py`. Basic usage:

```bash
# Train MSR on BTMRI dataset
python train.py \
  --config-file configs/trainers/MSR_BiomedCLIP/btmri/btmri.yaml \
  --dataset-config-file configs/datasets/btmri.yaml \
  --trainer MSR_BiomedCLIP \
  --root data \
  --seed 42 \
  --output-dir output/msr_btmri_seed42
```

This command runs one seed. For the manuscript's full-data protocol, run each method and dataset with seeds 42, 2026, and 3047, using a separate output directory for each run. The few-shot experiments additionally require the sampling and batch-size settings described in the Methods and Online Resource Section S1.3.

### Run with other methods

```bash
# Change --trainer and --config-file accordingly
python train.py \
  --config-file configs/trainers/MSR_BiomedCLIP/btmri/btmri.yaml \
  --trainer MSR_BiomedCLIP \
  ...  # same dataset config, root, output-dir

python train.py \
  --config-file configs/trainers/MaPLe_BiomedCLIP/btmri/btmri.yaml \
  --trainer MaPLe_BiomedCLIP \
  ...

python train.py \
  --config-file configs/trainers/BiomedCLIP_PEFT/btmri/btmri.yaml \
  --trainer BiomedCLIP \
  ...
```

### Supported datasets

`btmri`, `busi`, `chmnist`, `covid`, `ctkidney`, `dermamnist`, `kneexray`, `kvasir`, `lungcolon`, `retina`

## Project Structure

```
MSR_main/
├── train.py                 # Main entry point
├── configs/
│   ├── datasets/            # Dataset configs
│   └── trainers/            # Method + dataset specific configs
├── datasets/                # Dataset implementations
├── trainers/
│   └── BiomedCLIP_PEFT/       # All PEFT method implementations
├── clip/                    # BiomedCLIP model and tokenizer
├── open_clip/               # OpenCLIP library (vendored)
├── Dassl.pytorch/           # Dassl framework (vendored)
├── data/                    # Dataset split files
└── assets/                  # Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
