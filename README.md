# Multimodal Semantic Reconstruction (MSR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Language-Guided Visual Feature Reconstruction: A Parameter-Efficient Tuning Strategy for Medical Vision-Language Models**

This repository contains the official implementation of **MSR (Multimodal Semantic Reconstruction)**, a lightweight parameter-efficient fine-tuning (PEFT) framework for medical vision-language foundation models (BiomedCLIP). MSR achieves state-of-the-art classification performance across 10 medical imaging datasets with only ~49K trainable parameters.

## Key Features

- **Lightweight adaptation**: ~49K trainable parameters, only 5 more than MaPLe
- **Language-guided visual prompt coupling**: Cross-modal semantic constraints for visual adaptation
- **Multi-level visual feature aggregation (WAGG)**: Adaptive fusion of intermediate and final ViT features
- **Dual semantic reconstruction loss**: MSE alignment + InfoNCE contrastive for cross-modal consistency
- **Frozen backbone**: BiomedCLIP image and text encoders remain frozen during adaptation

## Supported Methods

| Method               | Reference                                      | Category                                    |
| -------------------- | ---------------------------------------------- | ------------------------------------------- |
| **MSR (Ours)** | Language-Guided Visual Feature Reconstruction | Prompt learning + aggregation + contrastive |
| MaPLe                | CVPR 2023                                      | Multi-modal prompt learning                 |
| VPT                  | ECCV 2022                                      | Visual prompt tuning                        |
| LoRA                 | ICLR 2022                                      | Low-rank adaptation                         |
| Adapter              | ICML 2019                                      | Bottleneck residual adapter                 |
| IA3                  | NeurIPS 2022                                   | Channel-wise scaling                        |
| Linear Probe         | CLIP (NeurIPS 2020)                            | Linear classification head                  |
| Zero-shot            | BiomedCLIP (2025)                              | Frozen inference                            |

## Results Overview

| Metric        | LP    | LoRA  | Adapter | IA3   | VPT   | MaPLe | **MSR (Ours)** |
| ------------- | ----- | ----- | ------- | ----- | ----- | ----- | -------------------- |
| Avg. Accuracy | 78.81 | 81.17 | 87.25   | 76.63 | 89.33 | 88.89 | **89.78**      |
| Avg. Macro F1 | 70.68 | 75.22 | 84.28   | 69.56 | 87.02 | 86.60 | **87.94**      |

## Installation

See [assets/INSTALL.md](assets/INSTALL.md) for detailed installation instructions.

Quick start:

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

## Training and Evaluation

The main entry point is `train.py`. Basic usage:

```bash
# Train MSR on BTMRI dataset
python train.py \
  --config-file configs/trainers/MSR_BiomedCLIP/btmri/btmri.yaml \
  --dataset-config-file configs/datasets/btmri.yaml \
  --trainer MSR_BiomedCLIP \
  --root data \
  --output-dir output/msr_btmri
```

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
