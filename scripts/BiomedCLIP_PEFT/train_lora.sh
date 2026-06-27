#!/bin/bash


# Training Script for BiomedCLIP_PEFT_LoRA
# Using Dassl Framework

# Usage:
#   bash scripts/BiomedCLIP_PEFT/train_lora.sh [DATA_ROOT] [DATASET_NAME]
#
# Example:
#   bash scripts/BiomedCLIP_PEFT/train_lora.sh ./data btmri

DATA_ROOT=${1:-"./data"}
DATASET=${2:-"btmri"}
OUTPUT_DIR="./output/BiomedCLIP_PEFT/${DATASET}_lora"

echo "=============================================="
echo "  BiomedCLIP_PEFT_LoRA Training on ${DATASET}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="

python train.py \
    --root ${DATA_ROOT} \
    --output-dir ${OUTPUT_DIR} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/BiomedCLIP_PEFT/${DATASET}/${DATASET}_lora.yaml \
    --trainer BiomedCLIP_LoRA \
    --seed 42
