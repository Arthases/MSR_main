#!/bin/bash


# Training Script for BiomedCLIP_PEFT_Adapter
# Using Dassl Framework

# Usage:
#   bash scripts/BiomedCLIP_PEFT/train_adapter.sh [DATA_ROOT] [DATASET_NAME]
#
# Example:
#   bash scripts/BiomedCLIP_PEFT/train_adapter.sh ./data btmri

DATA_ROOT=${1:-"./data"}
DATASET=${2:-"btmri"}
OUTPUT_DIR="./output/BiomedCLIP_PEFT/${DATASET}_adapter"

echo "=============================================="
echo "  BiomedCLIP_PEFT_Adapter Training on ${DATASET}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="

python train.py \
    --root ${DATA_ROOT} \
    --output-dir ${OUTPUT_DIR} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/BiomedCLIP_PEFT/${DATASET}/${DATASET}_adapter.yaml \
    --trainer BiomedCLIP_Adapter \
    --seed 42
