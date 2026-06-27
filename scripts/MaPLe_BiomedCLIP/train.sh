#!/bin/bash
# MaPLe_BiomedCLIP Training Script (single dataset)
# Usage: bash scripts/MaPLe_BiomedCLIP/train.sh [DATA_ROOT] [DATASET]


DATA_ROOT=${1:-"./data"}
DATASET=${2:-"btmri"}

python train.py \
    --root ${DATA_ROOT} \
    --output-dir "./output/MaPLe_BiomedCLIP/${DATASET}" \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/MaPLe_BiomedCLIP/${DATASET}/${DATASET}.yaml \
    --trainer MaPLe_BiomedCLIP \
    --seed 42
