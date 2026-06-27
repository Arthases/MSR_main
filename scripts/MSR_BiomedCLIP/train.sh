#!/bin/bash


# =============================================================================
# MSR_BiomedCLIP Training Script
# Improved Multimodal Semantic Reconstruction with:
#   - Token-Level Prompt Coupling (MaPLE-style)
#   - Multi-Level Feature Aggregation (MFVT-inspired)
#   - Enhanced SR Loss with Contrastive Alignment
# =============================================================================

# Usage:
#   bash scripts/MSR_BiomedCLIP/train.sh [DATA_ROOT] [DATASET]
#
# Examples:
#   bash scripts/MSR_BiomedCLIP/train.sh ./data btmri
#   bash scripts/MSR_BiomedCLIP/train.sh ./data all

DATA_ROOT=${1:-"./data"}
DATASET=${2:-"btmri"}

TRAINER="MSR_BiomedCLIP"
ALL_DATASETS=("btmri" "busi" "chmnist" "covid" "ctkidney" "dermamnist" "kneexray" "kvasir" "lungcolon" "retina")

if [ "$DATASET" == "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$DATASET")
fi

echo "========================================================================"
echo "  MSR_BiomedCLIP: Improved Multimodal Semantic Reconstruction"
echo "========================================================================"
echo "  Data Root : $DATA_ROOT"
echo "  Datasets  : ${DATASETS[*]}"
echo "  Trainer   : $TRAINER"
echo "========================================================================"
echo ""

TOTAL_JOBS=0
FAIL_JOBS=0

for ds in "${DATASETS[@]}"; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    CONFIG_FILE="configs/trainers/MSR_BiomedCLIP/${ds}/${ds}.yaml"
    OUTPUT_DIR="./output/MSR_BiomedCLIP/${ds}"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[SKIP] Config not found: $CONFIG_FILE"
        continue
    fi

    echo "------------------------------------------------------------------------"
    echo "  Job $TOTAL_JOBS: $TRAINER on $ds"
    echo "  Config  : $CONFIG_FILE"
    echo "  Output  : $OUTPUT_DIR"
    echo "------------------------------------------------------------------------"

    python train.py \
        --root ${DATA_ROOT} \
        --output-dir ${OUTPUT_DIR} \
        --dataset-config-file configs/datasets/${ds}.yaml \
        --config-file ${CONFIG_FILE} \
        --trainer ${TRAINER} \
        --seed 42

    if [ $? -eq 0 ]; then
        echo "[OK] $TRAINER on $ds completed successfully."
    else
        echo "[FAIL] $TRAINER on $ds failed."
        FAIL_JOBS=$((FAIL_JOBS + 1))
    fi
    echo ""
done

echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo "  Total jobs : $TOTAL_JOBS"
echo "  Failed     : $FAIL_JOBS"
echo "  Succeeded  : $((TOTAL_JOBS - FAIL_JOBS))"
echo "========================================================================"
