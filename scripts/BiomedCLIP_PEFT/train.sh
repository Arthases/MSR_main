#!/bin/bash


# =============================================================================
# BiomedCLIP_PEFT Training Script
# Supports: Linear Probe, LoRA, Adapter, IA3, VPT
# =============================================================================

# Usage:
#   bash scripts/BiomedCLIP_PEFT/train.sh [DATA_ROOT] [DATASET] [METHOD]
#
# Arguments:
#   DATA_ROOT - Root directory of datasets (default: ./data)
#   DATASET   - Dataset name: btmri, busi, chmnist, covid, ctkidney,
#               dermamnist, kneexray, kvasir, lungcolon,
#               retina, or "all" (default: all)
#   METHOD    - Fine-tuning method: lp, lora, adapter, ia3, vpt, or "all" (default: all)
#
# Examples:
#   # Train Linear Probe on all datasets
#   bash scripts/BiomedCLIP_PEFT/train.sh ./data all lp
#
#   # Train LoRA on a single dataset
#   bash scripts/BiomedCLIP_PEFT/train.sh ./data btmri lora
#
#   # Train all methods on all datasets
#   bash scripts/BiomedCLIP_PEFT/train.sh ./data all all

DATA_ROOT=${1:-"./data"}
DATASET=${2:-"all"}
METHOD=${3:-"all"}

# All supported datasets
ALL_DATASETS=("btmri" "busi" "chmnist" "covid" "ctkidney" "dermamnist" "kneexray" "kvasir" "lungcolon" "retina")

# All supported methods and their configs
declare -A METHOD_CONFIGS=(
    ["lp"]="BiomedCLIP"
    ["lora"]="BiomedCLIP_LoRA"
    ["adapter"]="BiomedCLIP_Adapter"
    ["ia3"]="BiomedCLIP_IA3"
    ["vpt"]="BiomedCLIP_VPT"
)

# Resolve dataset list
if [ "$DATASET" == "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$DATASET")
fi

# Resolve method list
if [ "$METHOD" == "all" ]; then
    METHODS=("lp" "lora" "adapter" "ia3" "vpt")
else
    METHODS=("$METHOD")
fi

# =============================================================================
# Helper function to get output directory suffix
# =============================================================================
get_output_suffix() {
    local m=$1
    case $m in
        lp)      echo "" ;;
        lora)    echo "_lora" ;;
        adapter) echo "_adapter" ;;
        ia3)     echo "_ia3" ;;
        vpt)     echo "_vpt" ;;
    esac
}

# =============================================================================
# Helper function to get trainer name
# =============================================================================
get_trainer_name() {
    local m=$1
    echo "${METHOD_CONFIGS[$m]}"
}

# =============================================================================
# Helper function to get config file (per-dataset subfolder structure)
# =============================================================================
get_config_file() {
    local ds=$1
    local m=$2
    local suffix=$(get_output_suffix $m)
    echo "configs/trainers/BiomedCLIP_PEFT/${ds}/${ds}${suffix}.yaml"
}

# =============================================================================
# Main training loop
# =============================================================================
echo "========================================================================"
echo "  BiomedCLIP_PEFT Training Suite"
echo "========================================================================"
echo "  Data Root : $DATA_ROOT"
echo "  Datasets  : ${DATASETS[*]}"
echo "  Methods   : ${METHODS[*]}"
echo "========================================================================"
echo ""

TOTAL_JOBS=0
SKIP_JOBS=0
FAIL_JOBS=0

for ds in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))

        TRAINER=$(get_trainer_name $m)
        CONFIG_FILE=$(get_config_file $ds $m)
        OUTPUT_SUFFIX=$(get_output_suffix $m)
        OUTPUT_DIR="./output/BiomedCLIP_PEFT/${ds}${OUTPUT_SUFFIX}"

        # Check if config file exists
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "[SKIP] Config not found: $CONFIG_FILE"
            SKIP_JOBS=$((SKIP_JOBS + 1))
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
done

echo "========================================================================"
echo "  Training Summary"
echo "========================================================================"
echo "  Total jobs : $TOTAL_JOBS"
echo "  Skipped    : $SKIP_JOBS"
echo "  Failed     : $FAIL_JOBS"
echo "  Succeeded  : $((TOTAL_JOBS - SKIP_JOBS - FAIL_JOBS))"
echo "========================================================================"
