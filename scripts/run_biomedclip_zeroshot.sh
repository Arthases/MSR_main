#!/bin/bash

# BiomedCLIP zero-shot evaluation for the BiomedCLIP_PEFT trainer family.
#
# Usage:
#   bash scripts/run_biomedclip_zeroshot.sh [DATA_ROOT] [DATASET] [PROMPT_MODE]
#
# Arguments:
#   DATA_ROOT   Dataset root directory, default: ./data
#   DATASET     One dataset name or "all", default: all
#   PROMPT_MODE custom or ensemble, default: custom
#
# Examples:
#   bash scripts/run_biomedclip_zeroshot.sh ./data btmri custom
#   bash scripts/run_biomedclip_zeroshot.sh ./data all ensemble

set -u




DATA_ROOT=${1:-"./data"}
DATASET=${2:-"all"}
PROMPT_MODE=${3:-"custom"}
TRAINER="BiomedCLIP_ZeroShot"

# 缁撴灉鏀堕泦
LOG_DIR="logs/BiomedCLIP_PEFT_ZeroShot/${PROMPT_MODE}"
RESULT_FILE="BiomedCLIP_PEFT_ZeroShot_${PROMPT_MODE}_缁撴灉.txt"
mkdir -p "${LOG_DIR}"

ALL_DATASETS=("btmri" "busi" "chmnist" "covid" "ctkidney" "dermamnist" "kneexray" "kvasir" "lungcolon" "retina")

if [ "$PROMPT_MODE" != "custom" ] && [ "$PROMPT_MODE" != "ensemble" ]; then
    echo "[ERROR] PROMPT_MODE must be 'custom' or 'ensemble', got: $PROMPT_MODE"
    exit 1
fi

if [ "$DATASET" == "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$DATASET")
fi

echo "========================================================================"
echo "  BiomedCLIP Zero-shot Evaluation"
echo "========================================================================"
echo "  Data Root   : $DATA_ROOT"
echo "  Datasets    : ${DATASETS[*]}"
echo "  Trainer     : $TRAINER"
echo "  Prompt mode : $PROMPT_MODE"
echo "========================================================================"
echo ""

TOTAL_JOBS=0
SKIP_JOBS=0
FAIL_JOBS=0

# 瀛樺偍鎵€鏈夌粨鏋? RESULTS[鏁版嵁闆哴="acc f1"
declare -A RESULTS

for ds in "${DATASETS[@]}"; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    DATASET_CONFIG="configs/datasets/${ds}.yaml"
    TRAINER_CONFIG="configs/trainers/BiomedCLIP_PEFT/zeroshot.yaml"
    OUTPUT_DIR="./output/BiomedCLIP_PEFT_ZeroShot/${PROMPT_MODE}/${ds}"
    LOG_FILE="${LOG_DIR}/${ds}.log"

    if [ ! -f "$DATASET_CONFIG" ]; then
        echo "[SKIP] Dataset config not found: $DATASET_CONFIG"
        SKIP_JOBS=$((SKIP_JOBS + 1))
        RESULTS["${ds}"]="N/A N/A"
        continue
    fi

    if [ ! -f "$TRAINER_CONFIG" ]; then
        echo "[ERROR] Trainer config not found: $TRAINER_CONFIG"
        exit 1
    fi

    echo "------------------------------------------------------------------------"
    echo "  Job $TOTAL_JOBS: $TRAINER on $ds"
    echo "  Dataset config : $DATASET_CONFIG"
    echo "  Trainer config : $TRAINER_CONFIG"
    echo "  Output dir     : $OUTPUT_DIR"
    echo "  Log file       : $LOG_FILE"
    echo "------------------------------------------------------------------------"

    python train.py \
        --root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-config-file "$DATASET_CONFIG" \
        --config-file "$TRAINER_CONFIG" \
        --trainer "$TRAINER" \
        --eval-only \
        --opts TRAINER.BIOMEDCLIP_ZEROSHOT.PROMPT_MODE "$PROMPT_MODE" 2>&1 | tee "${LOG_FILE}"

    TRAIN_EXIT=${PIPESTATUS[0]}

    # 浠庢棩蹇椾腑瑙ｆ瀽 accuracy 鍜?macro_f1
    read -r ACC F1 <<< $(python3 -c "
import re
try:
    with open('${LOG_FILE}') as f:
        text = f.read()
    blocks = re.split(r'=>\s*result', text)
    if len(blocks) >= 2:
        last = blocks[-1]
        end = last.find('=>')
        if end >= 0:
            last = last[:end]
        acc_m = re.search(r'\* accuracy:\s*([\d.]+)', last)
        f1_m  = re.search(r'\* macro_f1:\s*([\d.]+)', last)
        if acc_m:
            print(acc_m.group(1), f1_m.group(1) if f1_m else 'N/A')
            exit(0)
except Exception:
    pass
print('N/A N/A')
" 2>/dev/null || echo "N/A N/A")

    if [ "${TRAIN_EXIT}" -eq 0 ] && [ "${ACC}" != "N/A" ]; then
        RESULTS["${ds}"]="${ACC} ${F1}"
        echo "[OK]   $TRAINER on $ds: acc=${ACC}%, f1=${F1}%"
    else
        RESULTS["${ds}"]="N/A N/A"
        echo "[FAIL] $TRAINER on $ds failed or result not parsed."
        FAIL_JOBS=$((FAIL_JOBS + 1))
    fi
    echo ""
done

# 鈹€鈹€ 鍐欏叆缁撴灉鏂囦欢 鈹€鈹€
{
    echo "========================================================================"
    echo "  BiomedCLIP_PEFT Zero-Shot Evaluation (${PROMPT_MODE})"
    echo "  鏃堕棿   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Trainer: ${TRAINER}"
    echo "  Prompt : ${PROMPT_MODE}"
    echo "========================================================================"
    echo ""
    printf "%-15s %12s %12s\n" "Dataset" "Accuracy(%)" "Macro_F1(%)"
    printf "%-15s %12s %12s\n" "---------------" "------------" "------------"
    for ds in "${DATASETS[@]}"; do
        IFS=' ' read -r acc f1 <<< "${RESULTS[${ds}]}"
        printf "%-15s %12s %12s\n" "${ds}" "${acc}" "${f1}"
    done
    echo ""
    echo "  Summary: success=$((TOTAL_JOBS - SKIP_JOBS - FAIL_JOBS)) / total=$((TOTAL_JOBS - SKIP_JOBS)) (skipped=${SKIP_JOBS}, failed=${FAIL_JOBS})"
    echo "========================================================================"
} > "${RESULT_FILE}"

echo "========================================================================"
echo "  Zero-shot Evaluation Summary"
echo "========================================================================"
echo "  Total jobs : $TOTAL_JOBS"
echo "  Skipped    : $SKIP_JOBS"
echo "  Failed     : $FAIL_JOBS"
echo "  Succeeded  : $((TOTAL_JOBS - SKIP_JOBS - FAIL_JOBS))"
echo "  Result     : ${RESULT_FILE}"
echo "========================================================================"

if [ $FAIL_JOBS -ne 0 ]; then
    exit 1
fi
