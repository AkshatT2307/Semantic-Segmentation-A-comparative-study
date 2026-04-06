#!/usr/bin/env bash
# ============================================================================
#  train_all.sh — Train all deep segmentation models on Cityscapes,
#                 then run evaluation for classical + ML methods,
#                 all in one go.
#
#  Usage:
#      chmod +x train_all.sh
#      nohup ./train_all.sh &> train_all.log &
#      # or:
#      ./train_all.sh 2>&1 | tee train_all.log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="./data"
DATASET="cityscapes"
BATCH_SIZE=96
EPOCHS=50
ENCODER="resnet34"
LR=0.001
WEIGHT_DECAY=0.00001

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Capture ALL output (stdout + stderr) to a single master log ──────────────
MASTER_LOG="${LOG_DIR}/train_all_output.txt"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "========================================"
echo "  semseg-benchmark — Train All"
echo "  Started: $(date)"
echo "  Logs:    ${LOG_DIR}/"
echo "  Master:  ${MASTER_LOG}"
echo "========================================"

# ── 1. Validate everything first ─────────────────────────────────────────────
echo ""
echo ">>> [0/6] Running validation suite..."
python validate_all.py 2>&1 | tee "${LOG_DIR}/00_validate.log"
echo ">>> Validation complete."

# ── 2. Deep model training ───────────────────────────────────────────────────
for MODEL in unet segformer mask_rcnn; do
    echo ""
    echo "========================================"
    echo ">>> Training: ${MODEL} (${DATASET}, encoder=${ENCODER}, epochs=${EPOCHS})"
    echo ">>> Started:  $(date)"
    echo "========================================"

    python train.py \
        --model "$MODEL" \
        --encoder "$ENCODER" \
        --dataset "$DATASET" \
        --data-root "$DATA_ROOT" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight-decay "$WEIGHT_DECAY" \
        2>&1 | tee "${LOG_DIR}/train_${MODEL}.log"

    echo ">>> ${MODEL} training finished at $(date)"

    # Verify checkpoints were saved
    CKPT_DIR="runs/${MODEL}_${DATASET}/weights"
    if [ -f "${CKPT_DIR}/best.pt" ] && [ -f "${CKPT_DIR}/last.pt" ]; then
        echo ">>> ✓ Checkpoints saved:"
        ls -lh "${CKPT_DIR}/best.pt" "${CKPT_DIR}/last.pt"
    else
        echo ">>> ✗ WARNING: Checkpoints missing in ${CKPT_DIR}!"
    fi

    # Verify training log was saved
    LOG_FILE="runs/${MODEL}_${DATASET}/training_log.json"
    if [ -f "$LOG_FILE" ]; then
        echo ">>> ✓ Training log: ${LOG_FILE}"
    else
        echo ">>> ✗ WARNING: Training log missing!"
    fi

    # Verify training curves were saved
    CURVE_FILE="runs/${MODEL}_${DATASET}/training_curves.png"
    if [ -f "$CURVE_FILE" ]; then
        echo ">>> ✓ Training curves: ${CURVE_FILE}"
    else
        echo ">>> ✗ WARNING: Training curves missing!"
    fi
done

# ── 3. Classical + ML evaluation on val split ────────────────────────────────
for METHOD in otsu global edge graph_cut region kmeans gmm svm; do
    echo ""
    echo "========================================"
    echo ">>> Evaluating: ${METHOD} on ${DATASET} (val)"
    echo ">>> Started:  $(date)"
    echo "========================================"

    python run.py \
        --method "$METHOD" \
        --dataset "$DATASET" \
        --data-root "$DATA_ROOT" \
        --split val \
        --batch-size 1 \
        --visualize \
        --vis-count 5 \
        2>&1 | tee "${LOG_DIR}/eval_${METHOD}.log"

    echo ">>> ${METHOD} evaluation finished at $(date)"
done

# ── 4. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  ALL DONE — $(date)"
echo "========================================"
echo ""
echo "Trained models (checkpoints):"
for MODEL in unet segformer mask_rcnn; do
    CKPT_DIR="runs/${MODEL}_${DATASET}/weights"
    if [ -d "$CKPT_DIR" ]; then
        echo "  ${MODEL}:"
        ls -lh "${CKPT_DIR}/" 2>/dev/null | grep ".pt" || echo "    (no checkpoints)"
    fi
done
echo ""
echo "Evaluation results saved in: ${LOG_DIR}/"
echo "Visualization maps saved in: results/"
echo ""
echo "All logs:"
ls -lh "${LOG_DIR}/"
