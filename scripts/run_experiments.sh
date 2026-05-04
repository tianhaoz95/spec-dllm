#!/usr/bin/env bash
# Full experiment pipeline. Run sections individually or all at once.
# Assumes .venv is activated: source .venv/bin/activate

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PY=".venv/bin/python"

# ── Phase 1: Feature extraction ─────────────────────────────────────────────
echo "=== Phase 1: Pre-extract Qwen3-8B features ==="
$PY scripts/extract_features.py --config configs/primary.yaml

# ── Phase 2: Training ────────────────────────────────────────────────────────
echo "=== Phase 2a: Experiment 1 — BD-EAGLE (warm start, uniform masking) ==="
$PY scripts/train_bd_eagle.py --config configs/primary.yaml

echo "=== Phase 2b: Experiment 2 — BD-EAGLE (cold start) ==="
$PY scripts/train_bd_eagle.py --config configs/cold_start.yaml

echo "=== Phase 2c: Experiment 3 — BD-EAGLE (anchor masking) ==="
$PY scripts/train_bd_eagle.py --config configs/anchor_masking.yaml

# ── Phase 3: Throughput evaluation ──────────────────────────────────────────
echo "=== Phase 3: Throughput evaluation ==="

for SYSTEM in baseline eagle3 dflash bd_eagle; do
  CKPT=""
  if [ "$SYSTEM" = "bd_eagle" ]; then
    CKPT="--checkpoint checkpoints/bd_eagle_primary/step_011500"
  fi
  $PY scripts/eval_throughput.py \
    --config configs/primary.yaml \
    --system "$SYSTEM" \
    $CKPT \
    --n_prompts 500 \
    --output "results/throughput_${SYSTEM}.json"
done

# ── Phase 4: Quality evaluation (GSM8K) ─────────────────────────────────────
echo "=== Phase 4: Quality evaluation (GSM8K) ==="
for SYSTEM in baseline eagle3 dflash bd_eagle; do
  $PY scripts/eval_quality.py \
    --config configs/primary.yaml \
    --system "$SYSTEM" \
    --benchmark gsm8k \
    --output "results/quality_gsm8k_${SYSTEM}.json"
done

echo "=== All phases complete. Results in results/ ==="
