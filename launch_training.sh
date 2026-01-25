#!/bin/bash
# GLM-4.7 RUST AGENT - HYBRID PRECISION TRAINING LAUNCHER
# Authorization: TPM-BMAD-2026-Q1-GO

set -e  # Exit on error

echo "================================================ட்டான"
echo "GLM-4.7 RUST AGENT - HYBRID PRECISION TRAINING"
echo "Variant B: Safe Mode (Unsloth + No FA2)"
echo "================================================ட்டான"

# ============================================
# STEP 1: DATA GENERATION
# ============================================
echo ""
echo "📊 STEP 1: Data Transpilation"
echo "----------------------------------------"

if [ ! -f "glaive-function-calling-v2.jsonl" ]; then
    echo "⬇️  Downloading dataset..."
    wget -q https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.jsonl
fi

python3 scripts/01_transpile_glaive_to_glm4.py \
    --input glaive-function-calling-v2.jsonl \
    --output data/rust_agent_native_format.jsonl \
    --max_samples 50000

# Verify output integrity
echo ""
echo "🔍 Verifying data format..."
SAMPLE=$(head -n 1 data/rust_agent_native_format.jsonl)
if echo "$SAMPLE" | grep -q '\[TOOL_CALLS\]'; then
    echo "✅ [TOOL_CALLS] token detected"
else
    echo "❌ ERROR: [TOOL_CALLS] token missing in output"
    exit 1
fi

# ============================================
# STEP 2: CREATE VALIDATION SPLIT
# ============================================
echo ""
echo "📊 STEP 2: Creating Validation Split"
echo "----------------------------------------"

# Take last 2000 samples for validation
TOTAL_LINES=$(wc -l < data/rust_agent_native_format.jsonl)
TRAIN_LINES=$((TOTAL_LINES - 2000))

head -n $TRAIN_LINES data/rust_agent_native_format.jsonl > data/rust_agent_native_format_train.jsonl
tail -n 2000 data/rust_agent_native_format.jsonl > data/rust_validation_holdout.jsonl

echo "✅ Train: $TRAIN_LINES samples"
echo "✅ Validation: 2000 samples"

# Update config to use train split (sed hack for automation)
sed -i 's/dataset: rust_agent_native_format/dataset: rust_agent_native_format_train/g' config/train_glm4_rust_v3_FINAL.yaml

# ============================================
# STEP 3: SYSTEM CHECKS
# ============================================
echo ""
echo "🔧 STEP 3: System Validation"
echo "----------------------------------------"

# Check GPU Memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
if [ "$GPU_MEM" -lt 75000 ]; then
    echo "⚠️  WARNING: GPU has ${GPU_MEM}MB (< 75GB recommended)"
    echo "    Switching batch_size to 1 to prevent OOM..."
    sed -i 's/per_device_train_batch_size: 2/per_device_train_batch_size: 1/g' config/train_glm4_rust_v3_FINAL.yaml
fi
echo "✅ GPU: ${GPU_MEM}MB VRAM available"

# ============================================
# STEP 4: START MONITORING
# ============================================
echo ""
echo "👁️  STEP 4: Initializing Monitor"
echo "----------------------------------------"

if screen -list | grep -q "training_monitor"; then
    screen -X -S training_monitor quit
fi

screen -dmS training_monitor python3 scripts/02_monitor_training.py
sleep 2

if screen -list | grep -q "training_monitor"; then
    echo "✅ Monitor active (screen session: training_monitor)"
else
    echo "❌ WARNING: Monitor failed to start"
fi

# ============================================
# STEP 5: LAUNCH TRAINING
# ============================================
echo ""
echo "🚀 STEP 5: LAUNCHING TRAINING"
echo "========================================"
echo "  Model: THUDM/glm-4.7-flash"
echo "  Format: Native [TOOL_CALLS]"
echo "  Protected: moe.router, attention, lm_head"
echo ""
echo "Kill Triggers Active: Loss > 5.0 | Plateau @ 300 | VRAM > 78GB"
echo ""

# Launch training
llamafactory-cli train config/train_glm4_rust_v3_FINAL.yaml

echo ""
echo "================================================