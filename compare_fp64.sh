#!/bin/bash
#
# compare_fp64.sh - Compare native FP64 vs emulated FP64 performance
#
# Usage: ./compare_fp64.sh your_script.sh [args...]
#
# This wrapper runs your application twice:
# 1. With native FP64 (CUBLAS_PEDANTIC_MATH)
# 2. With emulated FP64 (CUBLAS_DEFAULT_MATH + ADP)
#
# It measures wall-clock time for both runs and reports the speedup.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script.sh> [args...]"
    echo ""
    echo "Example: $0 ./my_cuda_app.sh --matrix-size 8192"
    exit 1
fi

SCRIPT="$1"
shift
ARGS="$@"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' not found"
    exit 1
fi

echo "========================================"
echo "FP64 Performance Comparison"
echo "========================================"
echo "Script: $SCRIPT $ARGS"
echo ""

# ============================================
# Phase 1: Native FP64 (Pedantic Math)
# ============================================
echo "----------------------------------------"
echo "Phase 1: Native FP64 (Pedantic Math)"
echo "----------------------------------------"

export CUBLAS_MATH_MODE=CUBLAS_PEDANTIC_MATH
unset CUBLAS_EMULATION_STRATEGY
unset CUBLAS_WORKSPACE_CONFIG

START_NATIVE=$(date +%s.%N)
bash "$SCRIPT" $ARGS
END_NATIVE=$(date +%s.%N)

NATIVE_TIME=$(echo "$END_NATIVE - $START_NATIVE" | bc)
echo ""
echo "Native FP64 runtime: ${NATIVE_TIME}s"
echo ""

# ============================================
# Phase 2: Emulated FP64 (ADP)
# ============================================
echo "----------------------------------------"
echo "Phase 2: Emulated FP64 (ADP)"
echo "----------------------------------------"

export CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH
export CUBLAS_EMULATION_STRATEGY=performant
export CUBLAS_WORKSPACE_CONFIG=:4096:8

START_EMU=$(date +%s.%N)
bash "$SCRIPT" $ARGS
END_EMU=$(date +%s.%N)

EMU_TIME=$(echo "$END_EMU - $START_EMU" | bc)
echo ""
echo "Emulated FP64 runtime: ${EMU_TIME}s"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "Summary"
echo "========================================"
echo "Native FP64:   ${NATIVE_TIME}s"
echo "Emulated FP64: ${EMU_TIME}s"

if command -v bc &> /dev/null; then
    SPEEDUP=$(echo "scale=2; $NATIVE_TIME / $EMU_TIME" | bc)
    echo "Speedup:       ${SPEEDUP}x"
else
    echo "Speedup:       (install 'bc' to calculate)"
fi
echo ""
