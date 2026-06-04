#!/bin/bash

echo "================================"
echo "       ExpressiveMIDI"
echo "================================"

# Activate venv
source "$(dirname "$0")/venv/bin/activate"

# Find latest models
STYLE_MODEL=""
MODEL=$(ls -t experiments/best_hybrid_*.pt 2>/dev/null | head -1)
EXPR_MODEL=$(ls -t experiments/best_expression_*.pt 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then
    echo "ERROR: No trained model found in experiments/"
    exit 1
fi
echo "Using timing model: $MODEL"
if [ -n "$EXPR_MODEL" ]; then
    echo "Using expression model: $EXPR_MODEL"
fi

# Get input file
read -p "Enter path to input MIDI file: " INPUT
if [ ! -f "$INPUT" ]; then
    echo "ERROR: File not found: $INPUT"
    exit 1
fi

# Get output path
BASENAME=$(basename "$INPUT" .mid)
DEFAULT_OUT="${BASENAME}_humanised.mid"
read -p "Enter output filename [$DEFAULT_OUT]: " OUTPUT
OUTPUT="${OUTPUT:-$DEFAULT_OUT}"

# Get strength
read -p "Enter strength (0.5=subtle, 1.0=natural, 2.0=exaggerated) [1.0]: " STRENGTH
STRENGTH="${STRENGTH:-1.0}"

# Ask for style model
read -p "Use a style model? (leave blank to skip): " STYLE_INPUT
if [ -n "$STYLE_INPUT" ] && [ -f "$STYLE_INPUT" ]; then
    STYLE_MODEL="$STYLE_INPUT"
    echo "Using style model: $STYLE_MODEL"
fi

echo ""
echo "Processing..."
TIMING_MODEL="${STYLE_MODEL:-$MODEL}"
if [ -n "$EXPR_MODEL" ]; then
    python -m src.inference.humanise \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$TIMING_MODEL" \
        --expr_model "$EXPR_MODEL" \
        --strength "$STRENGTH"
else
    python -m src.inference.humanise \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$TIMING_MODEL" \
        --strength "$STRENGTH"
fi

echo ""
echo "Done! Output saved to: $OUTPUT"
