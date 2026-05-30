#!/bin/bash

echo "================================"
echo "       ExpressiveMIDI"
echo "================================"

# Activate venv
source "$(dirname "$0")/venv/bin/activate"

# Find latest model
MODEL=$(ls -t experiments/best_hybrid_*.pt 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then
    echo "ERROR: No trained model found in experiments/"
    exit 1
fi
echo "Using model: $MODEL"

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

echo ""
echo "Processing..."
python -m src.inference.humanise \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --strength "$STRENGTH"

echo ""
echo "Done! Output saved to: $OUTPUT"
