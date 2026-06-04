#!/bin/bash
echo "================================"
echo "   ExpressiveMIDI Style Cloning"
echo "================================"

source "$(dirname "$0")/venv/bin/activate"

PRETRAINED=$(ls -t experiments/best_hybrid_*.pt 2>/dev/null | head -1)
if [ -z "$PRETRAINED" ]; then
    echo "ERROR: No pretrained model found"
    exit 1
fi

read -p "Enter your style name (e.g. john): " NAME
if [ -z "$NAME" ]; then echo "ERROR: Name required"; exit 1; fi

echo "Enter paths to your MIDI files (3-5 recommended)."
echo "Press Enter with empty input when done."
MIDIS=()
while true; do
    read -p "MIDI file path: " MIDI
    [ -z "$MIDI" ] && break
    [ -f "$MIDI" ] && MIDIS+=("$MIDI") || echo "File not found, skipping"
done

if [ ${#MIDIS[@]} -lt 1 ]; then echo "ERROR: No valid files"; exit 1; fi

echo ""
echo "Fine-tuning on ${#MIDIS[@]} files..."
python -m src.training.finetune \
    --midis "${MIDIS[@]}" \
    --name "$NAME" \
    --pretrained "$PRETRAINED"

echo ""
echo "Done! Style model saved to experiments/style_${NAME}.pt"
echo "Use it with: bash humanise.sh (enter the style model path when asked)"
