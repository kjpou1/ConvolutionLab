#!/bin/bash

# Store the original working directory
ORIGINAL_DIR="$(pwd)"

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change the working directory to the script directory
cd "$SCRIPT_DIR" || exit

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Default input, output, and model file paths
INPUT_FILE=""
OUTPUT_FILE=""
MODEL_FILE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_FILE="$2"; shift ;;  # Capture --input argument
        --output) OUTPUT_FILE="$2"; shift ;;  # Capture --output argument
        --model-file) MODEL_FILE="$2"; shift ;;  # Capture --model-file argument
        *) echo "Unknown parameter: $1" ;;  # Handle unexpected arguments
    esac
    shift
done

# Build the command dynamically
CMD="python predict.py"

# Append input file argument if provided
if [[ -n "$INPUT_FILE" ]]; then
    CMD+=" --input \"$INPUT_FILE\""
fi

# Append output file argument if provided
if [[ -n "$OUTPUT_FILE" ]]; then
    CMD+=" --output \"$OUTPUT_FILE\""
fi

# Append model file argument if provided
if [[ -n "$MODEL_FILE" ]]; then
    CMD+=" --model-file \"$MODEL_FILE\""
fi

# Run the command
eval $CMD

# Deactivate virtual environment
deactivate

# Restore the original working directory
cd "$ORIGINAL_DIR"
