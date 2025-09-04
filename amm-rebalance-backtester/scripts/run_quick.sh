#!/bin/bash

# Quick test script for AMM backtester
# Usage: ./scripts/run_quick.sh [POOL] [FREQ] [FEE_MODE]

set -e

# Default values
POOL=${1:-ETHUSDC}
FREQ=${2:-1h}
FEE_MODE=${3:-proxy}

echo "Running quick test for $POOL with $FREQ data (fee mode: $FEE_MODE)"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
python3 -c "import click, yaml, rich, pandas, numpy" 2>/dev/null || {
    echo "Installing required packages..."
    pip install -r requirements.txt
}

# Run quick test
python3 run.py quick \
    --pool "$POOL" \
    --freq "$FREQ" \
    --fee-mode "$FEE_MODE"

echo "Quick test completed! Check results/ and reports/figs/ directories."
