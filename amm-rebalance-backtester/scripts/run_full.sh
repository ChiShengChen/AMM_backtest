#!/bin/bash

# Full analysis script for AMM backtester
# Usage: ./scripts/run_full.sh [POOL] [FREQ] [FEE_MODE] [STUDY_NAME] [N_TRIALS]

set -e

# Default values
POOL=${1:-ETHUSDC}
FREQ=${2:-1h}
FEE_MODE=${3:-proxy}
STUDY_NAME=${4:-ethusdc_wfa}
N_TRIALS=${5:-50}

echo "Running full analysis for $POOL with $FREQ data"
echo "Fee mode: $FEE_MODE, Study: $STUDY_NAME, Trials: $N_TRIALS"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
python3 -c "import click, yaml, rich, pandas, numpy, optuna" 2>/dev/null || {
    echo "Installing required packages..."
    pip install -r requirements.txt
}

# Run full analysis
python3 run.py full \
    --pool "$POOL" \
    --freq "$FREQ" \
    --fee-mode "$FEE_MODE" \
    --study-name "$STUDY_NAME" \
    --n-trials "$N_TRIALS"

echo "Full analysis completed! Check results/ and reports/figs/ directories."
echo "Optuna study saved to optuna_studies.db"
