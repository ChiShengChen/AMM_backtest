#!/bin/bash

# 整合回測執行腳本
# 整合 AMM 回測系統 + Steer Intent Backtester 策略

echo "🚀 Starting Integrated AMM + Steer Strategies Backtest"
echo "=================================================="

# 設置變量
POOL=${1:-"BTCUSDC"}
FREQ=${2:-"1d"}
TRIALS=${3:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_NAME="integrated_${POOL}_${TIMESTAMP}"

echo "📊 Configuration:"
echo "  Pool: $POOL"
echo "  Frequency: $FREQ"
echo "  Trials: $TRIALS"
echo "  Study: $STUDY_NAME"
echo ""

# 檢查依賴
echo "🔍 Checking dependencies..."

# 檢查Python環境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# 檢查必要文件
if [ ! -f "run_integrated.py" ]; then
    echo "❌ run_integrated.py not found"
    exit 1
fi

if [ ! -f "configs/integrated_experiment.yaml" ]; then
    echo "❌ configs/integrated_experiment.yaml not found"
    exit 1
fi

# 檢查數據文件
DATA_DIR="data/${POOL}"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory $DATA_DIR not found"
    echo "Please ensure data files are properly set up"
    exit 1
fi

echo "✅ Dependencies check passed"
echo ""

# 運行整合回測
echo "🎯 Running integrated backtest..."
echo "This may take several minutes depending on the number of trials..."
echo ""

python run_integrated.py

# 檢查結果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Integrated backtest completed successfully!"
    echo ""
    echo "📁 Results location:"
    echo "  📊 Charts: reports/figs/${POOL}/"
    echo "  📈 Data: results/"
    echo "  📋 Report: results/integrated_report_${STUDY_NAME}.txt"
    echo ""
    echo "🔍 To view results:"
    echo "  ls -la reports/figs/${POOL}/"
    echo "  cat results/integrated_report_${STUDY_NAME}.txt"
else
    echo ""
    echo "❌ Integrated backtest failed!"
    echo "Check the error messages above for details."
    exit 1
fi
