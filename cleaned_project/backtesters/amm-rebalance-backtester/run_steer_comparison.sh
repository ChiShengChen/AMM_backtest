#!/bin/bash

# Steer策略整合比較腳本
# 比較 AMM 回測系統 + Steer Intent Backtester 策略

echo "🚀 Starting AMM + Steer Strategies Comparison"
echo "=============================================="

# 設置變量
POOL=${1:-"BTCUSDC"}
FREQ=${2:-"1d"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📊 Configuration:"
echo "  Pool: $POOL"
echo "  Frequency: $FREQ"
echo "  Timestamp: $TIMESTAMP"
echo ""

# 檢查依賴
echo "🔍 Checking dependencies..."

# 檢查Python環境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# 檢查必要文件
if [ ! -f "final_integration.py" ]; then
    echo "❌ final_integration.py not found"
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

# 運行整合比較
echo "🎯 Running AMM vs Steer strategies comparison..."
echo "This will test both AMM and Steer strategies..."
echo ""

python final_integration.py

# 檢查結果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Integration comparison completed successfully!"
    echo ""
    echo "📁 Results location:"
    echo "  📊 Charts: reports/figs/${POOL}/"
    echo "  📈 Data: results/"
    echo "  📋 Report: results/final_integration_report_*.txt"
    echo ""
    echo "🔍 To view results:"
    echo "  ls -la results/final_integration_report_*.txt"
    echo "  cat results/final_integration_report_*.txt"
    echo ""
    echo "📊 Summary:"
    echo "  - AMM strategies: Traditional AMM rebalancing approaches"
    echo "  - Steer strategies: Advanced intent-based strategies"
    echo "  - Best performing strategy will be highlighted in the report"
else
    echo ""
    echo "❌ Integration comparison failed!"
    echo "Check the error messages above for details."
    exit 1
fi
