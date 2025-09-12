#!/bin/bash

# 生成組合圖表腳本
# 在同一張圖表中比較AMM和Steer策略

echo "🚀 Generating Combined AMM vs Steer Strategy Plots"
echo "=================================================="

# 設置變量
POOL=${1:-"ALL"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📊 Configuration:"
echo "  Pool: $POOL"
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
if [ ! -f "generate_combined_plots.py" ]; then
    echo "❌ generate_combined_plots.py not found"
    exit 1
fi

echo "✅ Dependencies check passed"
echo ""

# 運行組合圖表生成
echo "🎯 Generating combined plots..."
echo "This will create plots showing AMM and Steer strategies together..."
echo ""

python generate_combined_plots.py

# 檢查結果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Combined plots generation completed successfully!"
    echo ""
    echo "📁 Results location:"
    echo "  📊 Charts: reports/figs/{POOL}/"
    echo "  📈 Combined plots: *_combined_*.png"
    echo ""
    echo "🔍 To view results:"
    echo "  find reports/figs -name '*combined*' | sort"
    echo "  ls -la reports/figs/btcusdc/*combined*"
    echo ""
    echo "📊 Generated plot types:"
    echo "  - Combined APR vs MDD Scatter"
    echo "  - Combined Sensitivity Heatmap"
    echo "  - Combined Equity Curves"
    echo ""
    echo "🎯 Key features:"
    echo "  - AMM strategies: ○ markers, solid lines"
    echo "  - Steer strategies: △ markers, dashed lines"
    echo "  - Direct comparison in same chart"
    echo "  - Color-coded by strategy type"
else
    echo ""
    echo "❌ Combined plots generation failed!"
    echo "Check the error messages above for details."
    exit 1
fi
