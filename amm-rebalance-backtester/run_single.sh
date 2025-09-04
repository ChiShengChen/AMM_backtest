#!/bin/bash

# 單個幣種回測腳本
# 使用方法: ./run_single.sh [POOL] [FREQ] [TRIALS]

# 設置默認值
POOL=${1:-ETHUSDC}
FREQ=${2:-1d}
TRIALS=${3:-10}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 開始運行 $POOL 回測"
echo "=================================="
echo "📊 幣種: $POOL"
echo "⏰ 時間尺度: $FREQ"
echo "🔬 試驗次數: $TRIALS"
echo "📅 時間戳: $TIMESTAMP"
echo ""

# 運行回測
echo "=== 運行 $POOL 回測 ==="
python run.py full \
    --pool $POOL \
    --freq $FREQ \
    --study-name "${POOL,,}_${FREQ}_${TIMESTAMP}" \
    --n-trials $TRIALS

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ $POOL 回測完成！"
    echo "=================================="
    echo "📊 結果保存在: results/"
    echo "📈 圖表保存在: reports/figs/${POOL,,}/"
    echo ""
    
    # 顯示最新結果
    echo "📋 最新策略記錄:"
    find results/ -name "strategy_record_*.json" -type f | sort | tail -1 | while read file; do
        echo "  📄 $file"
        grep -A 3 "best_strategy" "$file" | head -4
    done
else
    echo ""
    echo "❌ $POOL 回測失敗"
    exit 1
fi

echo ""
echo "✨ 腳本執行完成！"
