#!/bin/bash

# AMM 回測完整腳本
# 運行所有幣種的回測和優化

echo "🚀 開始運行 AMM 回測完整分析"
echo "=================================="

# 設置變量
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
N_TRIALS=20  # 增加試驗次數以獲得更好的優化結果

echo "📅 時間戳: $TIMESTAMP"
echo "🔬 試驗次數: $N_TRIALS"
echo ""

# 1. ETH/USDC 深度優化回測
echo "=== 1. ETH/USDC 深度優化回測 ==="
echo "運行 ETH/USDC 1d 數據，$N_TRIALS 次試驗..."
python run.py full \
    --pool ETHUSDC \
    --freq 1d \
    --study-name "ethusdc_deep_${TIMESTAMP}" \
    --n-trials $N_TRIALS

if [ $? -eq 0 ]; then
    echo "✅ ETH/USDC 回測完成"
else
    echo "❌ ETH/USDC 回測失敗"
    exit 1
fi

echo ""

# 2. BTC/USDC 深度優化回測
echo "=== 2. BTC/USDC 深度優化回測 ==="
echo "運行 BTC/USDC 1d 數據，$N_TRIALS 次試驗..."
python run.py full \
    --pool BTCUSDC \
    --freq 1d \
    --study-name "btcusdc_deep_${TIMESTAMP}" \
    --n-trials $N_TRIALS

if [ $? -eq 0 ]; then
    echo "✅ BTC/USDC 回測完成"
else
    echo "❌ BTC/USDC 回測失敗"
    exit 1
fi

echo ""

# 3. USDC/USDT 深度優化回測
echo "=== 3. USDC/USDT 深度優化回測 ==="
echo "運行 USDC/USDT 1d 數據，$N_TRIALS 次試驗..."
python run.py full \
    --pool USDCUSDT \
    --freq 1d \
    --study-name "usdcusdt_deep_${TIMESTAMP}" \
    --n-trials $N_TRIALS

if [ $? -eq 0 ]; then
    echo "✅ USDC/USDT 回測完成"
else
    echo "❌ USDC/USDT 回測失敗"
    exit 1
fi

echo ""

# 4. 快速測試 (可選)
echo "=== 4. 快速測試 ==="
echo "運行快速測試以驗證系統..."
python run.py quick --pool ETHUSDC --freq 1d

if [ $? -eq 0 ]; then
    echo "✅ 快速測試完成"
else
    echo "⚠️  快速測試失敗，但繼續執行"
fi

echo ""

# 5. 生成結果摘要
echo "=== 5. 生成結果摘要 ==="
echo "檢查所有生成的結果..."

# 檢查結果目錄
echo "📁 結果目錄:"
ls -la results/ | grep "^d" | head -10

# 檢查圖表目錄
echo "📊 圖表目錄:"
ls -la reports/figs/*/ | grep "\.png$" | wc -l | xargs echo "總圖表數量:"

echo ""

# 6. 完成
echo "🎉 所有回測完成！"
echo "=================================="
echo "📊 結果保存在: results/"
echo "📈 圖表保存在: reports/figs/"
echo "🔍 查看目錄索引: cat results/INDEX.md"
echo ""

# 顯示最新的策略記錄
echo "📋 最新策略記錄:"
find results/ -name "strategy_record_*.json" -type f | sort | tail -3 | while read file; do
    echo "  📄 $file"
    grep -A 3 "best_strategy" "$file" | head -4
done

echo ""
echo "✨ 腳本執行完成！"