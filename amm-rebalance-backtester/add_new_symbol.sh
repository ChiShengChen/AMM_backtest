#!/bin/bash

# 快速添加新幣種腳本
# 使用方法: ./add_new_symbol.sh SYMBOL [FREQ] [DAYS]

set -e

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 檢查參數
if [[ $# -lt 1 ]]; then
    echo "🚀 快速添加新幣種腳本"
    echo "=================================="
    echo ""
    echo "使用方法:"
    echo "  $0 SYMBOL [FREQ] [DAYS]"
    echo ""
    echo "參數:"
    echo "  SYMBOL    幣種代碼 (如: SOLUSDC, ADAUSDC)"
    echo "  FREQ      時間尺度 (默認: 1d)"
    echo "  DAYS      下載天數 (默認: 365)"
    echo ""
    echo "示例:"
    echo "  $0 SOLUSDC           # 下載 SOL/USDC 1年日線數據"
    echo "  $0 ADAUSDC 1h 30    # 下載 ADA/USDC 30天小時數據"
    echo ""
    exit 1
fi

SYMBOL="$1"
FREQ="${2:-1d}"
DAYS="${3:-365}"

echo "🚀 快速添加新幣種: $SYMBOL"
echo "=================================="
echo "📊 幣種: $SYMBOL"
echo "⏰ 時間尺度: $FREQ"
echo "📅 天數: $DAYS"
echo ""

# 檢查下載腳本
if [[ ! -f "download_5year_data.py" ]]; then
    echo -e "${YELLOW}⚠️  下載腳本未找到，請先創建 download_5year_data.py${NC}"
    exit 1
fi

# 創建目錄
echo -e "${BLUE}📁 創建目錄...${NC}"
mkdir -p "data/$SYMBOL"
mkdir -p "data/temp"

# 計算日期
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "$END_DATE - $DAYS days" +%Y-%m-%d)

echo "📅 時間範圍: $START_DATE 到 $END_DATE"

# 下載數據
echo -e "${BLUE}📥 下載數據...${NC}"
if python3 download_5year_data.py \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --out "data/temp"; then
    
    echo -e "${GREEN}✅ 數據下載完成${NC}"
    
    # 查找下載的文件
    DOWNLOADED_FILE=$(find "data/temp" -name "*${SYMBOL}_${FREQ}*.csv" | head -1)
    
    if [[ -n "$DOWNLOADED_FILE" ]]; then
        # 移動並重命名文件
        TARGET_FILE="data/$SYMBOL/price_${FREQ}.csv"
        mv "$DOWNLOADED_FILE" "$TARGET_FILE"
        echo "📄 文件已移動到: $TARGET_FILE"
        
        # 創建說明文件
        cat > "data/$SYMBOL/README.md" << EOF
# $SYMBOL 數據說明

## 文件信息
- **文件名**: price_${FREQ}.csv
- **時間尺度**: ${FREQ}
- **下載時間**: $(date)
- **數據行數**: $(wc -l < "$TARGET_FILE")
- **時間範圍**: $START_DATE 到 $END_DATE

## 數據格式
CSV 格式，包含以下列：
- timestamp: 時間戳
- open: 開盤價
- high: 最高價
- low: 最低價
- close: 收盤價
- volume: 交易量

## 使用說明
此數據可用於 AMM 回測分析：

\`\`\`bash
# 快速測試
python run.py quick --pool $SYMBOL --freq ${FREQ}

# 完整回測
python run.py full --pool $SYMBOL --freq ${FREQ} --study-name ${SYMBOL,,}_${FREQ}_$(date +%Y%m%d_%H%M%S) --n-trials 10
\`\`\`

## 注意事項
- 確保數據完整性
- 檢查時間戳格式
- 驗證價格數據合理性
EOF
        
        echo "📝 說明文件已創建: data/$SYMBOL/README.md"
        
        # 顯示文件信息
        echo ""
        echo -e "${GREEN}📋 文件信息${NC}"
        echo "=================================="
        echo "📄 文件名: $(basename "$TARGET_FILE")"
        echo "📊 數據行數: $(wc -l < "$TARGET_FILE")"
        echo "💾 文件大小: $(du -h "$TARGET_FILE" | cut -f1)"
        echo "📁 目錄: data/$SYMBOL/"
        
        # 清理臨時文件
        rm -rf "data/temp"
        
        echo ""
        echo -e "${GREEN}🎉 新幣種 $SYMBOL 添加完成！${NC}"
        echo "=================================="
        echo "🔍 查看數據: ls -la data/$SYMBOL/"
        echo "📖 查看說明: cat data/$SYMBOL/README.md"
        echo "🚀 開始回測: python run.py quick --pool $SYMBOL --freq ${FREQ}"
        
    else
        echo -e "${YELLOW}⚠️  未找到下載的數據文件${NC}"
        exit 1
    fi
    
else
    echo -e "${YELLOW}❌ 數據下載失敗${NC}"
    exit 1
fi
