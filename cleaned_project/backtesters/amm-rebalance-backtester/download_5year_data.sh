#!/bin/bash
# 批量下載5年日線數據腳本
# 支持 ETH/USDC, BTC/USDC, USDT/USDC

set -e  # 遇到錯誤時退出

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/5year_daily"
LOG_FILE="$PROJECT_DIR/download_5year_data.log"

# 創建目錄
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 日誌函數
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 交易對配置
PAIRS=(
    "ETHUSDC"
    "BTCUSDC" 
    "USDCUSDT"  # 注意：Binance 沒有 USDTUSDC，用 USDCUSDT 替代
)

# 時間範圍：5年前到今天
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "5 years ago" +%Y-%m-%d)

log "開始批量下載5年日線數據"
log "時間範圍: $START_DATE 到 $END_DATE"
log "交易對: ${PAIRS[*]}"
log "數據目錄: $DATA_DIR"

# 檢查 Python 環境
if ! command -v python3 &> /dev/null; then
    log "錯誤: 需要 Python 3"
    exit 1
fi

# 檢查必要的 Python 包
python3 -c "import requests, pandas" 2>/dev/null || {
    log "安裝必要的 Python 包..."
    pip3 install requests pandas
}

# 下載每個交易對的數據
success_count=0
total_count=${#PAIRS[@]}

for pair in "${PAIRS[@]}"; do
    log "正在處理 $pair..."
    
    # 生成輸出文件名
    output_file="$DATA_DIR/${pair}_1d_5year.csv"
    
    # 使用 Python 腳本下載
    if python3 "$SCRIPT_DIR/download_5year_data.py" --symbol "$pair" --start "$START_DATE" --end "$END_DATE" --out "$output_file"; then
        log "✅ $pair 下載成功"
        success_count=$((success_count + 1))
        
        # 檢查文件
        if [ -f "$output_file" ]; then
            record_count=$(wc -l < "$output_file")
            file_size=$(du -h "$output_file" | cut -f1)
            log "  記錄數: $((record_count - 1)) (排除標題)"
            log "  文件大小: $file_size"
        fi
    else
        log "❌ $pair 下載失敗"
    fi
    
    # 在交易對之間添加延遲 (Binance 速率限制: 1200 requests/min)
    if [ "$pair" != "${PAIRS[-1]}" ]; then
        log "等待 5 秒後繼續下一個交易對..."
        sleep 5
    fi
done

# 總結報告
log "="*60
log "下載完成總結"
log "="*60
log "總交易對數: $total_count"
log "成功下載: $success_count"
log "失敗數量: $((total_count - success_count))"
log "數據保存目錄: $DATA_DIR"

if [ $success_count -eq $total_count ]; then
    log "🎉 所有交易對數據下載成功！"
else
    log "⚠️ 部分交易對下載失敗，請檢查日誌"
fi

# 顯示下載的文件
log "下載的文件:"
for file in "$DATA_DIR"/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        size=$(du -h "$file" | cut -f1)
        log "  $filename ($size)"
    fi
done

log "腳本執行完成"
