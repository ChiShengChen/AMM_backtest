#!/bin/bash

# 數據獲取和整理腳本
# 支持新的幣種、時間尺度和數據源

set -e  # 遇到錯誤時退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置變量
DATA_DIR="data"
PYTHON_SCRIPT="download_5year_data.py"
DEFAULT_SYMBOLS=("ETHUSDC" "BTCUSDC" "USDCUSDT")
DEFAULT_FREQS=("1d" "1h" "1m")

# 顯示幫助信息
show_help() {
    echo "🚀 AMM 數據獲取和整理腳本"
    echo "=================================="
    echo ""
    echo "使用方法:"
    echo "  $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -s, --symbols SYMBOLS    指定交易對 (用逗號分隔，如: ETHUSDC,BTCUSDC)"
    echo "  -f, --frequencies FREQS  指定時間尺度 (用逗號分隔，如: 1d,1h,1m)"
    echo "  -d, --days DAYS          指定下載天數 (默認: 1825 = 5年)"
    echo "  -o, --output DIR         指定輸出目錄 (默認: data/)"
    echo "  -c, --clean              清理舊數據"
    echo "  -v, --validate           驗證下載的數據"
    echo "  -h, --help               顯示此幫助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 下載所有默認數據"
    echo "  $0 -s ETHUSDC,BTCUSDC -f 1d,1h      # 下載特定幣種和時間尺度"
    echo "  $0 -d 365 -o data/new_data          # 下載1年數據到指定目錄"
    echo "  $0 -c -v                             # 清理舊數據並驗證新數據"
    echo ""
}

# 解析命令行參數
parse_args() {
    SYMBOLS=("${DEFAULT_SYMBOLS[@]}")
    FREQS=("${DEFAULT_FREQS[@]}")
    DAYS=1825
    OUTPUT_DIR="$DATA_DIR"
    CLEAN_DATA=false
    VALIDATE_DATA=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--symbols)
                IFS=',' read -ra SYMBOLS <<< "$2"
                shift 2
                ;;
            -f|--frequencies)
                IFS=',' read -ra FREQS <<< "$2"
                shift 2
                ;;
            -d|--days)
                DAYS="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -c|--clean)
                CLEAN_DATA=true
                shift
                ;;
            -v|--validate)
                VALIDATE_DATA=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 未知選項: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# 檢查依賴
check_dependencies() {
    echo -e "${BLUE}🔍 檢查依賴...${NC}"
    
    # 檢查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安裝${NC}"
        exit 1
    fi
    
    # 檢查 Python 腳本
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo -e "${RED}❌ 下載腳本未找到: $PYTHON_SCRIPT${NC}"
        exit 1
    fi
    
    # 檢查必要的 Python 包
    python3 -c "import requests, pandas" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  安裝必要的 Python 包...${NC}"
        pip install requests pandas
    }
    
    echo -e "${GREEN}✅ 依賴檢查完成${NC}"
}

# 創建目錄結構
create_directories() {
    echo -e "${BLUE}📁 創建目錄結構...${NC}"
    
    # 創建主數據目錄
    mkdir -p "$OUTPUT_DIR"
    
    # 為每個幣種創建目錄
    for symbol in "${SYMBOLS[@]}"; do
        mkdir -p "$OUTPUT_DIR/$symbol"
        echo "  📁 創建目錄: $OUTPUT_DIR/$symbol"
    done
    
    # 創建臨時目錄
    mkdir -p "$OUTPUT_DIR/temp"
    
    echo -e "${GREEN}✅ 目錄結構創建完成${NC}"
}

# 清理舊數據
clean_old_data() {
    if [[ "$CLEAN_DATA" == true ]]; then
        echo -e "${YELLOW}🧹 清理舊數據...${NC}"
        
        # 備份重要數據
        if [[ -d "$DATA_DIR" ]]; then
            BACKUP_DIR="data_backup_$(date +%Y%m%d_%H%M%S)"
            echo "  📦 備份到: $BACKUP_DIR"
            cp -r "$DATA_DIR" "$BACKUP_DIR"
        fi
        
        # 清理舊數據
        rm -rf "$DATA_DIR"/*.csv
        rm -rf "$DATA_DIR"/*/price_*.csv
        rm -rf "$DATA_DIR"/*/pool_*.csv
        
        echo -e "${GREEN}✅ 舊數據清理完成${NC}"
    fi
}

# 下載數據
download_data() {
    echo -e "${BLUE}📥 開始下載數據...${NC}"
    
    for symbol in "${SYMBOLS[@]}"; do
        for freq in "${FREQS[@]}"; do
            echo ""
            echo -e "${BLUE}📊 下載 $symbol $freq 數據...${NC}"
            
            # 計算開始和結束日期
            END_DATE=$(date +%Y-%m-%d)
            START_DATE=$(date -d "$END_DATE - $DAYS days" +%Y-%m-%d)
            
            echo "  📅 時間範圍: $START_DATE 到 $END_DATE"
            echo "  ⏰ 時間尺度: $freq"
            
            # 運行 Python 下載腳本
            if python3 "$PYTHON_SCRIPT" \
                --symbol "$symbol" \
                --start "$START_DATE" \
                --end "$END_DATE" \
                --out "$OUTPUT_DIR/temp"; then
                
                echo -e "  ${GREEN}✅ $symbol $freq 數據下載完成${NC}"
                
                # 移動文件到正確位置
                move_downloaded_files "$symbol" "$freq"
            else
                echo -e "  ${RED}❌ $symbol $freq 數據下載失敗${NC}"
            fi
        done
    done
}

# 移動下載的文件
move_downloaded_files() {
    local symbol="$1"
    local freq="$2"
    
    echo "  📁 整理文件..."
    
    # 查找下載的文件
    local downloaded_file=$(find "$OUTPUT_DIR/temp" -name "*${symbol}_${freq}*.csv" | head -1)
    
    if [[ -n "$downloaded_file" ]]; then
        # 重命名並移動到正確位置
        local target_file="$OUTPUT_DIR/$symbol/price_${freq}.csv"
        mv "$downloaded_file" "$target_file"
        echo "    📄 移動到: $target_file"
        
        # 創建數據說明文件
        create_data_info "$symbol" "$freq" "$target_file"
    else
        echo -e "    ${YELLOW}⚠️  未找到 $symbol $freq 數據文件${NC}"
    fi
}

# 創建數據說明文件
create_data_info() {
    local symbol="$1"
    local freq="$2"
    local file_path="$3"
    
    local info_file="$OUTPUT_DIR/$symbol/README.md"
    
    cat > "$info_file" << EOF
# $symbol 數據說明

## 文件信息
- **文件名**: price_${freq}.csv
- **時間尺度**: ${freq}
- **下載時間**: $(date)
- **數據行數**: $(wc -l < "$file_path")

## 數據格式
CSV 格式，包含以下列：
- timestamp: 時間戳
- open: 開盤價
- high: 最高價
- low: 最低價
- close: 收盤價
- volume: 交易量

## 使用說明
此數據可用於 AMM 回測分析，支持以下命令：
\`\`\`bash
python run.py full --pool $symbol --freq ${freq}
\`\`\`
EOF
    
    echo "    📝 創建說明文件: $info_file"
}

# 驗證數據
validate_data() {
    if [[ "$VALIDATE_DATA" == true ]]; then
        echo -e "${BLUE}🔍 驗證下載的數據...${NC}"
        
        # 檢查是否有驗證腳本
        if [[ -f "validate_downloaded_data.py" ]]; then
            python3 validate_downloaded_data.py --data-dir "$OUTPUT_DIR"
        else
            echo -e "${YELLOW}⚠️  驗證腳本未找到，跳過驗證${NC}"
        fi
    fi
}

# 生成數據摘要
generate_summary() {
    echo ""
    echo -e "${BLUE}📋 數據摘要${NC}"
    echo "=================================="
    
    for symbol in "${SYMBOLS[@]}"; do
        if [[ -d "$OUTPUT_DIR/$symbol" ]]; then
            echo "📊 $symbol:"
            for freq in "${FREQS[@]}"; do
                local file_path="$OUTPUT_DIR/$symbol/price_${freq}.csv"
                if [[ -f "$file_path" ]]; then
                    local line_count=$(wc -l < "$file_path")
                    local file_size=$(du -h "$file_path" | cut -f1)
                    echo "  📄 $freq: $line_count 行, $file_size"
                else
                    echo "  ❌ $freq: 文件缺失"
                fi
            done
        else
            echo "❌ $symbol: 目錄缺失"
        fi
    done
    
    echo ""
    echo -e "${GREEN}🎉 數據獲取完成！${NC}"
    echo "=================================="
    echo "📁 數據目錄: $OUTPUT_DIR"
    echo "🔍 查看數據: ls -la $OUTPUT_DIR/*/"
    echo "📖 查看說明: cat $OUTPUT_DIR/*/README.md"
}

# 主函數
main() {
    echo "🚀 開始 AMM 數據獲取和整理"
    echo "=================================="
    echo "📊 幣種: ${SYMBOLS[*]}"
    echo "⏰ 時間尺度: ${FREQS[*]}"
    echo "📅 天數: $DAYS"
    echo "📁 輸出目錄: $OUTPUT_DIR"
    echo ""
    
    check_dependencies
    create_directories
    clean_old_data
    download_data
    validate_data
    generate_summary
}

# 腳本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi
