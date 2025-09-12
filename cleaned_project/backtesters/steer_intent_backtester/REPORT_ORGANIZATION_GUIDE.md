# 報告組織化指南

## 概述

Steer Intent Backtester 現在使用新的組織化報告結構，每次實驗的結果都會整齊地存放在各自的資料夾中，不再散落在各處。

## 新的資料夾結構

每次回測實驗現在會創建以下結構：

```
reports/
└── {PAIR}_{STRATEGY}_{INTERVAL}_{DATE_RANGE}_{RUN_ID}/
    ├── figs/                           # 圖表文件
    │   ├── equity_curves_{run_id}.png
    │   ├── drawdown_curves_{run_id}.png
    │   └── lvr_analysis_{run_id}.png
    ├── data/                           # 數據文件
    │   └── equity_curves_{run_id}.csv
    ├── logs/                           # 報告和配置
    │   ├── summary_report_{run_id}.txt
    │   └── experiment_config_{run_id}.json
    └── index_{run_id}.html             # 實驗索引頁面
```

## 實驗命名規則

實驗資料夾名稱格式：`{PAIR}_{STRATEGY}_{INTERVAL}_{DATE_RANGE}_{RUN_ID}`

例如：`ETHUSDC_bollinger_1h_20240101_20240131_a1b2c3d4`

## 使用方式

### 1. 執行回測

```bash
# 執行回測（會自動生成組織化的報告）
python cli.py backtest --pair ETHUSDC --interval 1h --strategy bollinger --n 20 --k 2
```

### 2. 查看所有實驗

```bash
# 列出所有已完成的實驗
python cli.py list-experiments
```

### 3. 查看特定實驗

每個實驗資料夾都包含一個 `index_{run_id}.html` 文件，可以在瀏覽器中打開查看：
- 性能摘要
- 所有生成的文件連結
- 實驗詳細信息

### 4. 清理舊的散落文件

如果您有舊的散落報告文件，可以使用清理工具：

```bash
# 掃描散落的文件
python cleanup_reports.py scan

# 預覽組織化操作（不會實際移動文件）
python cleanup_reports.py organize --dry-run

# 實際組織化文件
python cleanup_reports.py organize

# 完整清理（包含備份）
python cleanup_reports.py cleanup
```

## 文件類型說明

### figs/ 資料夾
- **equity_curves_{run_id}.png**: 權益曲線圖表
- **drawdown_curves_{run_id}.png**: 回撤曲線圖表  
- **lvr_analysis_{run_id}.png**: LVR 分析圖表

### data/ 資料夾
- **equity_curves_{run_id}.csv**: 權益曲線數據（CSV格式）

### logs/ 資料夾
- **summary_report_{run_id}.txt**: 文字摘要報告
- **experiment_config_{run_id}.json**: 實驗配置和元數據

### 根目錄
- **index_{run_id}.html**: 實驗索引頁面，包含所有文件的連結和性能摘要

## 優勢

1. **組織化**: 每次實驗的所有文件都在一個資料夾中
2. **易於管理**: 可以輕鬆找到和比較不同實驗的結果
3. **可追溯性**: 每個實驗都有完整的配置記錄
4. **視覺化**: HTML 索引頁面提供友好的瀏覽體驗
5. **備份友好**: 可以輕鬆備份或移動整個實驗資料夾

## 遷移舊文件

如果您有舊的散落報告文件，建議使用清理工具進行組織化：

1. 首先掃描現有文件：`python cleanup_reports.py scan`
2. 預覽組織化操作：`python cleanup_reports.py organize --dry-run`
3. 執行組織化：`python cleanup_reports.py organize`
4. 或者使用完整清理：`python cleanup_reports.py cleanup`

清理工具會：
- 自動識別文件中的 run ID
- 按 run ID 分組文件
- 創建適當的資料夾結構
- 移動文件到正確位置
- 創建備份（可選）
- 清理空資料夾

## 注意事項

- 新的報告結構向後兼容，舊的 CLI 命令仍然有效
- 清理工具會創建備份，確保數據安全
- 建議定期清理舊的散落文件以保持報告目錄整潔
- HTML 索引頁面可以在任何現代瀏覽器中打開
