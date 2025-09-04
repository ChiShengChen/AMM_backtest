# Omnis Backtesting Suite

## 🎯 項目概述

Omnis Backtesting Suite 是一個綜合性的 AMM（自動化做市商）回測系統集合，專注於 Uniswap V3 風格的集中流動性做市商（CLMM）策略研究與優化。本套件包含兩個核心項目，提供從數據獲取到策略優化的完整解決方案。

## 🚀 核心項目

### 1. AMM Rebalance Backtester
**位置**: `amm-rebalance-backtester/`  
**專注**: 動態再平衡策略與參數優化

- **目標**: 證明動態策略優於固定策略（更高 APR、更低 MDD）
- **策略**: Baseline vs Dynamic 策略比較
- **優化**: Optuna 超參數優化
- **分析**: IL/LVR 與摩擦成本分析
- **數據**: 支持 ETH/USDC、BTC/USDC、USDC/USDT

### 2. Steer Intent Backtester
**位置**: `steer_intent_backtester/`  
**專注**: 基於意圖的動態再平衡策略

- **目標**: 生產級 CLMM 回測系統
- **策略**: 多種技術分析策略（布林帶、肯特納、唐奇安等）
- **數據源**: Binance、Kraken 多數據源支持
- **建模**: 真實的費用建模、滑點、Gas 成本
- **架構**: 模組化設計，易於擴展

## 📊 項目對比

| 特性 | AMM Rebalance Backtester | Steer Intent Backtester |
|------|---------------------------|-------------------------|
| **主要目標** | 動態 vs 固定策略比較 | 多策略技術分析回測 |
| **策略類型** | Baseline + Dynamic | 7種技術分析策略 |
| **優化框架** | Optuna 超參數優化 | 策略參數調優 |
| **數據源** | Binance API | Binance + Kraken |
| **時間尺度** | 1d, 1h, 1m | 1m, 15m, 1h, 4h, 1d |
| **分析深度** | IL/LVR 深度分析 | 完整性能指標 |
| **自動化** | 批量回測腳本 | CLI 工具 |
| **報告** | 7種圖表類型 | 5種圖表類型 |

## 🏗️ 系統架構

```
Omnis_bt/
├── amm-rebalance-backtester/          # 動態再平衡回測系統
│   ├── src/                          # 核心代碼
│   │   ├── core/                     # 回測引擎
│   │   ├── strategies/               # 策略實現
│   │   ├── opt/                      # 參數優化
│   │   ├── io/                       # 數據加載
│   │   └── reporting/                # 結果報告
│   ├── data/                         # 數據存儲
│   ├── results/                      # 回測結果
│   ├── reports/                      # 圖表和報告
│   ├── configs/                      # 配置文件
│   └── scripts/                      # 自動化腳本
└── steer_intent_backtester/          # 意圖回測系統
    ├── steerbt/                      # 核心模組
    │   ├── data/                     # 數據獲取
    │   ├── strategies/               # 策略實現
    │   ├── backtester.py             # 回測引擎
    │   └── reports.py                # 報告生成
    ├── data/                         # 數據存儲
    ├── reports/                      # 回測報告
    └── cli.py                        # 命令行界面
```

## 📥 數據獲取原理

### 統一數據源支持

| 數據源 | AMM Rebalance | Steer Intent | 描述 |
|--------|---------------|--------------|------|
| **Binance REST** | ✅ | ✅ | 主要數據源 |
| **Binance Vision** | ❌ | ✅ | 歷史數據批量下載 |
| **Kraken REST** | ❌ | ✅ | 備用數據源 |

### 支持的交易對

| 交易對 | 符號 | 兩個項目都支持 |
|--------|------|----------------|
| **ETH/USDC** | ETHUSDC | ✅ |
| **BTC/USDC** | BTCUSDC | ✅ |
| **USDC/USDT** | USDCUSDT | ✅ |

### 時間間隔支持

| 間隔 | AMM Rebalance | Steer Intent | 描述 |
|------|---------------|--------------|------|
| **1m** | ✅ | ✅ | 1 分鐘（高頻） |
| **15m** | ❌ | ✅ | 15 分鐘 |
| **1h** | ✅ | ✅ | 1 小時（推薦） |
| **4h** | ❌ | ✅ | 4 小時 |
| **1d** | ✅ | ✅ | 1 天（日線） |

## 🔄 回測引擎原理

### 1. AMM Rebalance Backtester

**核心特點**:
- **事件驅動**: 基於價格事件的策略觸發
- **參數優化**: Optuna 自動調參
- **深度分析**: IL/LVR 詳細計算
- **摩擦成本**: 完整的成本建模

**策略類型**:
```python
# Baseline 策略
- Baseline-Static: 被動超寬位置
- Baseline-Fixed: 固定寬度位置

# Dynamic 策略  
- Dynamic-Vol: 波動率自適應
- Dynamic-Inventory: 庫存偏差管理
```

### 2. Steer Intent Backtester

**核心特點**:
- **技術分析**: 基於技術指標的策略
- **多策略**: 7種不同的策略實現
- **真實建模**: 完整的 CLMM 估值
- **模組化**: 易於擴展的架構

**策略類型**:
```python
# 技術分析策略
- Bollinger Bands: SMA ± k×Std
- Keltner Channels: EMA ± m×ATR  
- Donchian Channels: HH/LL over N periods
- Classic: 經典再平衡策略
- Stable: 穩定錨定策略
- Fluid: 流體比例管理
- Channel Multiplier: 通道倍數策略
```

## 🎛️ 參數優化原理

### AMM Rebalance Backtester

**優化框架**: Optuna
```python
# 優化參數
{
    "k_width": (0.8, 2.0),           # 位置寬度調整係數
    "price_deviation_bps": (20, 120), # 價格偏差觸發閾值
    "rebalance_cooldown_hours": (6, 48) # 再平衡冷卻時間
}

# 優化目標: 最大化 APR
```

### Steer Intent Backtester

**優化方式**: 手動參數調優
```python
# 布林帶策略參數
{
    "n": 20,        # 回望期
    "k": 2.0        # 標準差倍數
}

# 肯特納策略參數
{
    "n": 20,        # EMA 週期
    "m": 2.0        # ATR 倍數
}
```

## 📈 結果生成原理

### 圖表類型對比

| 圖表類型 | AMM Rebalance | Steer Intent | 描述 |
|----------|---------------|--------------|------|
| **Equity Curves** | ✅ | ✅ | 淨值曲線比較 |
| **APR vs MDD Scatter** | ✅ | ❌ | 風險-收益散點圖 |
| **Fee vs Price PnL** | ✅ | ❌ | 費用與價格 PnL 分析 |
| **Sensitivity Heatmap** | ✅ | ❌ | 參數敏感性熱圖 |
| **Gas vs Frequency** | ✅ | ❌ | Gas 費用與頻率分析 |
| **IL Curve** | ✅ | ❌ | 無常損失曲線 |
| **LVR Estimates** | ✅ | ✅ | LVR 估算圖 |
| **Drawdown Curves** | ❌ | ✅ | 回撤分析 |
| **Fee Analysis** | ❌ | ✅ | 費用分析 |
| **Rebalance Frequency** | ❌ | ✅ | 再平衡頻率分析 |

### 報告格式

**AMM Rebalance Backtester**:
```
results/
├── {幣種}_{時間尺度}_{時間戳}/
│   ├── strategy_record_{時間戳}.json
│   ├── strategy_summary_{時間戳}.csv
│   └── strategy_report_{時間戳}.txt
└── INDEX.md
```

**Steer Intent Backtester**:
```
reports/
├── equity_curves_{run_id}.png
├── drawdown_curves_{run_id}.png
├── lvr_analysis_{run_id}.png
├── equity_curves_{run_id}.csv
└── summary_{run_id}.txt
```

## 🛠️ 使用方法

### 1. 環境設置

```bash
# 進入項目目錄
cd /Users/michael/Desktop/Omnis_bt

# 設置 AMM Rebalance Backtester
cd amm-rebalance-backtester
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 設置 Steer Intent Backtester
cd ../steer_intent_backtester
pip install -e .
```

### 2. 數據獲取

**AMM Rebalance Backtester**:
```bash
# 下載 5 年數據
./get_new_data.sh -s ETHUSDC,BTCUSDC -f 1d -d 1825

# 快速添加新幣種
./add_new_symbol.sh SOLUSDC 1d 365
```

**Steer Intent Backtester**:
```bash
# 基本數據獲取
python cli.py fetch \
  --source binance \
  --symbol ETHUSDC \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --out data/ETHUSDC_1h.csv
```

### 3. 運行回測

**AMM Rebalance Backtester**:
```bash
# 快速測試
python run.py quick --pool ETHUSDC --freq 1d

# 完整回測
python run.py full --pool ETHUSDC --freq 1d --study-name ethusdc_test --n-trials 20

# 批量回測
./run_all.sh
```

**Steer Intent Backtester**:
```bash
# 布林帶策略回測
python cli.py backtest \
  --pair ETHUSDC \
  --interval 1h \
  --strategy bollinger \
  --data-file data/ETHUSDC_1h.csv \
  --n 20 \
  --k 2.0 \
  --fee-bps 5 \
  --liq-share 0.002
```

### 4. 查看結果

**AMM Rebalance Backtester**:
```bash
# 查看圖表
ls reports/figs/ethusdc/

# 查看結果
ls results/ethusdc_1d_*/

# 查看目錄索引
cat results/INDEX.md
```

**Steer Intent Backtester**:
```bash
# 生成報告
python cli.py report --run-id <run_id>

# 查看策略信息
python cli.py strategies
```

## 🔧 配置說明

### 主要配置文件

**AMM Rebalance Backtester**:
```yaml
# configs/experiment_default.yaml
backtest:
  start_date: "2020-09-05"
  end_date: "2025-09-03"
  initial_capital: 10000

strategies:
  baseline_static:
    width_pct: 500.0
    rebalance_cooldown_hours: 168
  
  dynamic_vol:
    k_width: 1.5
    price_deviation_bps: 50.0
    rebalance_cooldown_hours: 24
```

**Steer Intent Backtester**:
```python
# 策略參數配置
{
    "pair": "ETHUSDC",
    "interval": "1h",
    "strategy": "bollinger",
    "initial_cash": 10000.0,
    "fee_bps": 5,
    "slippage_bps": 1,
    "gas_cost": 0.0,
    "liq_share": 0.002
}
```

## 📊 性能指標說明

### 共同指標

- **APR (Annual Percentage Rate)**: 年化收益率
- **MDD (Maximum Drawdown)**: 最大回撤
- **Sharpe Ratio**: 夏普比率
- **Calmar Ratio**: Calmar 比率
- **Volatility**: 年化波動率
- **IL (Impermanent Loss)**: 無常損失
- **LVR (Loss Versus Rebalancing)**: 相對於再平衡的損失

### AMM Rebalance 特有指標

- **參數敏感性**: 超參數對性能的影響
- **摩擦成本分析**: Gas、滑點、管理費的詳細分析
- **優化收斂性**: Optuna 優化過程的收斂分析

### Steer Intent 特有指標

- **再平衡頻率**: 策略觸發的頻率分析
- **費用分析**: 交易費用的詳細分解
- **技術指標有效性**: 各種技術指標的表現

## 🚨 注意事項

### 數據質量

- **完整性檢查**: 確保數據沒有缺失值
- **異常值處理**: 識別和處理異常價格數據
- **時間戳對齊**: 確保不同幣種的時間戳一致
- **數據新鮮度**: 定期更新數據以保持時效性

### 回測限制

- **前視偏差**: 避免使用未來信息
- **滑點建模**: 考慮真實的交易成本
- **流動性假設**: 假設足夠的流動性進行交易
- **市場衝擊**: 大額交易對市場的影響

### 策略注意

- **過擬合風險**: 避免過度優化歷史數據
- **樣本外測試**: 使用樣本外數據驗證策略
- **穩定性檢查**: 檢查參數的穩定性
- **風險管理**: 考慮極端市場條件

## 🔮 未來改進

### 功能擴展

- **更多策略**: 添加機器學習策略
- **實時回測**: 支持實時數據回測
- **多鏈支持**: 支持其他區塊鏈的 AMM
- **跨項目整合**: 統一兩個項目的功能

### 性能優化

- **並行計算**: 使用多進程加速回測
- **GPU 加速**: 利用 GPU 進行大規模計算
- **數據庫優化**: 使用數據庫存儲大量歷史數據
- **緩存機制**: 優化重複計算

### 用戶體驗

- **Web 界面**: 創建統一的 Web 儀表板
- **可視化增強**: 更多交互式圖表
- **報告自動化**: 自動生成 PDF 報告
- **API 接口**: 提供 REST API 接口

## 📞 支持與貢獻

### 問題報告

如果遇到問題，請：
1. 檢查日誌文件
2. 查看配置文件
3. 提交 Issue 到 GitHub

### 貢獻代碼

歡迎提交 Pull Request：
1. Fork 項目
2. 創建功能分支
3. 提交變更
4. 創建 Pull Request

### 聯繫方式

- **GitHub**: [項目地址]
- **Email**: [聯繫郵箱]
- **Discord**: [Discord 頻道]

---

## 📝 更新日誌

### v1.0.0 (2025-09-03)
- ✅ 初始版本發布
- ✅ AMM Rebalance Backtester 完整功能
- ✅ Steer Intent Backtester 完整功能
- ✅ 多數據源支持
- ✅ 完整的自動化腳本
- ✅ 詳細的文檔和示例

---

## 🎯 快速開始指南

### 新手推薦流程

1. **選擇項目**: 
   - 研究動態策略 → 使用 `amm-rebalance-backtester`
   - 技術分析策略 → 使用 `steer_intent_backtester`

2. **環境設置**: 按照上述環境設置步驟

3. **數據獲取**: 使用相應的數據獲取命令

4. **運行回測**: 從快速測試開始

5. **分析結果**: 查看生成的圖表和報告

6. **深入優化**: 根據結果調整參數

### 進階用戶

- 可以同時使用兩個項目進行對比分析
- 利用 AMM Rebalance 的優化功能找到最佳參數
- 使用 Steer Intent 的多策略功能進行策略組合

---

**🎉 感謝使用 Omnis Backtesting Suite！**

這是一個強大的 AMM 回測工具集，為您的研究和交易策略開發提供全面的支持。
