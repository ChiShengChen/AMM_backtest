# Steer Intent Backtester

## 🎯 項目概述

Steer Intent Backtester 是一個生產級的 CLMM（集中流動性做市商）回測系統，專門用於評估「基於意圖的動態再平衡」策略與傳統方法的比較。系統實現了多種流動性管理策略，適用於 Uniswap V3 風格的 CLMM，無需依賴外部 API，提供完整的回測功能，包括真實的費用建模、滑點和 Gas 成本。

## 🚀 核心功能

- **多策略支持**: 經典再平衡、布林帶、肯特納通道、唐奇安通道、穩定錨定、流體比例管理
- **多數據源**: Binance REST API、Binance Vision S3、Kraken REST API
- **真實建模**: CLMM 位置估值、費用累積、滑點、Gas 成本
- **完整指標**: APR、最大回撤、夏普比率、無常損失、LVR 代理
- **可擴展架構**: 模組化設計，易於添加策略和數據源

## 📊 策略類型

### 1. 經典策略 (Classic)
- **寬度模式**: 百分比、倍數或靜態 tick 基礎
- **觸發條件**: 中心偏離、範圍不活躍、百分比漂移、單向退出、時間流逝
- **曲線類型**: 線性、高斯、Sigmoid、對數、買賣價差

### 2. 通道倍數 (Channel Multiplier)
- 圍繞當前價格的單一對稱百分比寬度，一個 LP 位置

### 3. 布林帶 (Bollinger Bands)
- 公式: `Bands = SMA(n) ± k × Std(n)`
- n: 回望期，k: 標準差倍數

### 4. 肯特納通道 (Keltner Channels)
- 公式: `Bands = EMA(n) ± m × ATR(n)`
- n: EMA 週期，m: ATR 倍數

### 5. 唐奇安通道 (Donchian Channels)
- 上軌: N 期內最高價，下軌: N 期內最低價
- 可選寬度倍數

### 6. 穩定策略 (Stable)
- 圍繞計算「錨定」的多位置策略
- 使用高斯或線性曲線，可配置 bin 數量

### 7. 流體策略 (Fluid)
- 維持價值比例朝向 ideal_ratio
- 三種狀態：默認/不平衡/單邊
- 位置類型：默認/限價/擴展

## 🏗️ 系統架構

```
steer_intent_backtester/
├── steerbt/                    # 核心模組
│   ├── __init__.py
│   ├── data/                   # 數據獲取模組
│   │   ├── binance.py         # Binance API 接口
│   │   └── kraken.py          # Kraken API 接口
│   ├── uv3_math.py            # CLMM 位置估值
│   ├── portfolio.py           # 投資組合會計
│   ├── triggers.py            # 再平衡觸發器
│   ├── curves.py              # 流動性分佈曲線
│   ├── strategies/            # 策略實現
│   │   ├── base.py           # 基礎策略類
│   │   ├── classic.py        # 經典策略
│   │   ├── bollinger.py      # 布林帶策略
│   │   ├── keltner.py        # 肯特納策略
│   │   └── ...               # 其他策略
│   ├── backtester.py         # 主回測引擎
│   ├── metrics.py            # 性能指標
│   └── reports.py            # 圖表和導出
├── cli.py                     # 命令行界面
├── data/                      # 數據存儲
├── reports/                   # 回測報告
└── tests/                     # 測試文件
```

## 📥 數據獲取原理

### 1. 數據源支持

| 數據源 | API 端點 | 速率限制 | 適用場景 |
|--------|----------|----------|----------|
| **Binance REST** | `/api/v3/klines` | 1200 請求/分鐘 | 主要數據源，實時數據 |
| **Binance Vision** | S3 存儲桶 | 無限制 | 歷史數據，批量下載 |
| **Kraken REST** | `/0/public/OHLC` | 15 請求/10秒 | 備用，替代交易對 |

### 2. 支持的交易對

| 交易對 | Binance 符號 | Kraken 符號 | 描述 |
|--------|--------------|-------------|------|
| **USDCUSDT** | USDCUSDT | USDC/USDT | 穩定幣對 |
| **ETHUSDC** | ETHUSDC | ETH/USDC | 以太坊 vs USDC |
| **BTCUSDC** | BTCUSDC | XBT/USDC | 比特幣 vs USDC |
| **ETHUSDT** | ETHUSDT | ETH/USDT | 以太坊 vs USDT |
| **BTCUSDT** | BTCUSDT | XBT/USDT | 比特幣 vs USDT |

### 3. 支持的時間間隔

| 間隔 | Binance | Kraken | 描述 |
|------|---------|--------|------|
| **1m** | ✅ | ✅ | 1 分鐘（高頻） |
| **15m** | ✅ | ✅ | 15 分鐘 |
| **1h** | ✅ | ✅ | 1 小時（推薦） |
| **4h** | ✅ | ✅ | 4 小時 |
| **1d** | ✅ | ✅ | 1 天（日線分析） |

### 4. 數據獲取流程

```python
# 數據獲取核心流程
class BinanceDataFetcher:
    def fetch_klines(self, symbol, interval, start_date, end_date, limit=1000):
        # 1. API 調用
        # 2. 分頁處理
        # 3. 數據清洗
        # 4. 格式標準化
        # 5. 返回 DataFrame
```

**內部實現**:
1. **API 調用**: 使用 `requests` 庫調用交易所 API
2. **分頁處理**: 自動處理 API 限制，分批獲取數據
3. **數據清洗**: 處理缺失值、異常值、時間戳對齊
4. **格式標準化**: 轉換為系統標準 DataFrame 格式
5. **錯誤處理**: 重試機制、速率限制處理

### 5. 數據結構

```csv
timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote
2024-01-01T00:00:00Z,2500.00,2550.00,2480.00,2520.00,1234567,2024-01-01T00:59:59Z,3100000000,15000,600000,1500000000
```

## 🔄 回測引擎原理

### 1. 回測流程

```python
# 核心回測流程
Backtester.run() -> Dict[str, Any]
├── 1. 數據加載和預處理
├── 2. 策略初始化
├── 3. 事件驅動模擬
├── 4. 性能計算
└── 5. 結果輸出
```

### 2. 策略執行機制

**事件驅動架構**:
- **價格事件**: 每個時間點的 OHLCV 數據
- **再平衡觸發**: 基於策略邏輯的條件判斷
- **交易執行**: 模擬真實的 CLMM 操作
- **費用計算**: 包括 Gas 費用、滑點、管理費

**策略邏輯示例**:
```python
# 布林帶策略核心邏輯
class BollingerStrategy(BaseStrategy):
    def calculate_range(self, price_data, current_price, portfolio_value):
        # 1. 計算布林帶
        sma = self._calculate_sma(price_data['close'], self.n)
        std = self._calculate_std(price_data['close'], self.n)
        
        upper_band = sma + (self.k * std)
        lower_band = sma - (self.k * std)
        
        # 2. 創建位置範圍
        ranges = [(lower_band, upper_band)]
        liquidities = [portfolio_value / 2]  # 50% 流動性
        
        return ranges, liquidities
```

### 3. CLMM 估值原理

**Uniswap V3 公式**:
- **Amount 計算**: 給定價格 P 和範圍 [Pa, Pb]，計算 amount0 和 amount1
- **投資組合價值**: V = amount0 × P + amount1 + accumulated_fees
- **費用累積**: 每根 K 線的費用 ≈ pool_fee_bps × quote_volume_in_range × liquidity_share

```python
# CLMM 位置估值
def calculate_position_value(price, lower_tick, upper_tick, liquidity):
    # 計算 amount0 和 amount1
    amount0, amount1 = calculate_amounts(price, lower_tick, upper_tick, liquidity)
    
    # 計算總價值
    total_value = amount0 * price + amount1
    
    return total_value, amount0, amount1
```

### 4. 性能指標計算

**收益率指標**:
- **APR**: 年化收益率，基於累積收益計算
- **MDD**: 最大回撤，使用滾動最大值計算
- **Sharpe**: 夏普比率，風險調整後收益
- **Calmar**: Calmar 比率，收益與回撤比

**風險指標**:
- **IL (Impermanent Loss)**: 無常損失計算
- **LVR (Loss Versus Rebalancing)**: 相對於再平衡的損失
- **波動率**: 日收益率標準差

## 🎛️ 成本建模原理

### 1. 費用結構

**交易費用**:
- **Swap 費用**: 用戶定義的基點
- **再平衡費用**: 每次再平衡操作的 Gas 成本
- **滑點**: 對執行價格的影響

**費用計算**:
```python
# 費用計算示例
def calculate_trading_costs(amount, fee_bps, slippage_bps):
    fee_cost = amount * (fee_bps / 10000)
    slippage_cost = amount * (slippage_bps / 10000)
    total_cost = fee_cost + slippage_cost
    
    return total_cost
```

### 2. Gas 成本建模

```python
# Gas 成本計算
def calculate_gas_cost(operation_type, gas_price_gwei):
    gas_limits = {
        'mint': 200000,
        'burn': 150000,
        'swap': 100000
    }
    
    gas_cost_eth = (gas_limits[operation_type] * gas_price_gwei) / 1e9
    gas_cost_usd = gas_cost_eth * eth_price_usd
    
    return gas_cost_usd
```

## 📈 結果生成原理

### 1. 圖表生成

**支持的圖表類型**:
1. **Equity Curves**: 淨值曲線比較
2. **Drawdown Curves**: 回撤分析
3. **LVR Analysis**: LVR 代理圖表
4. **Fee Analysis**: 費用分析
5. **Rebalance Frequency**: 再平衡頻率分析

**圖表生成流程**:
```python
# 圖表生成器
class ReportGenerator:
    def generate_equity_curves(self, results, save_path):
        # 1. 數據準備
        strategy_equity = results['equity_curves']['strategy']
        baseline_equity = results['equity_curves']['hodl_50_50']
        
        # 2. 圖表創建
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 3. 數據繪製
        ax.plot(strategy_equity['timestamp'], strategy_equity['total_value'], 
                label='Strategy', linewidth=2)
        ax.plot(baseline_equity['timestamp'], baseline_equity['total_value'], 
                label='HODL 50:50', linewidth=2)
        
        # 4. 樣式設置
        ax.set_title(f'Equity Curves - {self.pair}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 保存圖表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

### 2. 報告生成

**策略記錄**:
- **JSON 格式**: 完整的策略參數和性能數據
- **CSV 摘要**: 關鍵指標的表格形式
- **文本報告**: 人類可讀的策略說明

**目錄組織**:
```
reports/
├── equity_curves_{run_id}.png      # 淨值曲線圖
├── drawdown_curves_{run_id}.png    # 回撤分析圖
├── lvr_analysis_{run_id}.png       # LVR 分析圖
├── equity_curves_{run_id}.csv      # 詳細 CSV 數據
└── summary_{run_id}.txt            # 摘要報告
```

## 🛠️ 使用方法

### 1. 環境設置

```bash
# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -e .
```

### 2. 數據獲取

```bash
# 基本數據獲取
python cli.py fetch \
  --source binance \
  --symbol ETHUSDC \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --out data/ETHUSDC_1h.csv

# 批量數據獲取
for pair in ETHUSDC BTCUSDC USDCUSDT; do
  python cli.py fetch \
    --source binance \
    --symbol $pair \
    --interval 1h \
    --start 2024-01-01 \
    --end 2024-01-31 \
    --out data/${pair}_1h.csv
done
```

### 3. 運行回測

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

# 肯特納策略回測
python cli.py backtest \
  --pair ETHUSDC \
  --interval 1h \
  --strategy keltner \
  --data-file data/ETHUSDC_1h.csv \
  --n 20 \
  --m 2.0 \
  --fee-bps 5 \
  --liq-share 0.002
```

### 4. 生成報告

```bash
# 生成報告（使用回測返回的 run_id）
python cli.py report --run-id <run_id>
```

### 5. 查看策略信息

```bash
# 列出可用策略
python cli.py strategies

# 列出可用曲線
python cli.py curves
```

## 🔧 配置說明

### 1. 策略參數

**布林帶策略**:
```python
{
    "n": 20,        # 回望期
    "k": 2.0        # 標準差倍數
}
```

**肯特納策略**:
```python
{
    "n": 20,        # EMA 週期
    "m": 2.0        # ATR 倍數
}
```

**經典策略**:
```python
{
    "width_mode": "percent",     # 寬度模式
    "width_value": 10.0,         # 寬度值
    "placement_mode": "center",  # 放置模式
    "curve_type": "uniform"      # 曲線類型
}
```

### 2. 回測配置

```python
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

### 1. 收益率指標

- **APR (Annual Percentage Rate)**: 年化收益率
- **MDD (Maximum Drawdown)**: 最大回撤
- **Sharpe Ratio**: 夏普比率，風險調整後收益
- **Calmar Ratio**: Calmar 比率，收益與回撤比

### 2. 風險指標

- **Volatility**: 年化波動率
- **VaR (Value at Risk)**: 風險價值
- **CVaR (Conditional VaR)**: 條件風險價值

### 3. 交易指標

- **Rebalance Count**: 再平衡次數
- **Average Gas Cost**: 平均 Gas 費用
- **Slippage**: 滑點損失
- **IL (Impermanent Loss)**: 無常損失

### 4. LVR 代理計算

LVR (Loss-Versus-Rebalancing) 代理計算公式：
```
LVR_t ≈ V_rebal50:50(t) - V_CLMM_no_fee(t)
```

這衡量了維持 CLMM 位置相對於簡單再平衡的機會成本。

## 🚨 注意事項

### 1. 數據質量

- **完整性檢查**: 確保數據沒有缺失值
- **異常值處理**: 識別和處理異常價格數據
- **時間戳對齊**: 確保不同幣種的時間戳一致

### 2. 回測限制

- **前視偏差**: 避免使用未來信息
- **滑點建模**: 考慮真實的交易成本
- **流動性假設**: 假設足夠的流動性進行交易

### 3. 策略注意

- **過擬合風險**: 避免過度優化歷史數據
- **樣本外測試**: 使用樣本外數據驗證策略
- **穩定性檢查**: 檢查參數的穩定性

## 🔮 未來改進

### 1. 功能擴展

- **更多策略**: 添加機器學習策略
- **實時回測**: 支持實時數據回測
- **多鏈支持**: 支持其他區塊鏈的 CLMM

### 2. 性能優化

- **並行計算**: 使用多進程加速回測
- **GPU 加速**: 利用 GPU 進行大規模計算
- **數據庫優化**: 使用數據庫存儲大量歷史數據

### 3. 用戶體驗

- **Web 界面**: 創建 Web 儀表板
- **可視化增強**: 更多交互式圖表
- **報告自動化**: 自動生成 PDF 報告

## 📞 支持與貢獻

### 1. 問題報告

如果遇到問題，請：
1. 檢查日誌文件
2. 查看配置文件
3. 提交 Issue 到 GitHub

### 2. 貢獻代碼

歡迎提交 Pull Request：
1. Fork 項目
2. 創建功能分支
3. 提交變更
4. 創建 Pull Request

### 3. 聯繫方式

- **GitHub**: [項目地址]
- **Email**: [聯繫郵箱]
- **Discord**: [Discord 頻道]

---

## 📝 更新日誌

### v1.0.0 (2025-09-03)
- ✅ 初始版本發布
- ✅ 支持多種策略（布林帶、肯特納、唐奇安等）
- ✅ 多數據源支持（Binance、Kraken）
- ✅ 完整的 CLMM 估值和費用建模
- ✅ 自動化腳本和工具

---

**🎉 感謝使用 Steer Intent Backtester！**