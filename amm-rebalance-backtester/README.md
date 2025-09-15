# AMM Dynamic Rebalancing Backtesting System

> **🚀 One-Command Backtesting**: `./run_single.sh BTCUSDC 1d 50`

## 🎯 Project Overview

This is a Python system specifically designed for AMM (Automated Market Maker, primarily targeting Uniswap V3) dynamic rebalancing strategy backtesting. The system can compare the performance of fixed strategies vs dynamic strategies, analyze IL/LVR and friction costs, and find optimal strategy configurations through parameter optimization.

## 🚀 Core Features

- **Multi-Strategy Backtesting**: Supports Baseline and Dynamic strategy comparison
- **Parameter Optimization**: Uses Optuna for hyperparameter optimization
- **Multi-Asset Support**: ETH/USDC, BTC/USDC, USDC/USDT, etc.
- **Multi-Timeframe**: Supports daily(1d), hourly(1h), minute(1m) data
- **Complete Analysis**: IL/LVR analysis, friction cost modeling, risk metrics calculation
- **Automation Scripts**: Data acquisition, backtest execution, result organization

## 📊 Strategy Types

### Baseline Strategies
- **Baseline-Static**: Passive ultra-wide position strategy, minimal rebalancing
- **Baseline-Fixed**: Fixed width position, fixed price deviation trigger

### Dynamic Strategies
- **Dynamic-Vol**: Volatility-adaptive width + price deviation trigger
- **Dynamic-Inventory**: Inventory deviation + fee density trigger + low-frequency reinvestment

## 🏗️ System Architecture

```
amm-rebalance-backtester/
├── src/                    # Core code
│   ├── core/              # Backtesting engine, mathematical calculations
│   ├── strategies/        # Strategy implementations
│   ├── io/                # Data loading and validation
│   ├── opt/               # Parameter optimization
│   └── reporting/         # Result reporting and charts
├── data/                  # Data storage
├── results/               # Backtesting results
├── reports/               # Charts and reports
├── configs/               # Configuration files
└── scripts/               # Automation scripts
```

## 📥 Data Acquisition Principles

### 1. Data Sources
The system uses **Binance REST API** to fetch historical price data, supporting:
- **K-line Data**: OHLCV (Open, High, Low, Close, Volume)
- **Time Range**: Customizable, default 5-year daily data
- **Trading Pairs**: Supports all Binance trading pairs

### 2. Data Acquisition Process

```bash
# Use automation scripts to download data
./get_new_data.sh -s ETHUSDC,BTCUSDC -f 1d,1h -d 1825

# Or quickly add new trading pairs
./add_new_symbol.sh SOLUSDC 1d 365
```

**Internal Implementation**:
1. **API Calls**: Uses `requests` library to call Binance API
2. **Pagination Handling**: Automatically handles API limits, fetches data in batches
3. **Data Cleaning**: Handles missing values, outliers, timestamp alignment
4. **Format Standardization**: Converts to system standard CSV format
5. **Directory Organization**: Automatically categorizes by trading pair and timeframe

### 3. Data Structure
```csv
timestamp,open,high,low,close,volume
2020-09-05,335.22,394.61,309.57,335.22,1234567
2020-09-06,334.89,360.32,316.02,352.84,2345678
...
```

## 🔄 Backtesting Engine Principles

### 1. Backtesting Process

```python
# Core backtesting process
BacktestEngine.run_full_evaluation() -> Dict[str, Any]
├── 1. Data loading and preprocessing
├── 2. Strategy initialization
├── 3. Event-driven simulation
├── 4. Performance calculation
└── 5. Result output
```

### 2. Strategy Execution Mechanism

**Event-Driven Architecture**:
- **Price Events**: OHLCV data at each time point
- **Rebalancing Triggers**: Condition-based judgment from strategy logic
- **Trade Execution**: Simulates real AMM operations
- **Cost Calculation**: Includes gas fees, slippage, management fees

**Strategy Logic Example**:
```python
# Dynamic-Vol strategy core logic
def calculate_ranges(self, price_data, current_price, portfolio_value):
    # 1. Calculate volatility
    volatility = price_data['returns'].rolling(30).std()
    
    # 2. Dynamically adjust position width
    vol_adjustment = 1.5 - volatility * 10
    
    # 3. Apply price deviation trigger
    if abs(price_change) > self.price_deviation_bps:
        return self._rebalance_positions()
    
    return current_ranges
```

### 3. Performance Metrics Calculation

**Return Metrics**:
- **APR**: Annualized return rate, based on cumulative returns calculation
- **MDD**: Maximum drawdown, using rolling maximum calculation
- **Sharpe**: Sharpe ratio, risk-adjusted returns
- **Calmar**: Calmar ratio, return to drawdown ratio

**Risk Metrics**:
- **IL (Impermanent Loss)**: Impermanent loss calculation
- **LVR (Loss Versus Rebalancing)**: Loss relative to rebalancing
- **Volatility**: Daily return standard deviation

## 🎛️ Parameter Optimization Principles

### 1. Optuna Optimization Framework

```python
# Optimization objective function
def _objective(self, trial):
    # 1. Parameter suggestions
    k_width = trial.suggest_float('k_width', 0.8, 2.0)
    price_deviation_bps = trial.suggest_float('price_deviation_bps', 20, 120)
    rebalance_cooldown_hours = trial.suggest_int('rebalance_cooldown_hours', 6, 48)
    
    # 2. Strategy backtesting
    strategy = DynamicVolatilityStrategy(k_width, price_deviation_bps, rebalance_cooldown_hours)
    results = self._run_backtest(strategy)
    
    # 3. Return optimization objective (maximize APR)
    return results['apr']
```

### 2. Optimization Parameters

**Core Parameters**:
- **k_width**: Position width adjustment coefficient (0.8-2.0)
- **price_deviation_bps**: Price deviation trigger threshold (20-120 bps)
- **rebalance_cooldown_hours**: Rebalancing cooldown time (6-48 hours)

**Optimization Strategy**:
- **Direction**: Maximize APR
- **Trial Count**: Configurable (recommended 20-100 trials)
- **Search Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Early Stopping**: Supports early termination of low-quality trials

### 3. Optimization Results

```bash
# Run optimization
python run.py full --pool ETHUSDC --freq 1d --study-name ethusdc_optimization --n-trials 20

# Optimization results example
Best trial: 15
Best value: 12.46
Best params: {
    'k_width': 1.91,
    'price_deviation_bps': 32.34,
    'rebalance_cooldown_hours': 41
}
```

## 📈 Result Generation Principles

### 1. Chart Generation

**Supported Chart Types**:
1. **Equity Curves**: Equity curve comparison
2. **APR vs MDD Scatter**: Risk-return scatter plot
3. **Fee vs Price PnL**: Fee vs price PnL analysis
4. **Sensitivity Heatmap**: Parameter sensitivity heatmap
5. **Gas vs Frequency Contour**: Gas cost vs frequency analysis
6. **IL Curve**: Impermanent loss curve
7. **LVR Estimates**: LVR estimation charts

**Chart Generation Process**:
```python
# Chart generator
class PlotGenerator:
    def plot_equity_curves(self, results, save_path):
        # 1. Data preparation
        strategies = self._extract_strategy_data(results)
        
        # 2. Chart creation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 3. Data plotting
        for strategy_name, strategy_data in strategies.items():
            equity_curve = self._calculate_equity_curve(strategy_data)
            ax.plot(equity_curve, label=strategy_name)
        
        # 4. Style settings
        ax.set_title(f'Equity Curves - {self.pool}')
        ax.legend()
        
        # 5. Save chart
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

### 2. Report Generation

**Strategy Records**:
- **JSON Format**: Complete strategy parameters and performance data
- **CSV Summary**: Key metrics in tabular format
- **Text Report**: Human-readable strategy description

**Directory Organization**:
```
results/
├── {trading_pair}_{timeframe}_{timestamp}/
│   ├── strategy_record_{timestamp}.json
│   ├── strategy_summary_{timestamp}.csv
│   └── strategy_report_{timestamp}.txt
├── common/                    # Common files
└── INDEX.md                  # Directory index
```

## 🚀 Quick Start

### One-Command Backtesting

```bash
# Most recommended: Use script for backtesting
./run_single.sh BTCUSDC 1d 50

# Direct command backtesting
python run.py full --pool BTCUSDC --freq 1d --n-trials 50
```

**Parameter Description**:
- `BTCUSDC`: Replace with your trading pair code
- `1d`: Data frequency (1d=daily, 1h=hourly)  
- `50`: Optimization trial count

**Result Locations**:
- 📊 Charts: `reports/figs/btcusdc/`
- 📈 Data: `results/strategy_*_*.json`
- 📋 Reports: `results/strategy_report_*.txt`

### 🔗 Integration with Steer Intent Backtester Strategies

```bash
# Compare AMM and Steer strategies
./run_steer_comparison.sh BTCUSDC 1d

# Or run directly
python final_integration.py
```

**Integration Features**:
- 🔄 Simultaneously test AMM and Steer strategies
- 📊 Generate strategy comparison reports
- 🏆 Find optimal strategy combinations
- 📈 Support multiple strategy type comparisons

### 📁 Data File Structure Setup

Expected data file structure:
```
data/
├── BTCUSDC/
│   └── price_1d.csv
├── ETHUSDC/
│   └── price_1d.csv
└── USDCUSDT/
    └── price_1d.csv
```

**If you have 5-year data files** (e.g., `data/5year_daily/ETHUSDC_1d_20200905_20250903.csv`):

```bash
# Create directory structure
mkdir -p data/ETHUSDC data/USDCUSDT

# Create symbolic links (recommended)
ln -sf ../5year_daily/ETHUSDC_1d_20200905_20250903.csv data/ETHUSDC/price_1d.csv
ln -sf ../5year_daily/USDCUSDT_1d_20200905_20250903.csv data/USDCUSDT/price_1d.csv

# Or copy files
cp data/5year_daily/ETHUSDC_1d_20200905_20250903.csv data/ETHUSDC/price_1d.csv
cp data/5year_daily/USDCUSDT_1d_20200905_20250903.csv data/USDCUSDT/price_1d.csv
```

### 📊 Backtesting Results Example

**USDCUSDT 5-Year Backtesting Results** (2020-2025):
```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Strategy          ┃ APR (%) ┃ MDD (%) ┃ Sharpe ┃ Calmar ┃ Rebalances ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ Baseline-Static   │ -0.00   │ 0.90    │ -0.00  │ -0.01  │ 2          │
│ Baseline-Fixed    │ -0.00   │ 1.30    │ -0.00  │ -0.01  │ 15         │
│ Dynamic-Vol       │ -0.00   │ 1.60    │ -0.00  │ -0.01  │ 28         │
│ Dynamic-Inventory │ -0.00   │ 0.30    │ -0.00  │ -0.00  │ 10         │
└───────────────────┴─────────┴─────────┴────────┴────────┴────────────┘
```

**Best Parameters**:
- K width multiplier: 1.96
- Price deviation threshold: 113.85 bps
- Rebalancing cooldown time: 41 hours

---

## 🛠️ 使用方法

### 1. 環境設置

```bash
# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 數據獲取

```bash
# 下載默認數據
./get_new_data.sh

# 添加新幣種
./add_new_symbol.sh SOLUSDC

# 自定義下載
./get_new_data.sh -s ETHUSDC,BTCUSDC -f 1d,1h -d 365
```

### 3. 運行回測

#### 🚀 一行命令快速回測

```bash
# 最簡單的一行命令 (推薦)
./run_single.sh BTCUSDC 1d 50

# 直接使用 Python 命令
python run.py full --pool BTCUSDC --freq 1d --n-trials 50

# 完整參數版本
python run.py full --pool BTCUSDC --freq 1d --study-name "BTCUSDC_$(date +%Y%m%d_%H%M%S)" --n-trials 50 --config configs/btcusdc_experiment.yaml
```

#### 📋 參數說明
- `BTCUSDC`: 幣種代碼 (可替換為 ETHUSDC, USDCUSDT 等)
- `1d`: 數據頻率 (1d=日線, 1h=小時線)
- `50`: 優化試驗次數 (建議 20-100)

#### ⚠️ 重要注意事項
1. **數據文件必須存在**: 確保 `data/{POOL_NAME}/price_{frequency}.csv` 文件存在
2. **頻率匹配**: 數據頻率必須與 `--freq` 參數匹配
3. **study-name**: 如果使用直接Python命令，必須提供 `--study-name` 參數

#### 🔧 其他回測方式

```bash
# 快速測試
python run.py quick --pool ETHUSDC --freq 1d

# 完整回測
python run.py full --pool ETHUSDC --freq 1d --study-name ethusdc_test --n-trials 10

# 批量回測
./run_all.sh
```

### 4. 查看結果

```bash
# 查看圖表
ls reports/figs/ethusdc/

# 查看結果
ls results/ethusdc_1d_*/

# 查看目錄索引
cat results/INDEX.md
```

## 🔧 配置說明

### 1. 主要配置文件

**`configs/experiment_default.yaml`**:
```yaml
# 回測配置
backtest:
  start_date: "2020-09-05"
  end_date: "2025-09-03"
  initial_capital: 10000
  
# 策略配置
strategies:
  baseline_static:
    width_pct: 500.0
    rebalance_cooldown_hours: 168
  
  dynamic_vol:
    k_width: 1.5
    price_deviation_bps: 50.0
    rebalance_cooldown_hours: 24

# 優化配置
wfa:
  n_trials: 20
  study_name: "default_optimization"
```

### 2. 環境變量

```bash
# 數據目錄
export DATA_DIR="data"

# 結果目錄
export RESULTS_DIR="results"

# 日誌級別
export LOG_LEVEL="INFO"
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

## 🚨 注意事項

### 1. 數據質量

- **完整性檢查**: 確保數據沒有缺失值
- **異常值處理**: 識別和處理異常價格數據
- **時間戳對齊**: 確保不同幣種的時間戳一致

### 2. 回測限制

- **前視偏差**: 避免使用未來信息
- **滑點建模**: 考慮真實的交易成本
- **流動性假設**: 假設足夠的流動性進行交易

### 3. 優化注意

- **過擬合風險**: 避免過度優化歷史數據
- **樣本外測試**: 使用樣本外數據驗證策略
- **穩定性檢查**: 檢查參數的穩定性

## 🔮 未來改進

### 1. 功能擴展

- **更多策略**: 添加機器學習策略
- **實時回測**: 支持實時數據回測
- **多鏈支持**: 支持其他區塊鏈的 AMM

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
- ✅ 支持 ETH/USDC、BTC/USDC、USDC/USDT
- ✅ 實現 4 種策略
- ✅ 完整的參數優化框架
- ✅ 自動化腳本和工具

---

**🎉 感謝使用 AMM 動態再平衡回測系統！**
