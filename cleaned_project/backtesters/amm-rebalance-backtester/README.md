# AMM 動態再平衡回測系統

> **🚀 一行命令回測**: `./run_single.sh BTCUSDC 1d 50`

## 🎯 項目概述

這是一個專門用於 AMM（自動化做市商，主要針對 Uniswap V3）動態再平衡策略回測的 Python 系統。系統能夠比較固定策略與動態策略的表現，分析 IL/LVR 和摩擦成本，並通過參數優化找到最佳策略配置。

## 🚀 核心功能

- **多策略回測**: 支持 Baseline 和 Dynamic 策略比較
- **參數優化**: 使用 Optuna 進行超參數優化
- **多幣種支持**: ETH/USDC、BTC/USDC、USDC/USDT 等
- **多時間尺度**: 支持日線(1d)、小時線(1h)、分鐘線(1m)
- **完整分析**: IL/LVR 分析、摩擦成本建模、風險指標計算
- **自動化腳本**: 數據獲取、回測執行、結果整理

## 📊 策略類型

### Baseline 策略
- **Baseline-Static**: 被動超寬位置策略，最小再平衡
- **Baseline-Fixed**: 固定寬度位置，固定價格偏差觸發

### Dynamic 策略
- **Dynamic-Vol**: 波動率自適應寬度 + 價格偏差觸發
- **Dynamic-Inventory**: 庫存偏差 + 費用密度觸發 + 低頻再投資

## 🏗️ 系統架構

```
amm-rebalance-backtester/
├── src/                    # 核心代碼
│   ├── core/              # 回測引擎、數學計算
│   ├── strategies/        # 策略實現
│   ├── io/                # 數據加載和驗證
│   ├── opt/               # 參數優化
│   └── reporting/         # 結果報告和圖表
├── data/                  # 數據存儲
├── results/               # 回測結果
├── reports/               # 圖表和報告
├── configs/               # 配置文件
└── scripts/               # 自動化腳本
```

## 📥 數據獲取原理

### 1. 數據來源
系統使用 **Binance REST API** 獲取歷史價格數據，支持：
- **K-line 數據**: OHLCV (開盤、最高、最低、收盤、成交量)
- **時間範圍**: 可自定義，默認 5 年日線數據
- **幣種對**: 支持所有 Binance 交易對

### 2. 數據獲取流程

```bash
# 使用自動化腳本下載數據
./get_new_data.sh -s ETHUSDC,BTCUSDC -f 1d,1h -d 1825

# 或快速添加新幣種
./add_new_symbol.sh SOLUSDC 1d 365
```

**內部實現**:
1. **API 調用**: 使用 `requests` 庫調用 Binance API
2. **分頁處理**: 自動處理 API 限制，分批獲取數據
3. **數據清洗**: 處理缺失值、異常值、時間戳對齊
4. **格式標準化**: 轉換為系統標準 CSV 格式
5. **目錄組織**: 按幣種和時間尺度自動分類

### 3. 數據結構
```csv
timestamp,open,high,low,close,volume
2020-09-05,335.22,394.61,309.57,335.22,1234567
2020-09-06,334.89,360.32,316.02,352.84,2345678
...
```

## 🔄 回測引擎原理

### 1. 回測流程

```python
# 核心回測流程
BacktestEngine.run_full_evaluation() -> Dict[str, Any]
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
- **交易執行**: 模擬真實的 AMM 操作
- **費用計算**: 包括 Gas 費用、滑點、管理費

**策略邏輯示例**:
```python
# Dynamic-Vol 策略核心邏輯
def calculate_ranges(self, price_data, current_price, portfolio_value):
    # 1. 計算波動率
    volatility = price_data['returns'].rolling(30).std()
    
    # 2. 動態調整位置寬度
    vol_adjustment = 1.5 - volatility * 10
    
    # 3. 應用價格偏差觸發
    if abs(price_change) > self.price_deviation_bps:
        return self._rebalance_positions()
    
    return current_ranges
```

### 3. 性能指標計算

**收益率指標**:
- **APR**: 年化收益率，基於累積收益計算
- **MDD**: 最大回撤，使用滾動最大值計算
- **Sharpe**: 夏普比率，風險調整後收益
- **Calmar**: Calmar 比率，收益與回撤比

**風險指標**:
- **IL (Impermanent Loss)**: 無常損失計算
- **LVR (Loss Versus Rebalancing)**: 相對於再平衡的損失
- **波動率**: 日收益率標準差

## 🎛️ 參數優化原理

### 1. Optuna 優化框架

```python
# 優化目標函數
def _objective(self, trial):
    # 1. 參數建議
    k_width = trial.suggest_float('k_width', 0.8, 2.0)
    price_deviation_bps = trial.suggest_float('price_deviation_bps', 20, 120)
    rebalance_cooldown_hours = trial.suggest_int('rebalance_cooldown_hours', 6, 48)
    
    # 2. 策略回測
    strategy = DynamicVolatilityStrategy(k_width, price_deviation_bps, rebalance_cooldown_hours)
    results = self._run_backtest(strategy)
    
    # 3. 返回優化目標 (最大化 APR)
    return results['apr']
```

### 2. 優化參數

**核心參數**:
- **k_width**: 位置寬度調整係數 (0.8-2.0)
- **price_deviation_bps**: 價格偏差觸發閾值 (20-120 bps)
- **rebalance_cooldown_hours**: 再平衡冷卻時間 (6-48 小時)

**優化策略**:
- **方向**: 最大化 APR
- **試驗次數**: 可配置 (建議 20-100 次)
- **搜索算法**: TPE (Tree-structured Parzen Estimator)
- **早停機制**: 支持提前終止低質量試驗

### 3. 優化結果

```bash
# 運行優化
python run.py full --pool ETHUSDC --freq 1d --study-name ethusdc_optimization --n-trials 20

# 優化結果示例
Best trial: 15
Best value: 12.46
Best params: {
    'k_width': 1.91,
    'price_deviation_bps': 32.34,
    'rebalance_cooldown_hours': 41
}
```

## 📈 結果生成原理

### 1. 圖表生成

**支持的圖表類型**:
1. **Equity Curves**: 淨值曲線比較
2. **APR vs MDD Scatter**: 風險-收益散點圖
3. **Fee vs Price PnL**: 費用與價格 PnL 分析
4. **Sensitivity Heatmap**: 參數敏感性熱圖
5. **Gas vs Frequency Contour**: Gas 費用與頻率分析
6. **IL Curve**: 無常損失曲線
7. **LVR Estimates**: LVR 估算圖

**圖表生成流程**:
```python
# 圖表生成器
class PlotGenerator:
    def plot_equity_curves(self, results, save_path):
        # 1. 數據準備
        strategies = self._extract_strategy_data(results)
        
        # 2. 圖表創建
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 3. 數據繪製
        for strategy_name, strategy_data in strategies.items():
            equity_curve = self._calculate_equity_curve(strategy_data)
            ax.plot(equity_curve, label=strategy_name)
        
        # 4. 樣式設置
        ax.set_title(f'Equity Curves - {self.pool}')
        ax.legend()
        
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
results/
├── {幣種}_{時間尺度}_{時間戳}/
│   ├── strategy_record_{時間戳}.json
│   ├── strategy_summary_{時間戳}.csv
│   └── strategy_report_{時間戳}.txt
├── common/                    # 通用文件
└── INDEX.md                  # 目錄索引
```

## 🚀 快速開始

### 一行命令回測新CSV

```bash
# 最推薦：使用腳本回測
./run_single.sh BTCUSDC 1d 50

# 直接命令回測
python run.py full --pool BTCUSDC --freq 1d --n-trials 50
```

**參數說明**：
- `BTCUSDC`: 替換為您的幣種代碼
- `1d`: 數據頻率 (1d=日線, 1h=小時線)  
- `50`: 優化試驗次數

**結果位置**：
- 📊 圖表：`reports/figs/btcusdc/`
- 📈 數據：`results/strategy_*_*.json`
- 📋 報告：`results/strategy_report_*.txt`

### 🔗 整合Steer Intent Backtester策略

```bash
# 比較AMM和Steer策略
./run_steer_comparison.sh BTCUSDC 1d

# 或者直接運行
python final_integration.py
```

**整合功能**：
- 🔄 同時測試AMM和Steer策略
- 📊 生成策略比較報告
- 🏆 找出最佳策略組合
- 📈 支持多種策略類型比較

### 📁 數據文件結構設置

系統期望的數據文件結構：
```
data/
├── BTCUSDC/
│   └── price_1d.csv
├── ETHUSDC/
│   └── price_1d.csv
└── USDCUSDT/
    └── price_1d.csv
```

**如果您有5年數據文件** (如 `data/5year_daily/ETHUSDC_1d_20200905_20250903.csv`)：

```bash
# 創建目錄結構
mkdir -p data/ETHUSDC data/USDCUSDT

# 創建符號鏈接 (推薦)
ln -sf ../5year_daily/ETHUSDC_1d_20200905_20250903.csv data/ETHUSDC/price_1d.csv
ln -sf ../5year_daily/USDCUSDT_1d_20200905_20250903.csv data/USDCUSDT/price_1d.csv

# 或者複製文件
cp data/5year_daily/ETHUSDC_1d_20200905_20250903.csv data/ETHUSDC/price_1d.csv
cp data/5year_daily/USDCUSDT_1d_20200905_20250903.csv data/USDCUSDT/price_1d.csv
```

### 📊 回測結果示例

**USDCUSDT 5年回測結果** (2020-2025):
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

**最佳參數**：
- K寬度倍數: 1.96
- 價格偏差閾值: 113.85 bps
- 再平衡冷卻時間: 41小時

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
