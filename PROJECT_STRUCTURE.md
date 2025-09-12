# 專案結構說明

## 🎯 清理後的專案結構

### 📁 cleaned_project/ (推薦使用)
**簡潔的核心專案，包含所有必要的執行代碼**

```
cleaned_project/
├── core_scripts/           # 核心執行腳本
│   ├── training_testing_toolkit.py      # 綜合訓練工具
│   ├── improved_training_demo.py        # 經典ML訓練
│   ├── improved_quantum_training_demo.py # 量子ML訓練
│   ├── simplified_qasa_benchmark.py     # QASA基準測試
│   └── quantum_angle_encoding.py        # 角度編碼工具
├── analysis_tools/         # 分析工具
│   ├── create_unified_model_comparison.py # 統一模型比較
│   ├── analyze_classical_features_for_quantum.py # 特徵分析
│   └── quantum_ml_with_angle_encoding.py # 量子ML分析
├── backtesters/            # 回測系統
│   ├── amm-rebalance-backtester/        # AMM回測
│   └── steer_intent_backtester/         # 集中流動性回測
├── paper_assets/           # 論文資源
│   ├── paper_figures/                    # 18張論文圖表
│   ├── QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex # LaTeX論文
│   ├── references.bib                    # 參考文獻
│   └── compile_latex_paper.sh            # 編譯腳本
├── README.md               # 使用說明
├── requirements.txt        # 依賴列表
└── run_all_experiments.py # 統一執行腳本
```

### 📁 原始專案文件 (保留用於參考)

#### amm-rebalance-backtester/
- **用途**: AMM自動化做市商回測系統
- **核心文件**: `run.py`, `src/` 目錄
- **使用**: `python run.py quick --pool ETHUSDC --freq 1h`

#### steer_intent_backtester/
- **用途**: 集中流動性回測系統
- **核心文件**: `cli.py`, `steerbt/` 目錄
- **使用**: `python cli.py backtest --pair ETHUSDC --interval 1h`

#### paper_figures/
- **用途**: 論文圖表資源
- **內容**: 18張高品質圖表，按類別組織
- **說明**: `FIGURE_DESCRIPTIONS.md`

## 🚀 快速開始

### 1. 使用清理後的專案 (推薦)
```bash
cd cleaned_project
pip install -r requirements.txt
python run_all_experiments.py
```

### 2. 單獨執行核心功能
```bash
# 經典ML訓練
python core_scripts/improved_training_demo.py

# 量子ML訓練
python core_scripts/improved_quantum_training_demo.py

# 統一模型比較
python analysis_tools/create_unified_model_comparison.py
```

### 3. 回測系統
```bash
# AMM回測
cd backtesters/amm-rebalance-backtester
python run.py quick --pool ETHUSDC --freq 1h

# 集中流動性回測
cd backtesters/steer_intent_backtester
python cli.py backtest --pair ETHUSDC --interval 1h
```

## 📊 核心功能說明

### 模型訓練
- **經典ML**: Random Forest, Gradient Boosting, Logistic Regression
- **量子ML**: Qiskit VQC, PennyLane QNN
- **混合模型**: QASA (Quantum-Classical Hybrid)

### 特徵工程
- **經典特徵**: 122個技術指標和市場特徵
- **量子特徵**: 角度編碼映射到[0, 2π]範圍
- **特徵選擇**: 基於重要性的特徵排序

### 回測系統
- **AMM策略**: 自動化做市商再平衡
- **集中流動性**: 布林帶位置管理
- **多資產支持**: BTCUSDC, ETHUSDC, USDCUSDT

### 分析工具
- **性能比較**: 54個模型的綜合比較
- **統計分析**: 顯著性檢驗和置信區間
- **可視化**: 高品質圖表生成

## 🎯 主要發現

1. **量子模型優勢**: 在BTCUSDC和技術指標任務中表現優異
2. **經典模型穩定性**: 在高波動環境中更穩定
3. **混合策略**: QASA在特定場景下表現最佳
4. **特徵工程**: 角度編碼有效提升量子模型性能

## 📈 結果查看

### 訓練結果
- `cleaned_project/` 目錄下運行後會生成 `reports/` 目錄
- 包含所有訓練結果和分析圖表

### 回測結果
- AMM回測: `backtesters/amm-rebalance-backtester/reports/`
- 集中流動性回測: `backtesters/steer_intent_backtester/reports/`

### 論文資源
- 圖表: `paper_assets/paper_figures/`
- LaTeX論文: `paper_assets/QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex`

---

*專案結構說明 - 2025-09-12*
