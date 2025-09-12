# Quantum vs Classical ML in AMM Trading Strategies

## 🚀 快速開始

### 1. 核心訓練腳本

#### 經典機器學習訓練
```bash
python core_scripts/improved_training_demo.py
```

#### 量子機器學習訓練
```bash
python core_scripts/improved_quantum_training_demo.py
```

#### 綜合訓練測試工具包
```bash
python core_scripts/training_testing_toolkit.py
```

#### QASA基準測試
```bash
python core_scripts/simplified_qasa_benchmark.py
```

### 2. 分析工具

#### 統一模型比較
```bash
python analysis_tools/create_unified_model_comparison.py
```

#### 經典特徵分析
```bash
python analysis_tools/analyze_classical_features_for_quantum.py
```

#### 量子角度編碼
```bash
python analysis_tools/quantum_ml_with_angle_encoding.py
```

### 3. 回測系統

#### AMM回測
```bash
cd backtesters/amm-rebalance-backtester
python run.py quick --pool ETHUSDC --freq 1h --fee-mode proxy
```

#### 集中流動性回測
```bash
cd backtesters/steer_intent_backtester
python cli.py backtest --pair ETHUSDC --interval 1h
```

### 4. 論文編譯

```bash
cd paper_assets
chmod +x compile_latex_paper.sh
./compile_latex_paper.sh
```

## 📁 專案結構

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
│   ├── paper_figures/                    # 論文圖表
│   ├── QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex # LaTeX論文
│   ├── references.bib                    # 參考文獻
│   └── compile_latex_paper.sh            # 編譯腳本
└── README.md               # 本文件
```

## 🎯 核心功能

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

## 📊 結果查看

### 訓練結果
- `reports/training_evaluation/` - 經典ML結果
- `reports/quantum_training_evaluation/` - 量子ML結果
- `reports/unified_model_comparison/` - 統一比較結果

### 回測結果
- `backtesters/amm-rebalance-backtester/reports/` - AMM回測結果
- `backtesters/steer_intent_backtester/reports/` - 集中流動性回測結果

### 論文圖表
- `paper_assets/paper_figures/` - 18張高品質論文圖表

## 🔧 依賴安裝

```bash
pip install -r requirements.txt
```

主要依賴：
- pandas, numpy, scikit-learn
- qiskit, pennylane
- matplotlib, seaborn
- optuna, click

## 📈 主要發現

1. **量子模型優勢**: 在BTCUSDC和技術指標任務中表現優異
2. **經典模型穩定性**: 在高波動環境中更穩定
3. **混合策略**: QASA在特定場景下表現最佳
4. **特徵工程**: 角度編碼有效提升量子模型性能

---

*簡潔版專案結構 - 2025-09-12*
