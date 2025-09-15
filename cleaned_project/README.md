# Quantum vs Classical ML in AMM Trading Strategies

## ⭐ Main Entry Point

**`python analysis_tools/unified_label_training.py`** is the core file of this project!

This script will:
- 🚀 Train 10 different machine learning models (classical + quantum + hybrid)
- 📊 Automatically generate all analysis charts (accuracy, confusion matrices, feature importance, etc.)
- 🔍 Perform uncertainty analysis and risk assessment
- 💾 Save complete training results and reports

**Quick Start**:
```bash
cd analysis_tools
python unified_label_training.py
```

## 🎯 Project Overview

This is a comprehensive comparative study of quantum machine learning versus classical machine learning in Automated Market Maker (AMM) and Decentralized Finance (DeFi) trading strategies. Through 5-year backtesting (2020-2025) of 54 models on 3 cryptocurrency assets, we provide empirical evidence for quantum machine learning applications in the financial domain.

---

## 🚀 Quick Start

### 1. **Unified Training System** ⭐ **Main Recommendation**

#### Complete Model Training and Comparison
```bash
cd analysis_tools
python unified_label_training.py
```

**🎯 This is the core file of the project!** One-click completion of all features:
- Train 10 models (3 classical + 7 quantum/hybrid)
- Automatically generate all analysis charts
- Include uncertainty analysis and 3x3 confusion matrices
- Support LSTM_QNN and true QASA Sequence models
- Fixed all known issues (VQE Classifier, confusion matrices, feature importance, etc.)

#### Individual Model Training
```bash
# LSTM + Quantum Neural Network
python analysis_tools/lstm_qnn_model.py

# True QASA Algorithm
python analysis_tools/qasa_sequence_model.py

# Quantum RWKV Model
python analysis_tools/qrwkv_model.py
```

### 2. **Analysis Tools**

#### Uncertainty Charts Generation
```bash
python analysis_tools/create_uncertainty_charts.py
```

#### Experiment Results Export
```bash
python analysis_tools/export_experiment_results.py
```

#### Multiple Runs Analysis
```bash
python analysis_tools/multiple_runs_analysis.py
```

### 3. **Backtesting Systems**

#### AMM Backtesting
```bash
cd backtesters/amm-rebalance-backtester
python run.py quick --pool ETHUSDC --freq 1h --fee-mode proxy
```

#### Steer Intent Backtesting
```bash
cd backtesters/steer_intent_backtester
python cli.py backtest --pair ETHUSDC --interval 1h
```

### 4. **Paper Compilation**

```bash
cd paper_assets
chmod +x compile_latex_paper.sh
./compile_latex_paper.sh
```

## 📁 Project Structure

```
cleaned_project/
├── analysis_tools/         # 🎯 Analysis Tools (Main)
│   ├── unified_label_training.py          # ⭐ Main Entry Point - Unified Training System
│   ├── create_uncertainty_charts.py       # Uncertainty Charts
│   ├── export_experiment_results.py       # Results Export
│   ├── multiple_runs_analysis.py          # Multiple Runs Analysis
│   ├── lstm_qnn_model.py                  # LSTM+Quantum Model
│   ├── qasa_sequence_model.py             # True QASA Algorithm
│   ├── qrwkv_model.py                     # Quantum RWKV Model
│   └── reports/                           # Analysis Results
│       ├── unified_label_training/        # Unified Training Results
│       ├── uncertainty_charts/            # Uncertainty Charts
│       └── exported_results/              # Exported Results
├── core_scripts/           # 🔧 Core Execution Scripts
│   ├── training_testing_toolkit.py        # Comprehensive Training Toolkit
│   ├── improved_training_demo.py          # Classical ML Training
│   ├── improved_quantum_training_demo.py  # Quantum ML Training
│   └── quantum_angle_encoding.py          # Angle Encoding Tools
├── backtesters/            # 🚀 Backtesting Systems
│   ├── amm-rebalance-backtester/          # AMM Backtesting
│   └── steer_intent_backtester/           # Steer Intent Backtesting
├── paper_assets/           # 📄 Paper Resources
│   ├── paper_figures/                      # Paper Figures
│   ├── QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex # LaTeX Paper
│   ├── references.bib                      # References
│   └── compile_latex_paper.sh              # Compilation Script
└── README.md               # This File
```

## 🎯 Core Features

### **Model Training (10 Models)**
- **Classical ML**: Random Forest, Gradient Boosting, Logistic Regression
- **Quantum ML**: VQE Classifier, QNN, QSVM
- **Hybrid Models**: QASA Hybrid, QuantumRWKV, LSTM_QNN, QASA Sequence

### **Feature Engineering**
- **Core Feature**: `price_ma_ratio` (Price/20-day Moving Average Ratio)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, etc.
- **Feature Importance**: Automatic analysis and visualization

### **Chart Generation**
- **Accuracy Comparison**: Bar charts + Pie charts
- **Confusion Matrices**: 3x3 layout, automatically hide empty plots
- **Feature Importance**: Dynamic adaptation to feature count
- **Uncertainty Analysis**: Error bars, shadow regions, Sharpe ratio reference lines
- **Equity Curves**: Time series charts with uncertainty

### **Backtesting Systems**
- **AMM Strategy**: Automated Market Maker rebalancing
- **Steer Intent**: Bollinger Band position management
- **Multi-Asset Support**: BTCUSDC, ETHUSDC, USDCUSDT

## 📊 Results Viewing

### **Unified Training Results**
- `analysis_tools/reports/unified_label_training/` - Main results directory
  - `accuracy_comparison.png` - Accuracy comparison chart
  - `confusion_matrices.png` - 3x3 confusion matrices
  - `feature_importance.png` - Feature importance analysis
  - `performance_summary.png` - Performance summary
  - `uncertainty_charts/` - Uncertainty analysis charts

### **Exported Results**
- `analysis_tools/reports/exported_results/` - All experiment data
  - CSV format raw results
  - JSON format statistical data
  - Excel format summary reports
  - Markdown format README

### **Paper Figures**
- `paper_assets/paper_figures/` - 18 high-quality paper figures

## 🔧 Dependencies Installation

```bash
pip install -r requirements.txt
```

**Main Dependencies**:
- **Base**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Quantum**: qiskit, qiskit-algorithms, pennylane
- **Deep Learning**: torch, torchvision
- **Optimization**: optuna
- **Others**: click, openpyxl

## 🎯 Key Findings

### **1. Feature Importance Analysis**
- **`price_ma_ratio`** is the most important feature
- Gradient Boosting uses only this one feature (normal phenomenon)
- Other features are highly correlated with `price_ma_ratio`

### **2. Model Performance**
- **Classical Models**: Random Forest, Gradient Boosting accuracy > 99%
- **Quantum Models**: VQE Classifier accuracy ~54%
- **Hybrid Models**: QuantumRWKV accuracy ~83%

### **3. Data Splitting Strategy**
- Uses 252 trading days data (2024)
- 70/15/15 train/validation/test split
- Based on AMM Baseline label standards

## 🚨 Known Issues and Fixes

### **Fixed Issues**
1. ✅ **VQE Classifier 0% accuracy** - Fixed Qiskit API compatibility issues
2. ✅ **Confusion matrix empty plots** - Changed to 3x3 layout, automatically hide empty subplots
3. ✅ **Feature importance dimension mismatch** - Dynamic adaptation to feature count
4. ✅ **Uncertainty charts not generated** - Integrated into unified training system

### **Model Descriptions**
- **LSTM_QNN**: Originally "QASA Sequence", actually LSTM+Quantum Neural Network
- **QASA Sequence**: New implementation of true QASA algorithm
- **VQE Classifier**: Uses Variational Quantum Classifier, API issues fixed

## 📈 Usage Recommendations

1. **⭐ First Time Use**: Run `python analysis_tools/unified_label_training.py` for complete results
2. **Model Comparison**: Check `accuracy_comparison.png` and `confusion_matrices.png`
3. **Feature Analysis**: Check `feature_importance.png` to understand feature importance
4. **Uncertainty Analysis**: Check charts in `uncertainty_charts/` directory
5. **Data Export**: Run `export_experiment_results.py` to export all data

## 🔄 Update Log

- **2025-01-13**: Integrated unified training system, fixed all chart issues
- **2025-01-13**: Added LSTM_QNN and true QASA Sequence models
- **2025-01-13**: Fixed VQE Classifier API compatibility issues
- **2025-01-13**: Enhanced uncertainty analysis and 3x3 confusion matrices

## 🌐 Language Versions

- **English**: Current version (this file)
- **中文版**: [README_中文.md](README_中文.md) - 完整中文版文檔

---

*Complete Project Documentation - 2025-01-13*