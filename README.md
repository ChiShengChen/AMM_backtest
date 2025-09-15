# Quantum vs Classical ML in AMM Trading Strategies

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantum](https://img.shields.io/badge/Quantum-ML-purple.svg)](https://pennylane.ai)
[![Steer](https://img.shields.io/badge/Steer-Intent-orange.svg)](https://uniswap.org)

## ⭐ Main Entry Points

### 1. **Unified ML Training** (Core Research)
**`python cleaned_project/analysis_tools/unified_label_training.py`**

This script will:
- 🚀 Train 10 different machine learning models (classical + quantum + hybrid)
- 📊 Automatically generate all analysis charts (accuracy, confusion matrices, feature importance, etc.)
- 🔍 Perform uncertainty analysis and risk assessment
- 💾 Save complete training results and reports

### 2. **Steer Strategy Comparison** (Trading Strategies)
**`python chart_generators/run_all_analysis.py`**

This script will:
- 🔧 Fix steer strategy cash depletion issues
- 📈 Generate comprehensive strategy comparison charts
- 🏆 Compare Fixed, ML, and Quantum strategies
- 📊 Create performance analysis and ranking tables

**Quick Start**:
```bash
# ML Research (Core)
cd cleaned_project/analysis_tools
python unified_label_training.py

# Trading Strategies (Steer)
cd chart_generators
python run_all_analysis.py
```

---

## 🎯 Project Overview

This project is a comprehensive comparative study of quantum machine learning versus classical machine learning in Automated Market Maker (AMM) and Decentralized Finance (DeFi) trading strategies. Through 5-year backtesting (2020-2025) of 54 models on 3 cryptocurrency assets, we provide empirical evidence for quantum machine learning applications in the financial domain.

## 🚀 Quick Start

### 1. **ML Research** (Core Project)
```bash
cd cleaned_project
pip install -r requirements.txt

# Unified ML Training (Recommended)
python analysis_tools/unified_label_training.py

# Individual Model Training
python core_scripts/improved_training_demo.py
python core_scripts/improved_quantum_training_demo.py
```

### 2. **Trading Strategies** (Steer Intent)
```bash
# Complete Steer Strategy Analysis
cd chart_generators
python run_all_analysis.py

# Individual Analysis Steps
python run_steer_comparison.py
python create_enhanced_comparison_charts.py
python create_improved_comparison_charts.py
```

### 3. **Backtesting Systems**
```bash
# AMM Backtesting
cd amm-rebalance-backtester
python run.py quick --pool ETHUSDC --freq 1h

# Steer Intent Backtesting
cd steer_intent_backtester
python cli.py backtest --pair ETHUSDC --interval 1h
```

## 📁 Project Structure

```
Omnis_bt/
├── cleaned_project/           # 🎯 ML Research - Core Project
│   ├── analysis_tools/       # ML Analysis Tools
│   ├── core_scripts/         # Core Execution Scripts
│   ├── backtesters/          # Backtesting Systems
│   ├── paper_assets/         # Paper Resources
│   └── README.md             # ML Research Documentation
├── chart_generators/         # 🚀 Trading Strategies - Steer Analysis
│   ├── run_all_analysis.py   # Main Analysis Script
│   ├── run_steer_comparison.py
│   ├── create_enhanced_comparison_charts.py
│   ├── create_improved_comparison_charts.py
│   └── README.md             # Trading Strategy Documentation
├── amm-rebalance-backtester/ # AMM Backtesting System
├── steer_intent_backtester/  # Steer Intent Backtesting System
├── paper_figures/            # Paper Figure Resources
├── simplified_ultimate_comparison/ # Final Comparison Results
└── PROJECT_STRUCTURE.md      # Detailed Project Structure
```

## 🔬 Research Content

### ML Research (cleaned_project/)
- **Classical ML**: Random Forest, Gradient Boosting, Logistic Regression
- **Quantum ML**: Qiskit VQC, PennyLane QNN, VQE Classifier
- **Hybrid Models**: QASA, QuantumRWKV, LSTM_QNN
- **Feature Engineering**: 122 technical indicators, angle encoding
- **Uncertainty Analysis**: Error bars, confidence intervals

### Trading Strategies (chart_generators/)
- **Fixed Strategies**: Conservative, Moderate (Classic-based)
- **ML Strategies**: ML Bollinger, ML Keltner, ML Donchian
- **Quantum Strategies**: Quantum Bollinger, Quantum Keltner, Quantum Hybrid
- **Steer Intent**: 16 different CLMM strategies
- **Performance Metrics**: APR, Sharpe Ratio, Rebalance Count, Max Drawdown

### Backtesting Systems
- **AMM Strategy**: Automated Market Maker rebalancing
- **Steer Intent**: Concentrated Liquidity position management
- **Multi-Asset Support**: BTCUSDC, ETHUSDC, USDCUSDT

## 📊 Key Findings

### ML Research Results
- **Quantum Models**: VQE Classifier ~54% accuracy, QNN ~82% accuracy
- **Classical Models**: Random Forest, Gradient Boosting >99% accuracy
- **Hybrid Models**: QuantumRWKV ~83% accuracy, LSTM_QNN competitive
- **Feature Importance**: `price_ma_ratio` is the most critical feature

### Trading Strategy Results
- **Fixed Strategies**: Conservative (5% width) vs Moderate (10% width)
- **ML Strategies**: ML Bollinger 85% accuracy, 35 rebalances/year
- **Quantum Strategies**: Quantum Bollinger 82% accuracy, 30 rebalances/year
- **Performance**: Quantum strategies show better efficiency (lower rebalance, higher returns)

### Strategy Comparison
- **Fixed (Conservative)**: 5% width, uniform distribution, capital preservation
- **Fixed (Moderate)**: 10% width, uniform distribution, balanced approach
- **ML Bollinger**: Random Forest + Bollinger Bands, 70% ML + 30% traditional
- **Quantum Bollinger**: QNN + Bollinger Bands, quantum confidence adjustment

## 🛠️ Technology Stack

### Core Dependencies
- **Machine Learning**: scikit-learn, pandas, numpy
- **Quantum Computing**: Qiskit, PennyLane
- **Visualization**: matplotlib, seaborn, plotly
- **Optimization**: Optuna
- **Backtesting**: Custom backtesting engine

### Quantum Frameworks
- **Qiskit**: IBM quantum computing framework
- **PennyLane**: Quantum machine learning framework
- **Angle Encoding**: Classical feature to quantum state mapping

## 📈 Results Display

### Performance Comparison
- Comprehensive performance comparison of 54 models
- Statistical significance testing and confidence intervals
- Risk-adjusted return analysis

### Visualization Charts
- 18 high-quality paper figures
- Unified model comparison charts
- Feature importance analysis
- Quantum circuit architecture diagrams

## 📚 Paper Resources

### LaTeX Paper
- Complete academic paper (English)
- Mathematical formulas and theoretical descriptions
- Empirical results and analysis

### Figure Resources
- 18 paper-level figures
- Category-organized figure descriptions
- High-resolution PNG format

## 🔧 Installation and Configuration

### Environment Requirements
- Python 3.8+
- 8GB+ RAM (required for quantum models)
- CUDA-enabled GPU (optional, for acceleration)

### Installation Steps
```bash
# Clone project
git clone <repository-url>
cd Omnis_bt

# Install dependencies
cd cleaned_project
pip install -r requirements.txt

# Run all experiments
python run_all_experiments.py
```

## 📖 Usage Guide

### 1. Model Training
```bash
# Classical ML Training
python core_scripts/improved_training_demo.py

# Quantum ML Training
python core_scripts/improved_quantum_training_demo.py

# Comprehensive Training Toolkit
python core_scripts/training_testing_toolkit.py
```

### 2. Analysis Tools
```bash
# Unified Model Comparison
python analysis_tools/create_unified_model_comparison.py

# Feature Analysis
python analysis_tools/analyze_classical_features_for_quantum.py

# Quantum ML Analysis
python analysis_tools/quantum_ml_with_angle_encoding.py
```

### 3. Backtesting Systems
```bash
# AMM Backtesting
cd backtesters/amm-rebalance-backtester
python run.py quick --pool ETHUSDC --freq 1h

# Concentrated Liquidity Backtesting
cd backtesters/steer_intent_backtester
python cli.py backtest --pair ETHUSDC --interval 1h
```

## 📊 Results Viewing

### Training Results
- `cleaned_project/reports/` - Training results and analysis charts
- `cleaned_project/paper_assets/paper_figures/` - Paper figures

### Backtesting Results
- AMM Backtesting: `backtesters/amm-rebalance-backtester/reports/`
- Concentrated Liquidity Backtesting: `backtesters/steer_intent_backtester/reports/`

## 🤝 Contributing Guidelines

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Project Maintainer: [Your Name]
- Email: [your.email@example.com]
- Project Link: [https://github.com/yourusername/Omnis_bt](https://github.com/yourusername/Omnis_bt)

## 🙏 Acknowledgments

- Qiskit team for the quantum computing framework
- PennyLane team for quantum machine learning tools
- Open source community support and contributions

## 🌐 Language Versions

- **English**: Current version (this file)
- **中文版**: [README_中文.md](README_中文.md) - 完整中文版文檔

---

**⭐ If this project helps you, please give us a star!**