# Quantum vs Classical ML in AMM Trading Strategies

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantum](https://img.shields.io/badge/Quantum-ML-purple.svg)](https://pennylane.ai)

## ⭐ Main Entry Point

**`python analysis_tools/unified_label_training.py`** is the core file of this project!

This script will:
- 🚀 Train 10 different machine learning models (classical + quantum + hybrid)
- 📊 Automatically generate all analysis charts (accuracy, confusion matrices, feature importance, etc.)
- 🔍 Perform uncertainty analysis and risk assessment
- 💾 Save complete training results and reports

**Quick Start**:
```bash
cd cleaned_project/analysis_tools
python unified_label_training.py
```

---

## 🎯 Project Overview

This project is a comprehensive comparative study of quantum machine learning versus classical machine learning in Automated Market Maker (AMM) and Decentralized Finance (DeFi) trading strategies. Through 5-year backtesting (2020-2025) of 54 models on 3 cryptocurrency assets, we provide empirical evidence for quantum machine learning applications in the financial domain.

## 🚀 Quick Start

### 1. Use Cleaned Project (Recommended)
```bash
cd cleaned_project
pip install -r requirements.txt
python run_all_experiments.py
```

### 2. Execute Core Functions Separately
```bash
# Classical Machine Learning Training
python core_scripts/improved_training_demo.py

# Quantum Machine Learning Training
python core_scripts/improved_quantum_training_demo.py

# Unified Model Comparison Analysis
python analysis_tools/create_unified_model_comparison.py
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

## 📁 Project Structure

```
Omnis_bt/
├── cleaned_project/           # 🎯 Recommended - Clean Core Project
│   ├── core_scripts/         # Core Execution Scripts
│   ├── analysis_tools/       # Analysis Tools
│   ├── backtesters/          # Backtesting Systems
│   ├── paper_assets/         # Paper Resources
│   └── README.md             # Detailed Usage Instructions
├── amm-rebalance-backtester/ # AMM Backtesting System
├── steer_intent_backtester/  # Concentrated Liquidity Backtesting System
├── paper_figures/            # Paper Figure Resources
└── PROJECT_STRUCTURE.md      # Detailed Project Structure
```

## 🔬 Research Content

### Model Types
- **Classical Machine Learning**: Random Forest, Gradient Boosting, Logistic Regression
- **Quantum Machine Learning**: Qiskit VQC, PennyLane QNN
- **Hybrid Models**: QASA (Quantum-Classical Hybrid)

### Feature Engineering
- **Classical Features**: 122 technical indicators and market features
- **Quantum Features**: Angle encoding mapped to [0, 2π] range
- **Feature Selection**: Importance-based feature ranking

### Backtesting Strategies
- **AMM Strategy**: Automated Market Maker rebalancing
- **Concentrated Liquidity**: Bollinger Band position management
- **Multi-Asset Support**: BTCUSDC, ETHUSDC, USDCUSDT

## 📊 Key Findings

### Quantum Model Advantages
- Excellent performance in BTCUSDC asset trading
- 8.97% average return advantage in technical indicator pattern recognition tasks
- Quantum entanglement advantages in nonlinear relationship processing

### Classical Model Stability
- More stable in high volatility environments (like ETHUSDC)
- Fast response and computational efficiency advantages
- Reliability in production environments

### Hybrid Strategies
- QASA performs best in specific scenarios
- Combines quantum computing advantages with classical stability
- Validation of feature engineering effectiveness

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