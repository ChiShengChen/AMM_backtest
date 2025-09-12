# Quantum vs Classical ML in AMM Trading Strategies

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantum](https://img.shields.io/badge/Quantum-ML-purple.svg)](https://pennylane.ai)

## 🎯 專案概述

本專案是一個全面的量子機器學習與經典機器學習在自動化做市商(AMM)和去中心化金融(DeFi)交易策略中的比較研究。通過54個模型在3個加密貨幣資產上的5年回測(2020-2025)，我們提供了量子機器學習在金融領域應用的實證證據。

## 🚀 快速開始

### 1. 使用清理後的專案 (推薦)
```bash
cd cleaned_project
pip install -r requirements.txt
python run_all_experiments.py
```

### 2. 單獨執行核心功能
```bash
# 經典機器學習訓練
python core_scripts/improved_training_demo.py

# 量子機器學習訓練
python core_scripts/improved_quantum_training_demo.py

# 統一模型比較分析
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

## 📁 專案結構

```
Omnis_bt/
├── cleaned_project/           # 🎯 推薦使用 - 簡潔的核心專案
│   ├── core_scripts/         # 核心執行腳本
│   ├── analysis_tools/       # 分析工具
│   ├── backtesters/          # 回測系統
│   ├── paper_assets/         # 論文資源
│   └── README.md             # 詳細使用說明
├── amm-rebalance-backtester/ # AMM回測系統
├── steer_intent_backtester/  # 集中流動性回測系統
├── paper_figures/            # 論文圖表資源
└── PROJECT_STRUCTURE.md      # 專案結構詳細說明
```

## 🔬 研究內容

### 模型類型
- **經典機器學習**: Random Forest, Gradient Boosting, Logistic Regression
- **量子機器學習**: Qiskit VQC, PennyLane QNN
- **混合模型**: QASA (Quantum-Classical Hybrid)

### 特徵工程
- **經典特徵**: 122個技術指標和市場特徵
- **量子特徵**: 角度編碼映射到[0, 2π]範圍
- **特徵選擇**: 基於重要性的特徵排序

### 回測策略
- **AMM策略**: 自動化做市商再平衡
- **集中流動性**: 布林帶位置管理
- **多資產支持**: BTCUSDC, ETHUSDC, USDCUSDT

## 📊 主要發現

### 量子模型優勢
- 在BTCUSDC資產交易中表現優異
- 技術指標模式識別任務中平均收益優勢8.97%
- 量子糾纏在非線性關係處理中的優勢

### 經典模型穩定性
- 在高波動環境(如ETHUSDC)中更穩定
- 快速響應和計算效率優勢
- 生產環境中的可靠性

### 混合策略
- QASA在特定場景下表現最佳
- 結合量子計算優勢和經典穩定性
- 特徵工程的有效性驗證

## 🛠️ 技術棧

### 核心依賴
- **機器學習**: scikit-learn, pandas, numpy
- **量子計算**: Qiskit, PennyLane
- **可視化**: matplotlib, seaborn, plotly
- **優化**: Optuna
- **回測**: 自定義回測引擎

### 量子框架
- **Qiskit**: IBM量子計算框架
- **PennyLane**: 量子機器學習框架
- **角度編碼**: 經典特徵到量子態的映射

## 📈 結果展示

### 性能比較
- 54個模型的綜合性能比較
- 統計顯著性檢驗和置信區間
- 風險調整後收益分析

### 可視化圖表
- 18張高品質論文圖表
- 統一模型比較圖
- 特徵重要性分析
- 量子電路架構圖

## 📚 論文資源

### LaTeX論文
- 完整的學術論文 (英文)
- 數學公式和理論描述
- 實證結果和分析

### 圖表資源
- 18張論文級別圖表
- 按類別組織的圖表說明
- 高解析度PNG格式

## 🔧 安裝與配置

### 環境要求
- Python 3.8+
- 8GB+ RAM (量子模型需要)
- 支援CUDA的GPU (可選，用於加速)

### 安裝步驟
```bash
# 克隆專案
git clone <repository-url>
cd Omnis_bt

# 安裝依賴
cd cleaned_project
pip install -r requirements.txt

# 運行所有實驗
python run_all_experiments.py
```

## 📖 使用指南

### 1. 模型訓練
```bash
# 經典ML訓練
python core_scripts/improved_training_demo.py

# 量子ML訓練
python core_scripts/improved_quantum_training_demo.py

# 綜合訓練工具
python core_scripts/training_testing_toolkit.py
```

### 2. 分析工具
```bash
# 統一模型比較
python analysis_tools/create_unified_model_comparison.py

# 特徵分析
python analysis_tools/analyze_classical_features_for_quantum.py

# 量子ML分析
python analysis_tools/quantum_ml_with_angle_encoding.py
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

## 📊 結果查看

### 訓練結果
- `cleaned_project/reports/` - 訓練結果和分析圖表
- `cleaned_project/paper_assets/paper_figures/` - 論文圖表

### 回測結果
- AMM回測: `backtesters/amm-rebalance-backtester/reports/`
- 集中流動性回測: `backtesters/steer_intent_backtester/reports/`

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權

本專案採用 MIT 授權 - 查看 [LICENSE](LICENSE) 文件了解詳情。

## 📞 聯繫方式

- 專案維護者: [Your Name]
- 電子郵件: [your.email@example.com]
- 專案連結: [https://github.com/yourusername/Omnis_bt](https://github.com/yourusername/Omnis_bt)

## 🙏 致謝

- Qiskit 團隊提供的量子計算框架
- PennyLane 團隊提供的量子機器學習工具
- 開源社群的支持和貢獻

---

**⭐ 如果這個專案對您有幫助，請給我們一個星標！**
