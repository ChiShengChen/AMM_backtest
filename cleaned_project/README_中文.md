# 量子與經典機器學習在AMM交易策略中的應用

## ⭐ 主要入口點

**`python analysis_tools/unified_label_training.py`** 是本項目的核心文件！

這個腳本會：
- 🚀 訓練10個不同的機器學習模型（經典+量子+混合）
- 📊 自動生成所有分析圖表（準確率、混淆矩陣、特徵重要性等）
- 🔍 進行不確定性分析和風險評估
- 💾 保存完整的訓練結果和報告

**快速開始**：
```bash
cd analysis_tools
python unified_label_training.py
```

---

## 🚀 快速開始

### 1. 統一訓練系統 ⭐ **主要推薦**

#### 完整模型訓練與比較
```bash
cd analysis_tools
python unified_label_training.py
```

**🎯 這是本項目的核心文件！** 一鍵完成所有功能：
- 訓練10個模型（3個經典 + 7個量子/混合）
- 自動生成所有分析圖表
- 包含不確定性分析和3x3混淆矩陣
- 支持LSTM_QNN和真正的QASA Sequence模型
- 修正了所有已知問題（VQE Classifier、混淆矩陣、特徵重要性等）

#### 單個模型訓練
```bash
# LSTM + 量子神經網絡
python analysis_tools/lstm_qnn_model.py

# 真正的QASA算法
python analysis_tools/qasa_sequence_model.py

# 量子RWKV模型
python analysis_tools/qrwkv_model.py
```

### 2. 分析工具

#### 不確定性圖表生成
```bash
python analysis_tools/create_uncertainty_charts.py
```

#### 實驗結果導出
```bash
python analysis_tools/export_experiment_results.py
```

#### 多輪運行分析
```bash
python analysis_tools/multiple_runs_analysis.py
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
├── analysis_tools/         # 分析工具 (主要)
│   ├── unified_label_training.py          # ⭐ 主要入口點 - 統一訓練系統
│   ├── create_uncertainty_charts.py       # 不確定性圖表
│   ├── export_experiment_results.py       # 結果導出
│   ├── multiple_runs_analysis.py          # 多輪分析
│   ├── lstm_qnn_model.py                  # LSTM+量子模型
│   ├── qasa_sequence_model.py             # 真正QASA算法
│   ├── qrwkv_model.py                     # 量子RWKV模型
│   └── reports/                           # 分析結果
│       ├── unified_label_training/        # 統一訓練結果
│       ├── uncertainty_charts/            # 不確定性圖表
│       └── exported_results/              # 導出結果
├── core_scripts/           # 核心執行腳本
│   ├── training_testing_toolkit.py        # 綜合訓練工具
│   ├── improved_training_demo.py          # 經典ML訓練
│   ├── improved_quantum_training_demo.py  # 量子ML訓練
│   └── quantum_angle_encoding.py          # 角度編碼工具
├── backtesters/            # 回測系統
│   ├── amm-rebalance-backtester/          # AMM回測
│   └── steer_intent_backtester/           # 集中流動性回測
├── paper_assets/           # 論文資源
│   ├── paper_figures/                      # 論文圖表
│   ├── QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex # LaTeX論文
│   ├── references.bib                      # 參考文獻
│   └── compile_latex_paper.sh              # 編譯腳本
└── README.md               # 本文件
```

## 🎯 核心功能

### 模型訓練 (10個模型)
- **經典ML**: Random Forest, Gradient Boosting, Logistic Regression
- **量子ML**: VQE Classifier, QNN, QSVM
- **混合模型**: QASA Hybrid, QuantumRWKV, LSTM_QNN, QASA Sequence

### 特徵工程
- **核心特徵**: `price_ma_ratio` (價格/20日均線比率)
- **技術指標**: RSI, MACD, 布林帶, ATR等
- **特徵重要性**: 自動分析並可視化

### 圖表生成
- **準確率比較**: 條形圖 + 餅圖
- **混淆矩陣**: 3x3布局，自動隱藏空白圖
- **特徵重要性**: 動態適應特徵數量
- **不確定性分析**: 誤差條、陰影區域、Sharpe比率參考線
- **權益曲線**: 帶不確定性的時間序列圖

### 回測系統
- **AMM策略**: 自動化做市商再平衡
- **集中流動性**: 布林帶位置管理
- **多資產支持**: BTCUSDC, ETHUSDC, USDCUSDT

## 📊 結果查看

### 統一訓練結果
- `analysis_tools/reports/unified_label_training/` - 主要結果目錄
  - `accuracy_comparison.png` - 準確率比較圖
  - `confusion_matrices.png` - 3x3混淆矩陣
  - `feature_importance.png` - 特徵重要性分析
  - `performance_summary.png` - 性能摘要
  - `uncertainty_charts/` - 不確定性分析圖表

### 導出結果
- `analysis_tools/reports/exported_results/` - 所有實驗數據
  - CSV格式的原始結果
  - JSON格式的統計數據
  - Excel格式的摘要報告
  - Markdown格式的README

### 論文圖表
- `paper_assets/paper_figures/` - 18張高品質論文圖表

## 🔧 依賴安裝

```bash
pip install -r requirements.txt
```

主要依賴：
- **基礎**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **量子**: qiskit, qiskit-algorithms, pennylane
- **深度學習**: torch, torchvision
- **優化**: optuna
- **其他**: click, openpyxl

## 🎯 重要發現

### 1. 特徵重要性分析
- **`price_ma_ratio`** 是最重要的特徵
- Gradient Boosting只使用這一個特徵（正常現象）
- 其他特徵與`price_ma_ratio`高度相關

### 2. 模型性能
- **經典模型**: Random Forest, Gradient Boosting 準確率 > 99%
- **量子模型**: VQE Classifier 準確率 ~54%
- **混合模型**: QuantumRWKV 準確率 ~83%

### 3. 數據分割策略
- 使用252天交易數據（2024年）
- 70/15/15 訓練/驗證/測試分割
- 基於AMM Baseline標籤標準

## 🚨 已知問題與修正

### 已修正的問題
1. ✅ **VQE Classifier 0%準確率** - 修正了Qiskit API兼容性問題
2. ✅ **混淆矩陣空白圖** - 改為3x3布局，自動隱藏空白子圖
3. ✅ **特徵重要性維度不匹配** - 動態適應特徵數量
4. ✅ **不確定性圖表未生成** - 已整合到統一訓練系統

### 模型說明
- **LSTM_QNN**: 原"QASA Sequence"，實際是LSTM+量子神經網絡
- **QASA Sequence**: 新實現的真正QASA算法
- **VQE Classifier**: 使用變分量子分類器，已修正API問題

## 📈 使用建議

1. **⭐ 首次使用**: 運行 `python analysis_tools/unified_label_training.py` 獲得完整結果
2. **模型比較**: 查看 `accuracy_comparison.png` 和 `confusion_matrices.png`
3. **特徵分析**: 查看 `feature_importance.png` 了解特徵重要性
4. **不確定性分析**: 查看 `uncertainty_charts/` 目錄中的圖表
5. **數據導出**: 運行 `export_experiment_results.py` 導出所有數據

## 🔄 更新日誌

- **2025-01-13**: 整合統一訓練系統，修正所有圖表問題
- **2025-01-13**: 添加LSTM_QNN和真正QASA Sequence模型
- **2025-01-13**: 修正VQE Classifier API兼容性問題
- **2025-01-13**: 完善不確定性分析和3x3混淆矩陣

## 🌐 語言版本

- **English**: [README.md](README.md) - 英文版文檔
- **中文版**: 當前版本（本文件）

---

*完整版專案文檔 - 2025-01-13*
