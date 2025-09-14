# 實驗結果數據導出

## 概述
本目錄包含所有機器學習模型在金融數據上的實驗結果。

## 數據統計
- **總模型數**: 10
- **總運行次數**: 50
- **實驗期間**: 2024年1月1日 - 2024年12月31日 (252個交易日)

## 文件說明

### 主要文件
- `experiment_summary.json`: 完整的實驗摘要統計
- `main_metrics_summary.csv`: 主要性能指標摘要
- `performance_rankings.json`: 各項指標的性能排名
- `performance_rankings.csv`: 性能排名表格
- `experiment_results_summary.xlsx`: Excel格式的完整摘要報告

### 時間序列數據
- `all_models_time_series.csv`: 所有模型的時間序列數據
- `[Model_Name]_time_series.csv`: 個別模型的時間序列數據

### 原始結果
- `all_raw_results.csv`: 所有原始實驗結果
- `[Model_Name]_raw_results.csv`: 個別模型的原始結果

## 性能排名 (基於Sharpe Ratio)

1. **QASA Sequence**: 1.761
2. **Random Forest**: 1.680
3. **Gradient Boosting**: 1.677
4. **QASA Hybrid**: 1.324
5. **Transformer**: 1.231
6. **QuantumRWKV**: 1.185
7. **Logistic Regression**: 1.056
8. **QSVM**: 0.867
9. **QNN**: 0.824
10. **VQE Classifier**: 0.788

## 模型類型分組

### Classic ML
- Random Forest
- Gradient Boosting  
- Logistic Regression

### Quantum ML
- VQE Classifier
- QNN
- QSVM

### Hybrid Quantum
- QASA Hybrid
- QASA Sequence
- QuantumRWKV

### Transformer
- Transformer

## 主要指標說明

- **Annual Return**: 年化收益率
- **Volatility**: 波動率
- **Sharpe Ratio**: 夏普比率 (風險調整後收益)
- **Max Drawdown**: 最大回撤
- **Calmar Ratio**: 卡爾瑪比率 (年化收益/最大回撤)
- **Win Rate**: 勝率
- **Profit Factor**: 盈利因子

## 數據格式

所有CSV文件使用UTF-8編碼，JSON文件使用UTF-8編碼並格式化輸出。

生成時間: 2025-09-13 13:07:39
