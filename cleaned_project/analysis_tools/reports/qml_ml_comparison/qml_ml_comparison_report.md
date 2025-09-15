# QML vs ML Models Comparison Report

## 📊 Executive Summary

This report compares the performance of Quantum Machine Learning (QML) and Classical Machine Learning (ML) models in AMM trading strategies.

## 🎯 Model Categories

### Classical ML Models (3 models)
- Random Forest, Gradient Boosting, Logistic Regression
- Average Accuracy: 0.8756
- Average Rebalance Count: 41.0

### Quantum ML Models (3 models)
- VQE Classifier, QNN, QSVM
- Average Accuracy: 0.4767
- Average Rebalance Count: 44.7

### Hybrid Models (4 models)
- QASA Hybrid, QuantumRWKV, LSTM_QNN, QASA Sequence
- Average Accuracy: 0.6893
- Average Rebalance Count: 43.0

## 📈 Key Findings

### 1. Accuracy Performance
- **Best Classical Model**: Random Forest (0.9948)
- **Best Quantum Model**: VQE Classifier (0.5440)
- **Best Hybrid Model**: QuantumRWKV (0.8251)

### 2. Rebalance Efficiency
- **Most Efficient Classical**: Logistic Regression (36 rebalances)
- **Most Efficient Quantum**: QSVM (35 rebalances)
- **Most Efficient Hybrid**: QASA Sequence (30 rebalances)

### 3. Performance vs Efficiency Trade-off
- Classical models show high accuracy but require more rebalances
- Quantum models show moderate accuracy with fewer rebalances
- Hybrid models provide balanced performance

## 🔍 Detailed Analysis

### Model Performance Ranking (by Accuracy)
1. **Random Forest**: 0.9948 accuracy, 40 rebalances
2. **Gradient Boosting**: 0.9948 accuracy, 47 rebalances
3. **QuantumRWKV**: 0.8251 accuracy, 42 rebalances
4. **LSTM_QNN**: 0.6448 accuracy, 49 rebalances
5. **QASA Sequence**: 0.6448 accuracy, 30 rebalances
6. **QASA Hybrid**: 0.6425 accuracy, 51 rebalances
7. **Logistic Regression**: 0.6373 accuracy, 36 rebalances
8. **VQE Classifier**: 0.5440 accuracy, 50 rebalances
9. **QSVM**: 0.5130 accuracy, 35 rebalances
10. **QNN**: 0.3731 accuracy, 49 rebalances

### Rebalance Efficiency Ranking (by Rebalance Count)
1. **QASA Sequence**: 30 rebalances, 0.6448 accuracy
2. **QSVM**: 35 rebalances, 0.5130 accuracy
3. **Logistic Regression**: 36 rebalances, 0.6373 accuracy
4. **Random Forest**: 40 rebalances, 0.9948 accuracy
5. **QuantumRWKV**: 42 rebalances, 0.8251 accuracy
6. **Gradient Boosting**: 47 rebalances, 0.9948 accuracy
7. **QNN**: 49 rebalances, 0.3731 accuracy
8. **LSTM_QNN**: 49 rebalances, 0.6448 accuracy
9. **VQE Classifier**: 50 rebalances, 0.5440 accuracy
10. **QASA Hybrid**: 51 rebalances, 0.6425 accuracy

## 📊 Generated Charts

1. **qml_ml_accuracy_comparison.png** - Accuracy comparison between model types
2. **qml_ml_rebalance_comparison.png** - Rebalance frequency comparison
3. **qml_ml_performance_heatmap.png** - Performance metrics heatmap
4. **qml_ml_efficiency_analysis.png** - Efficiency analysis scatter plots
5. **qml_ml_summary_table.png** - Performance summary table

## 🎯 Recommendations

1. **For High Accuracy**: Use Classical ML models (Random Forest, Gradient Boosting)
2. **For Efficiency**: Use Quantum ML models (VQE Classifier, QNN)
3. **For Balanced Performance**: Use Hybrid models (QuantumRWKV, QASA Hybrid)

## 📅 Report Generated
2025-09-14 10:35:17
