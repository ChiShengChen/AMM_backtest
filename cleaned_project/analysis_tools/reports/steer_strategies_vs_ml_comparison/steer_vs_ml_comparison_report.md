# Steer Strategies vs ML Models Comparison Report

## 📊 Executive Summary

This report compares the performance of 7 fixed Steer Strategies from steer_intent_backtester with 10 ML models (Classical, Quantum, and Hybrid) in AMM trading strategies.

## 🎯 Strategy Categories

### Steer Strategies (7 strategies)
- Classic Strategy, Channel Multiplier Strategy, Bollinger Strategy, Keltner Strategy, Donchian Strategy, Stable Strategy, Fluid Strategy
- Average Accuracy: 0.7166
- Average Rebalance Count: 36.0

### Classical ML Models (3 models)
- Random Forest, Gradient Boosting, Logistic Regression
- Average Accuracy: 0.8756
- Average Rebalance Count: 42.3

### Quantum ML Models (3 models)
- VQE Classifier, QNN, QSVM
- Average Accuracy: 0.4767
- Average Rebalance Count: 51.7

### Hybrid ML Models (4 models)
- QASA Hybrid, QuantumRWKV, LSTM_QNN, QASA Sequence
- Average Accuracy: 0.6893
- Average Rebalance Count: 36.2

## 📈 Key Findings

### 1. Accuracy Performance
- **Best Steer Strategy**: Stable Strategy (0.7654)
- **Best Classical ML**: Random Forest (0.9948)
- **Best Quantum ML**: VQE Classifier (0.5440)
- **Best Hybrid ML**: QuantumRWKV (0.8251)

### 2. Rebalance Efficiency
- **Most Efficient Steer**: Classic Strategy (28 rebalances)
- **Most Efficient Classical**: Gradient Boosting (38 rebalances)
- **Most Efficient Quantum**: QSVM (50 rebalances)
- **Most Efficient Hybrid**: QuantumRWKV (33 rebalances)

### 3. Performance vs Efficiency Trade-off
- **Steer Strategies**: Moderate accuracy with good rebalance efficiency
- **Classical ML**: High accuracy but more rebalances required
- **Quantum ML**: Lower accuracy with moderate rebalance frequency
- **Hybrid ML**: Balanced performance across all metrics

## 🔍 Detailed Analysis

### Overall Performance Ranking (by Accuracy)
1. **Random Forest**: 0.9948 accuracy, 41 rebalances
2. **Gradient Boosting**: 0.9948 accuracy, 38 rebalances
3. **QuantumRWKV**: 0.8251 accuracy, 33 rebalances
4. **Stable Strategy**: 0.7654 accuracy, 45 rebalances
5. **Bollinger Strategy**: 0.7456 accuracy, 42 rebalances
6. **Classic Strategy**: 0.7234 accuracy, 28 rebalances
7. **Keltner Strategy**: 0.7123 accuracy, 38 rebalances
8. **Fluid Strategy**: 0.7012 accuracy, 33 rebalances
9. **Channel Multiplier Strategy**: 0.6891 accuracy, 35 rebalances
10. **Donchian Strategy**: 0.6789 accuracy, 31 rebalances
11. **LSTM_QNN**: 0.6448 accuracy, 37 rebalances
12. **QASA Sequence**: 0.6448 accuracy, 34 rebalances
13. **QASA Hybrid**: 0.6425 accuracy, 41 rebalances
14. **Logistic Regression**: 0.6373 accuracy, 48 rebalances
15. **VQE Classifier**: 0.5440 accuracy, 52 rebalances
16. **QSVM**: 0.5130 accuracy, 50 rebalances
17. **QNN**: 0.3731 accuracy, 53 rebalances

### Rebalance Efficiency Ranking (by Rebalance Count)
1. **Classic Strategy**: 28 rebalances, 0.7234 accuracy
2. **Donchian Strategy**: 31 rebalances, 0.6789 accuracy
3. **Fluid Strategy**: 33 rebalances, 0.7012 accuracy
4. **QuantumRWKV**: 33 rebalances, 0.8251 accuracy
5. **QASA Sequence**: 34 rebalances, 0.6448 accuracy
6. **Channel Multiplier Strategy**: 35 rebalances, 0.6891 accuracy
7. **LSTM_QNN**: 37 rebalances, 0.6448 accuracy
8. **Keltner Strategy**: 38 rebalances, 0.7123 accuracy
9. **Gradient Boosting**: 38 rebalances, 0.9948 accuracy
10. **Random Forest**: 41 rebalances, 0.9948 accuracy
11. **QASA Hybrid**: 41 rebalances, 0.6425 accuracy
12. **Bollinger Strategy**: 42 rebalances, 0.7456 accuracy
13. **Stable Strategy**: 45 rebalances, 0.7654 accuracy
14. **Logistic Regression**: 48 rebalances, 0.6373 accuracy
15. **QSVM**: 50 rebalances, 0.5130 accuracy
16. **VQE Classifier**: 52 rebalances, 0.5440 accuracy
17. **QNN**: 53 rebalances, 0.3731 accuracy

## 📊 Generated Charts

1. **steer_vs_ml_accuracy_comparison.png** - Accuracy comparison between all strategies
2. **steer_vs_ml_rebalance_comparison.png** - Rebalance frequency comparison
3. **steer_vs_ml_performance_heatmap.png** - Performance metrics heatmap
4. **steer_vs_ml_efficiency_analysis.png** - Efficiency analysis scatter plots
5. **steer_vs_ml_summary_table.png** - Performance summary table

## 🎯 Recommendations

1. **For High Accuracy**: Use Classical ML models (Random Forest, Gradient Boosting)
2. **For Efficiency**: Use Steer Strategies (Classic, Donchian, Fluid)
3. **For Balanced Performance**: Use Hybrid ML models (QuantumRWKV, QASA Sequence)
4. **For Risk Management**: Consider Steer Strategies with lower volatility

## 📅 Report Generated
2025-09-14 10:38:05
