# Unified Label Training Report

## 🎯 Training Objective

All models were trained using the **AMM Baseline label standard** for fair comparison:

- **Label Definition**: Rebalance when price deviation from 20-period MA > 2%
- **Label Formula**: `y = 1 if |price/MA_20 - 1| > 0.02 else 0`
- **Threshold**: 2% price deviation
- **Problem Type**: Binary classification

## 📊 Model Performance Results

### Classical Machine Learning Models

| Model | Accuracy | Type |
|-------|----------|------|
| Random Forest | 0.9948 | Classical |
| Gradient Boosting | 0.9948 | Classical |
| Logistic Regression | 0.6373 | Classical |

### Quantum Machine Learning Models

| Model | Accuracy | Type |
|-------|----------|------|
| VQE Classifier | 0.5440 | Quantum |
| QNN | 0.3731 | Quantum |
| QSVM | 0.5130 | Quantum |
| QASA Hybrid | 0.6425 | Quantum |
| QuantumRWKV | 0.8251 | Quantum |
| LSTM_QNN | 0.6448 | Quantum |
| QASA Sequence | 0.6448 | Quantum |

## 🔍 Key Findings

1. **Best Performing Model**: Random Forest (Accuracy: 0.9948)
2. **Average Classical ML Accuracy**: 0.8756
3. **Average Quantum ML Accuracy**: 0.5982
4. **Classical ML performs better** by 0.2775 on average

## 📈 Generated Charts

1. **accuracy_comparison.png** - Model accuracy comparison
2. **confusion_matrices.png** - Confusion matrices for all models
3. **feature_importance.png** - Feature importance for classical models
4. **performance_summary.png** - Classical vs Quantum comparison

## ✅ Conclusion

By using unified AMM Baseline labels, we achieved fair comparison between all models.
All models now solve the same binary classification problem with identical evaluation criteria.
