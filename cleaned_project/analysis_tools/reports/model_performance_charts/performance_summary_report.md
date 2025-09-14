# 模型性能比較摘要報告

## 📊 基本統計

- **總模型數量**: 8
- **模型類型**: Classic ML, Quantum ML, PennyLane, QASA Hybrid, Baseline
- **數據來源**: Mock_Data

## 🏷️ 模型類型分布

- **Classic ML**: 3 個模型
- **Baseline**: 2 個模型
- **Quantum ML**: 1 個模型
- **PennyLane**: 1 個模型
- **QASA Hybrid**: 1 個模型

## 📈 性能摘要

| 模型類型 | APR (%) | 夏普比率 | 最大回撤 (%) | 波動率 (%) |
|----------|---------|----------|--------------|------------|
| Baseline | 6.64 | 0.97 | 14.43 | 15.55 |
| Classic ML | 13.02 | 1.79 | 7.62 | 12.09 |
| PennyLane | 22.80 | 1.46 | 14.51 | 15.94 |
| QASA Hybrid | 16.80 | 2.19 | 5.42 | 8.73 |
| Quantum ML | 2.24 | 1.68 | 12.75 | 14.51 |

## 🏆 最佳表現模型

- **最高APR**: QNN (22.80%)
- **最高夏普比率**: QASA Benchmark (2.19)
- **最低回撤**: QASA Benchmark (5.42%)

## 📊 生成的圖表

1. **equity_curves_comparison.png** - 資金曲線比較圖
2. **apr_comparison.png** - APR比較圖
3. **risk_return_scatter.png** - 風險收益散點圖
4. **drawdown_analysis.png** - 回撤分析圖
5. **performance_heatmap.png** - 性能熱力圖
6. **comprehensive_dashboard.png** - 綜合儀表板

## 📁 文件說明

- **integrated_model_performance.csv** - 整合的性能數據
- **integrated_model_performance.json** - JSON格式的性能數據
- **performance_summary_report.md** - 本摘要報告
