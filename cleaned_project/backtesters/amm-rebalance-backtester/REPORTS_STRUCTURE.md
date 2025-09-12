# 報告目錄結構說明

## 📁 圖表目錄組織

為了避免不同幣種的圖表互相覆蓋，我們現在按幣種分離圖表：

```
reports/figs/
├── ethusdc/          # ETH/USDC 圖表
│   ├── equity_curves.png
│   ├── apr_mdd_scatter.png
│   ├── fee_vs_price_pnl.png
│   ├── sensitivity_heatmap.png
│   ├── gas_frequency_contour.png
│   ├── il_curve.png
│   └── lvr_estimates.png
│
└── btcusdc/          # BTC/USDC 圖表
    ├── equity_curves.png
    ├── apr_mdd_scatter.png
    ├── fee_vs_price_pnl.png
    ├── sensitivity_heatmap.png
    ├── gas_frequency_contour.png
    ├── il_curve.png
    └── lvr_estimates.png
```

## 🔧 自動化腳本

### 1. 重新生成單個幣種圖表
```bash
# 重新生成 BTC/USDC 圖表
python regenerate_btcusdc_plots.py

# 重新生成 ETH/USDC 圖表  
python regenerate_ethusdc_plots.py
```

### 2. 重新生成所有幣種圖表
```bash
python generate_all_pool_plots.py
```

## 📊 圖表類型

每個幣種都包含以下 7 種圖表：

1. **equity_curves.png** - 淨值曲線比較
2. **apr_mdd_scatter.png** - 風險-收益散點圖
3. **fee_vs_price_pnl.png** - 費用 vs 價格 PnL 分析
4. **sensitivity_heatmap.png** - 參數敏感性熱圖
5. **gas_frequency_contour.png** - Gas 成本 vs 頻率分析
6. **il_curve.png** - 無常損失曲線
7. **lvr_estimates.png** - LVR 估算分析

## 🎯 幣種標識

每個圖表都包含：
- 標題中標明幣種（如 "Equity Curves Comparison - BTCUSDC Pool"）
- 右下角水印顯示幣種代碼
- 圖表顏色與策略保持一致

## 📈 數據來源

- **ETH/USDC**: 1小時頻率數據，來自 `data/ETHUSDC/price_1h.csv`
- **BTC/USDC**: 日線頻率數據，來自 `data/BTCUSDC/price_1d.csv`

## 🚀 未來擴展

當添加新的幣種時：
1. 創建對應的配置文件（如 `configs/newcoin_experiment.yaml`）
2. 在 `generate_all_pool_plots.py` 中添加新幣種
3. 圖表會自動保存到 `reports/figs/newcoin/` 目錄
