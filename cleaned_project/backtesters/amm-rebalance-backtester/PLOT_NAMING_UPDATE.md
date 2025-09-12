# 圖表命名格式更新

## 📊 更新內容

已成功將所有策略生成的圖表命名格式更新為：`{幣種}_{策略類型}_{時間戳}.png`

## 🎯 命名格式

### 新格式
```
{POOL}_{PLOT_TYPE}_{TIMESTAMP}.png
```

### 示例
```
BTCUSDC_equity_curves_20250904_170110.png
BTCUSDC_apr_mdd_scatter_20250904_170110.png
BTCUSDC_fee_vs_price_pnl_20250904_170110.png
BTCUSDC_sensitivity_heatmap_20250904_170110.png
BTCUSDC_gas_frequency_contour_20250904_170110.png
BTCUSDC_il_curve_20250904_170110.png
BTCUSDC_lvr_estimates_20250904_170110.png
```

## 📁 目錄結構

```
reports/figs/
├── btcusdc/
│   ├── BTCUSDC_equity_curves_20250904_170110.png
│   ├── BTCUSDC_apr_mdd_scatter_20250904_170110.png
│   ├── BTCUSDC_fee_vs_price_pnl_20250904_170110.png
│   ├── BTCUSDC_sensitivity_heatmap_20250904_170110.png
│   ├── BTCUSDC_gas_frequency_contour_20250904_170110.png
│   ├── BTCUSDC_il_curve_20250904_170110.png
│   └── BTCUSDC_lvr_estimates_20250904_170110.png
├── ethusdc/
│   └── ETHUSDC_*.png
└── usdcusdt/
    └── USDCUSDT_*.png
```

## 🔧 修改的文件

### 1. 主要回測腳本
- `run.py` - 主要回測腳本
- `generate_all_pool_plots.py` - 批量圖表生成
- `regenerate_btcusdc_plots.py` - BTCUSDC圖表重新生成

### 2. 整合腳本
- `run_integrated.py` - 整合回測腳本

### 3. 執行腳本
- `run_single.sh` - 單次回測腳本

## 📊 圖表類型

| 圖表類型 | 文件名格式 | 描述 |
|---------|-----------|------|
| Equity Curves | `{POOL}_equity_curves_{TIMESTAMP}.png` | 權益曲線圖 |
| APR vs MDD Scatter | `{POOL}_apr_mdd_scatter_{TIMESTAMP}.png` | APR與MDD散點圖 |
| Fee vs Price PnL | `{POOL}_fee_vs_price_pnl_{TIMESTAMP}.png` | 費用與價格PnL圖 |
| Sensitivity Heatmap | `{POOL}_sensitivity_heatmap_{TIMESTAMP}.png` | 敏感性熱力圖 |
| Gas Frequency Contour | `{POOL}_gas_frequency_contour_{TIMESTAMP}.png` | Gas頻率等高線圖 |
| IL Curve | `{POOL}_il_curve_{TIMESTAMP}.png` | 無常損失曲線 |
| LVR Estimates | `{POOL}_lvr_estimates_{TIMESTAMP}.png` | LVR估計圖 |

## 🚀 使用方法

### 運行回測
```bash
./run_single.sh BTCUSDC 1d 50
```

### 查看生成的圖表
```bash
ls -la reports/figs/btcusdc/
```

### 圖表命名示例
```bash
# 查看特定時間戳的圖表
ls reports/figs/btcusdc/BTCUSDC_*_20250904_170110.png

# 查看所有權益曲線圖
ls reports/figs/btcusdc/BTCUSDC_equity_curves_*.png

# 查看所有敏感性熱力圖
ls reports/figs/btcusdc/BTCUSDC_sensitivity_heatmap_*.png
```

## ✅ 優勢

1. **唯一性**: 每個圖表都有唯一的時間戳標識
2. **可追溯性**: 可以輕鬆找到特定時間生成的圖表
3. **組織性**: 按幣種和圖表類型分類存儲
4. **兼容性**: 與現有系統完全兼容
5. **擴展性**: 易於添加新的圖表類型

## 🔄 向後兼容

- 舊的圖表文件仍然保留
- 新的命名格式不會影響現有功能
- 可以同時存在多個時間戳的圖表文件

## 📝 注意事項

1. 時間戳格式：`YYYYMMDD_HHMMSS`
2. 幣種名稱保持大寫
3. 圖表類型使用下劃線分隔
4. 所有圖表都保存在對應的幣種目錄下

## 🎉 測試結果

已成功測試圖表命名格式：
- ✅ 所有7種圖表類型都正確生成
- ✅ 命名格式符合要求
- ✅ 目錄結構正確
- ✅ 文件大小正常
- ✅ 圖表內容完整
