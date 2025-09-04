# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your ETH/USDC data in the `data/ETHUSDC/` directory:
- `price_1h.csv` (required) - OHLCV data
- `pool_1h.csv` (optional) - Pool metrics data

### 3. Run Quick Test
```bash
# Test with proxy fee mode (recommended for first run)
python run.py quick --pool ETHUSDC --freq 1h --fee-mode proxy

# Or use the script
chmod +x scripts/run_quick.sh
./scripts/run_quick.sh ETHUSDC 1h proxy
```

### 4. View Results
Check the generated files:
- `results/quick_test_summary.csv` - Strategy comparison
- `reports/figs/quick_equity_curves.png` - Performance charts
- `reports/figs/quick_apr_mdd_scatter.png` - Risk-return analysis

## 🔧 Available Commands

### Quick Test (60 days)
```bash
python run.py quick --pool ETHUSDC --freq 1h --fee-mode proxy
```

### Full Analysis (Walk-Forward + Optimization)
```bash
python run.py full --pool ETHUSDC --freq 1h --fee-mode exact --study-name ethusdc_wfa --n-trials 50
```

### Generate Reports Only
```bash
python run.py report --pool ETHUSDC
```

### List Available Pools
```bash
python run.py list-pools
```

## 📊 What You'll Get

### Quick Test Results
- **Strategy Comparison**: APR, MDD, Sharpe ratio, rebalance frequency
- **Basic Charts**: Equity curves, APR vs MDD scatter
- **CSV Export**: Summary statistics and trade details

### Full Analysis Results
- **Hyperparameter Optimization**: Best parameters for each strategy
- **Walk-Forward Analysis**: Out-of-sample validation
- **Comprehensive Charts**: All 7 chart types as specified
- **Statistical Analysis**: Significance tests and confidence intervals

## 🎯 Strategy Comparison

| Strategy | Type | Description | Expected Performance |
|----------|------|-------------|---------------------|
| Baseline-Static | Passive | Ultra-wide positions, minimal rebalancing | Lower fees, higher IL risk |
| Baseline-Fixed | Traditional | Fixed width + fixed triggers | Platform-style approach |
| Dynamic-Vol | Adaptive | Volatility-based width + price triggers | **Best APR/MDD balance** |
| Dynamic-Inventory | Smart | Inventory skew + fee density triggers | **Lowest MDD** |

## ⚠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory
2. **Data Not Found**: Check `data/ETHUSDC/` directory structure
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Permission Denied**: Make scripts executable with `chmod +x scripts/*.sh`

### Data Format Issues

- Ensure timestamps are in ISO8601 format or Unix timestamp
- Check that required columns exist in CSV files
- Verify data is sorted chronologically

## 🔍 Next Steps

After running the quick test:

1. **Review Results**: Check the generated charts and CSV files
2. **Adjust Parameters**: Modify `configs/experiment_default.yaml`
3. **Run Full Analysis**: Use the `full` command for comprehensive results
4. **Customize Strategies**: Add your own strategies in `src/strategies/`

## 📚 Learn More

- **README.md**: Complete project documentation
- **configs/**: Configuration files and examples
- **src/**: Source code structure and implementation
- **test_project.py**: Run tests to verify installation

## 🆘 Need Help?

1. Check the logs in `amm_backtest.log`
2. Run `python test_project.py` to verify installation
3. Review error messages in the console output
4. Check that all required files exist in the correct locations
