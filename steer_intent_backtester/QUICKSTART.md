# Quick Start Guide

## Installation

### Option 1: Install from source
```bash
git clone <repository-url>
cd steer_intent_backtester
pip install -e .
```

### Option 2: Install dependencies manually
```bash
pip install -r requirements.txt
```

## Quick Test

Run the test suite to verify everything is working:
```bash
python run_tests.py
```

## Basic Usage

### 1. Fetch Data
```bash
python cli.py fetch --source binance --symbol ETHUSDC --interval 1h --start 2024-01-01 --end 2024-01-31 --out data/ETHUSDC_1h.csv
```

### 2. Run Backtest
```bash
python cli.py backtest --pair ETHUSDC --interval 1h --strategy bollinger --n 20 --k 2 --fee_bps 5 --liq_share 0.002
```

### 3. Generate Reports
```bash
python cli.py report --run_id <auto_id>
```

## Example Script

Run the comprehensive example:
```bash
python example.py
```

## Available Strategies

- **classic**: Configurable width and placement modes
- **bollinger**: Bollinger Bands (SMA ± k*Std)
- **keltner**: Keltner Channels (EMA ± m*ATR)
- **donchian**: Donchian Channels (HH/LL over N periods)
- **stable**: Multi-position around computed peg
- **fluid**: Maintain value ratio toward ideal_ratio

## Available Curves

- **linear**: Linear liquidity distribution
- **gaussian**: Gaussian (normal) distribution
- **sigmoid**: Sigmoid distribution
- **logarithmic**: Logarithmic distribution
- **bid_ask**: Twin peaks distribution
- **uniform**: Equal distribution

## CLI Commands

```bash
# List available strategies
python cli.py strategies

# List available curves
python cli.py curves

# Get help
python cli.py --help
python cli.py backtest --help
```

## Configuration

The system supports various parameters:
- **fee_bps**: Trading fees in basis points
- **slippage_bps**: Slippage in basis points
- **gas_cost**: Gas cost per rebalance
- **liq_share**: Liquidity share percentage
- **walkforward**: Enable walkforward analysis

## Output

Reports are generated in the `reports/` directory:
- Equity curves (PNG)
- Drawdown charts (PNG)
- LVR analysis (PNG)
- CSV data export
- Summary report (TXT)

## Troubleshooting

1. **Import errors**: Make sure you're in the project directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Data issues**: Check that your data file has the required columns
4. **Strategy errors**: Verify strategy parameters are valid

## Next Steps

1. Explore different strategies and parameters
2. Test with your own data
3. Implement custom strategies by extending BaseStrategy
4. Add new liquidity distribution curves
5. Integrate with on-chain data sources
