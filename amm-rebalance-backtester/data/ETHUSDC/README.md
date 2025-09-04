# ETHUSDC Data Directory

This directory should contain the following files:

## Required Files

### price_1h.csv
Price data with hourly frequency. Must contain these columns:
- `timestamp`: ISO8601 format or Unix timestamp
- `open`: Opening price
- `high`: High price  
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

## Optional Files

### pool_1h.csv
Pool data with hourly frequency. May contain:
- `timestamp`: ISO8601 format or Unix timestamp
- `sqrtPriceX96`: Square root of price in X96 format
- `tick`: Current tick
- `liquidity`: Total liquidity
- `volume_token0`: Volume of token0
- `volume_token1`: Volume of token1
- `fee_growth_global0X128`: Fee growth for token0
- `fee_growth_global1X128`: Fee growth for token1

## Data Format Examples

### price_1h.csv
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,2500.50,2510.25,2495.75,2508.90,1250.5
2024-01-01T01:00:00Z,2508.90,2515.30,2505.20,2512.45,1180.3
```

### pool_1h.csv
```csv
timestamp,sqrtPriceX96,tick,liquidity,volume_token0,volume_token1
2024-01-01T00:00:00Z,158456325028528675187087900672,194615,500000000000000000000,125.5,314375.0
```

## Notes

- If `pool_1h.csv` is missing or incomplete, the system will use proxy fee calculation mode
- Timestamps should be in UTC timezone
- Data should be sorted chronologically
- Missing values will be handled automatically with forward/backward fill
