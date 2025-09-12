#!/usr/bin/env python3
"""
Debug script to analyze drawdown calculation issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

def debug_drawdown_calculation():
    """Debug drawdown calculation step by step."""
    print("🔍 Debugging Drawdown Calculation...")
    print("=" * 60)
    
    # Load minimal data for quick test
    data_file = "data/ETHUSDC_1h.csv"
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 7 days for debugging
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=7)
    data = data[data.index >= start_date]
    
    print(f"📊 Using debug data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Simple strategy configuration
    config = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.001,
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "strategy": "classic",
        "strategy_params": {
            "width_mode": "percent",
            "width_value": 5.0,
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.001
        }
    }
    
    try:
        print("\n🔄 Running backtest...")
        
        # Create backtester
        backtester = Backtester(config)
        
        # Run backtest
        results = backtester.run(data)
        
        # Get summary
        summary = backtester.get_summary()
        
        print(f"\n📊 Backtest Results:")
        print(f"   Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        
        # Debug equity curve data
        if results and "equity_curves" in results:
            strategy_equity = results["equity_curves"].get("strategy", [])
            if strategy_equity:
                df = pd.DataFrame(strategy_equity)
                print(f"\n📈 Equity Curve Data:")
                print(f"   Length: {len(df)}")
                print(f"   Columns: {list(df.columns)}")
                
                if "total_value" in df.columns:
                    values = df["total_value"]
                    print(f"\n💰 Value Analysis:")
                    print(f"   Initial Value: ${values.iloc[0]:.2f}")
                    print(f"   Final Value: ${values.iloc[-1]:.2f}")
                    print(f"   Min Value: ${values.min():.2f}")
                    print(f"   Max Value: ${values.max():.2f}")
                    
                    # Calculate running maximum
                    running_max = values.expanding().max()
                    print(f"\n📊 Running Maximum Analysis:")
                    print(f"   Initial Running Max: ${running_max.iloc[0]:.2f}")
                    print(f"   Final Running Max: ${running_max.iloc[-1]:.2f}")
                    
                    # Calculate drawdown manually
                    drawdowns = (values / running_max - 1) * 100
                    print(f"\n📉 Drawdown Analysis:")
                    print(f"   Min Drawdown: {drawdowns.min():.4f}%")
                    print(f"   Max Drawdown: {drawdowns.max():.4f}%")
                    print(f"   Drawdown Range: {drawdowns.min():.4f}% to {drawdowns.max():.4f}%")
                    
                    # Check if all values are increasing
                    is_monotonic = values.is_monotonic_increasing
                    print(f"\n🔍 Monotonic Check:")
                    print(f"   Is Always Increasing: {is_monotonic}")
                    
                    # Check for any decreases
                    decreases = (values.diff() < 0).sum()
                    print(f"   Number of Decreases: {decreases}")
                    
                    if decreases > 0:
                        print(f"   Decrease Indices: {values.diff()[values.diff() < 0].index.tolist()[:5]}")
                    
                    # Sample some values
                    print(f"\n📋 Sample Values (first 10):")
                    for i in range(min(10, len(values))):
                        print(f"   {i}: ${values.iloc[i]:.2f} (running max: ${running_max.iloc[i]:.2f}, DD: {drawdowns.iloc[i]:.4f}%)")
                    
                    # Check last 10 values
                    print(f"\n📋 Sample Values (last 10):")
                    for i in range(max(0, len(values)-10), len(values)):
                        print(f"   {i}: ${values.iloc[i]:.2f} (running max: ${running_max.iloc[i]:.2f}, DD: {drawdowns.iloc[i]:.4f}%)")
                        
        else:
            print("❌ No equity curve data found!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🚀 Starting Drawdown Debug Session")
    print("=" * 70)
    
    debug_drawdown_calculation()
    
    print("\n🔍 Debug session completed!")
    print("\n💡 Expected Findings:")
    print("   1. Should see actual drawdown values > 0%")
    print("   2. Should see some decreases in portfolio value")
    print("   3. Running maximum should not always equal current value")

if __name__ == "__main__":
    main()
