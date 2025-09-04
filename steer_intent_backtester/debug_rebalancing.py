#!/usr/bin/env python3
"""
Debug script to analyze strategy rebalancing behavior.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

def debug_rebalancing_behavior():
    """Debug strategy rebalancing behavior."""
    print("🔍 Debugging Strategy Rebalancing Behavior...")
    print("=" * 70)
    
    # Load minimal data for quick test
    data_file = "data/ETHUSDC_1h.csv"
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 3 days for detailed debugging
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=3)
    data = data[data.index >= start_date]
    
    print(f"📊 Using debug data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Test with very aggressive settings
    config = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.1,  # Very high liquidity share
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "strategy": "classic",
        "strategy_params": {
            "width_mode": "percent",
            "width_value": 20.0,  # Very wide range: 20%
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.1  # Very aggressive scaling
        }
    }
    
    try:
        print("\n🔄 Running backtest with very aggressive settings...")
        
        # Create backtester
        backtester = Backtester(config)
        
        # Run backtest
        results = backtester.run(data)
        
        # Get summary
        summary = backtester.get_summary()
        
        print(f"\n📊 Backtest Results:")
        print(f"   Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        print(f"   Rebalance Count: {summary.get('rebalance_count', 0)}")
        
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
                        
                        # Show the actual decreases
                        decrease_data = values.diff()[values.diff() < 0]
                        print(f"\n📉 Actual Decreases:")
                        for idx, decrease in decrease_data.head(5).items():
                            prev_value = values.loc[idx - pd.Timedelta(hours=1)] if idx - pd.Timedelta(hours=1) in values.index else "N/A"
                            curr_value = values.loc[idx]
                            pct_change = (curr_value / prev_value - 1) * 100 if prev_value != "N/A" else 0
                            print(f"   {idx}: ${prev_value:.2f} → ${curr_value:.2f} ({pct_change:.2f}%)")
                    else:
                        print(f"   ⚠️  Still no decreases in portfolio value!")
                        
                        # Check if this is due to excessive rebalancing
                        print(f"   ⚠️  Strategy rebalanced {summary.get('rebalance_count', 0)} times in {len(data)} periods")
                        print(f"   ⚠️  Rebalance frequency: {summary.get('rebalance_count', 0) / len(data) * 100:.1f}%")
                        
                        # Check if strategy is too aggressive
                        print(f"   ⚠️  Strategy might be too aggressive (liquidity_scale: 0.1, width: 20%)")
                        
                        # Check if the issue is in the strategy logic itself
                        print(f"   🔍 Checking if the issue is in strategy logic...")
                        
                        # Let's look at the actual strategy behavior
                        if "positions" in results:
                            positions = results["positions"]
                            print(f"   📊 Position data available: {len(positions) if positions else 0} records")
                            
                            if positions:
                                pos_df = pd.DataFrame(positions)
                                print(f"   📊 Position columns: {list(pos_df.columns)}")
                                
                                if "action" in pos_df.columns:
                                    actions = pos_df["action"].value_counts()
                                    print(f"   📊 Actions taken: {dict(actions)}")
                                
                                if "timestamp" in pos_df.columns:
                                    pos_df["timestamp"] = pd.to_datetime(pos_df["timestamp"])
                                    print(f"   📊 Position timestamps: {pos_df['timestamp'].min()} to {pos_df['timestamp'].max()}")
                        
        else:
            print("❌ No equity curve data found!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🚀 Starting Rebalancing Behavior Debug Session")
    print("=" * 70)
    
    debug_rebalancing_behavior()
    
    print("\n🔍 Debug session completed!")
    print("\n💡 Expected Findings:")
    print("   1. Should see portfolio value decreases with aggressive settings")
    print("   2. Should see non-zero drawdown with wider ranges")
    print("   3. Strategy should show realistic risk behavior")

if __name__ == "__main__":
    main()
