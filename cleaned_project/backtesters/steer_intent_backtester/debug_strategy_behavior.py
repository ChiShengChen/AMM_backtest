#!/usr/bin/env python3
"""
Debug script to check strategy behavior during price drops.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

def debug_strategy_behavior():
    """Debug strategy behavior during price drops."""
    print("🔍 Debugging Strategy Behavior During Price Drops...")
    print("=" * 70)
    
    # Load data
    data_file = "data/ETHUSDC_1h.csv"
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Find a period with significant price drops
    print("🔍 Looking for periods with significant price drops...")
    
    # Calculate hourly returns
    returns = data['close'].pct_change().dropna()
    
    # Find periods with significant drops
    significant_drops = returns[returns < -0.05]  # >5% drops
    
    if not significant_drops.empty:
        print(f"📉 Found {len(significant_drops)} significant drops (>5%)")
        
        # Use the largest drop period
        largest_drop = significant_drops.min()
        drop_time = significant_drops.idxmin()
        
        print(f"🚨 Largest drop: {largest_drop*100:.2f}% at {drop_time}")
        
        # Get data around this drop (1 day before and after)
        start_time = drop_time - timedelta(days=1)
        end_time = drop_time + timedelta(days=1)
        
        test_data = data[(data.index >= start_time) & (data.index <= end_time)]
        
        print(f"\n📊 Test period: {start_time} to {end_time}")
        print(f"   Records: {len(test_data)}")
        
        # Check price movement in this period
        if len(test_data) > 1:
            price_change = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
            print(f"   Total price change: {price_change:.2f}%")
            
            # Run strategy on this period
            print(f"\n🔄 Running strategy on drop period...")
            
            config = {
                "pair": "ETHUSDC",
                "interval": "1h",
                "initial_cash": 10000.0,
                "fee_bps": 5,
                "slippage_bps": 1,
                "gas_cost": 0.0,
                "liq_share": 0.001,
                "start_date": test_data.index[0],
                "end_date": test_data.index[-1],
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
                backtester = Backtester(config)
                results = backtester.run(test_data)
                summary = backtester.get_summary()
                
                print(f"\n📊 Strategy Results on Drop Period:")
                print(f"   Total Return: {summary.get('total_return_pct', 0):.2f}%")
                print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
                
                # Analyze equity curve
                if results and "equity_curves" in results:
                    strategy_equity = results["equity_curves"].get("strategy", [])
                    if strategy_equity:
                        df = pd.DataFrame(strategy_equity)
                        
                        if "total_value" in df.columns:
                            values = df["total_value"]
                            
                            print(f"\n📈 Equity Curve Analysis:")
                            print(f"   Initial Value: ${values.iloc[0]:.2f}")
                            print(f"   Final Value: ${values.iloc[-1]:.2f}")
                            print(f"   Min Value: ${values.min():.2f}")
                            print(f"   Max Value: ${values.max():.2f}")
                            
                            # Check for any decreases
                            decreases = (values.diff() < 0).sum()
                            print(f"   Number of Decreases: {decreases}")
                            
                            if decreases > 0:
                                print(f"   Decrease Indices: {values.diff()[values.diff() < 0].index.tolist()[:5]}")
                                
                                # Calculate drawdown manually
                                running_max = values.expanding().max()
                                drawdowns = (values / running_max - 1) * 100
                                min_dd = drawdowns.min()
                                print(f"   Manual Min Drawdown: {min_dd:.4f}%")
                            else:
                                print(f"   ⚠️  Still no decreases in portfolio value!")
                                
                                # Check if this is due to rebalancing
                                print(f"   Rebalance Count: {summary.get('rebalance_count', 0)}")
                                
                                # Check if strategy is too conservative
                                print(f"   ⚠️  Strategy might be too conservative (liquidity_scale: 0.001)")
                                
            except Exception as e:
                print(f"❌ Error running strategy: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Not enough data in test period")
    else:
        print("❌ No significant price drops found")

def main():
    """Main function."""
    print("🚀 Starting Strategy Behavior Debug")
    print("=" * 70)
    
    debug_strategy_behavior()
    
    print("\n🔍 Debug session completed!")
    print("\n💡 Expected Findings:")
    print("   1. Should see portfolio value decreases during price drops")
    print("   2. Should see non-zero drawdown during volatile periods")
    print("   3. Strategy should show realistic risk behavior")

if __name__ == "__main__":
    main()
