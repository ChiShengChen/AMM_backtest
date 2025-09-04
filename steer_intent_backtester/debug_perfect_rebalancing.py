#!/usr/bin/env python3
"""
Debug script to check if strategy is "perfectly" rebalancing.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

def debug_perfect_rebalancing():
    """Debug if strategy is perfectly rebalancing."""
    print("🔍 Debugging Perfect Rebalancing Hypothesis...")
    print("=" * 70)
    
    # Create very volatile synthetic data
    print("📊 Creating highly volatile test data...")
    
    # 20 hours of data with extreme price movements
    timestamps = pd.date_range(start='2025-07-01', periods=20, freq='1h')
    
    # Create extreme volatility: price goes up and down dramatically
    base_price = 100
    prices = []
    for i in range(20):
        if i % 4 == 0:  # Every 4 hours, big move
            if i % 8 == 0:  # Every 8 hours, big up
                prices.append(base_price * 1.5)  # +50%
                base_price *= 1.5
            else:  # Big down
                prices.append(base_price * 0.7)  # -30%
                base_price *= 0.7
        else:  # Small random moves
            prices.append(base_price * (1 + np.random.uniform(-0.1, 0.1)))
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices),
        'quote_volume': [p * 1000 for p in prices]
    })
    data = data.set_index('timestamp')
    
    print(f"📈 Test data created: {len(data)} records")
    print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(f"   Price volatility: {data['close'].pct_change().std()*100:.2f}%")
    
    # Show price movements
    print(f"\n📊 Price Movements:")
    for i in range(min(10, len(prices))):
        if i > 0:
            change = (prices[i] / prices[i-1] - 1) * 100
            print(f"   Hour {i}: ${prices[i-1]:.2f} → ${prices[i]:.2f} ({change:+.1f}%)")
    
    # Test with strategy that should show drawdown
    config = {
        "pair": "TEST",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 0,  # No fees to isolate the issue
        "slippage_bps": 0,  # No slippage
        "gas_cost": 0.0,
        "liq_share": 0.1,  # High liquidity share
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "strategy": "classic",
        "strategy_params": {
            "width_mode": "percent",
            "width_value": 20.0,  # Wide range: 20%
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.1,  # High scaling
            "max_positions": 3,  # Multiple positions
            # Very conservative triggers to reduce rebalancing
            "gap_bps": 1000,  # 10% gap
            "drift_threshold_pct": 25.0,  # 25% drift
            "time_threshold_hours": 168  # 1 week
        }
    }
    
    try:
        print(f"\n🔄 Running backtest with conservative triggers...")
        
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
                    
                    # Check if values changed at all
                    if values.iloc[0] == values.iloc[-1]:
                        print(f"   ⚠️  Portfolio value never changed!")
                    else:
                        print(f"   ✅ Portfolio value changed: ${values.iloc[-1] - values.iloc[0]:.2f}")
                    
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
                    
                    # Check for any decreases
                    decreases = (values.diff() < 0).sum()
                    print(f"\n🔍 Decrease Check:")
                    print(f"   Number of Decreases: {decreases}")
                    
                    if decreases > 0:
                        print(f"   ✅ Found decreases in portfolio value!")
                        
                        # Show the decreases
                        decrease_data = values.diff()[values.diff() < 0]
                        print(f"\n📉 Portfolio Decreases:")
                        for idx, decrease in decrease_data.head(5).items():
                            prev_value = values.loc[idx - pd.Timedelta(hours=1)] if idx - pd.Timedelta(hours=1) in values.index else "N/A"
                            curr_value = values.loc[idx]
                            pct_change = (curr_value / prev_value - 1) * 100 if prev_value != "N/A" else 0
                            print(f"   {idx}: ${prev_value:.2f} → ${curr_value:.2f} ({pct_change:.2f}%)")
                    else:
                        print(f"   ⚠️  No decreases found - strategy still too perfect!")
                        
                        # Check if this is due to excessive rebalancing
                        print(f"   ⚠️  Strategy rebalanced {summary.get('rebalance_count', 0)} times in {len(data)} periods")
                        print(f"   ⚠️  Rebalance frequency: {summary.get('rebalance_count', 0) / len(data) * 100:.1f}%")
                        
                        # Check if the issue is in the strategy logic itself
                        print(f"   🔍 The strategy might be fundamentally flawed...")
                        
        else:
            print("❌ No equity curve data found!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🚀 Starting Perfect Rebalancing Debug")
    print("=" * 70)
    
    debug_perfect_rebalancing()
    
    print("\n🔍 Debug session completed!")
    print("\n💡 Expected Findings:")
    print("   1. With extreme volatility, should see portfolio decreases")
    print("   2. Should see non-zero drawdown")
    print("   3. If still 0% drawdown, strategy has fundamental issues")

if __name__ == "__main__":
    main()
