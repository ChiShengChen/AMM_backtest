#!/usr/bin/env python3
"""
Minimal test to verify if strategies are working at all.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

def test_minimal_strategy():
    """Test if strategy works at all."""
    print("🔍 Testing Minimal Strategy Functionality...")
    print("=" * 60)
    
    # Create very simple synthetic data
    print("📊 Creating synthetic test data...")
    
    # 10 hours of data with price going up and down
    timestamps = pd.date_range(start='2025-07-01', periods=10, freq='1H')
    prices = [100, 105, 95, 110, 90, 115, 85, 120, 80, 125]  # Volatile prices
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    })
    data = data.set_index('timestamp')
    
    print(f"📈 Test data created: {len(data)} records")
    print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(f"   Price volatility: {data['close'].pct_change().std()*100:.2f}%")
    
    # Very simple strategy config
    config = {
        "pair": "TEST",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 0,  # No fees
        "slippage_bps": 0,  # No slippage
        "gas_cost": 0.0,
        "liq_share": 0.5,  # High liquidity share
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "strategy": "classic",
        "strategy_params": {
            "width_mode": "percent",
            "width_value": 50.0,  # Very wide range: 50%
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 1.0,  # Full scaling
            "max_positions": 1
        }
    }
    
    try:
        print("\n🔄 Running minimal backtest...")
        
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
                    else:
                        print(f"   ⚠️  No decreases found - strategy too conservative")
                        
                    # Check positions
                    if "positions" in results:
                        positions = results["positions"]
                        print(f"\n📊 Position Data:")
                        print(f"   Position records: {len(positions) if positions else 0}")
                        
                        if positions:
                            pos_df = pd.DataFrame(positions)
                            print(f"   Position columns: {list(pos_df.columns)}")
                            
                            if "action" in pos_df.columns:
                                actions = pos_df["action"].value_counts()
                                print(f"   Actions: {dict(actions)}")
                        
        else:
            print("❌ No equity curve data found!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🚀 Starting Minimal Strategy Test")
    print("=" * 70)
    
    test_minimal_strategy()
    
    print("\n🔍 Test completed!")
    print("\n💡 Expected Findings:")
    print("   1. Strategy should create positions")
    print("   2. Portfolio value should change")
    print("   3. Should see some drawdown with volatile prices")

if __name__ == "__main__":
    main()
