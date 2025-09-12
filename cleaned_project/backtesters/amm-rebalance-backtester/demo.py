#!/usr/bin/env python3
"""
AMM Rebalance Backtester Demo Script

This script demonstrates the key features of the AMM backtester.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.strategies.baseline_static import BaselineStaticStrategy
from src.strategies.baseline_fixed import BaselineFixedStrategy
from src.strategies.dyn_vol import DynamicVolatilityStrategy
from src.strategies.dyn_inventory import DynamicInventoryStrategy
from src.core.math_v3 import UniswapV3Math
from src.core.frictions import FrictionModel

def demo_strategies():
    """Demonstrate different strategies."""
    print("🚀 AMM Dynamic Rebalancing Strategy Demo")
    print("=" * 60)
    
    # Initialize strategies
    strategies = {
        'Baseline-Static': BaselineStaticStrategy(width_pct=500.0, rebalance_cooldown_hours=168),
        'Baseline-Fixed': BaselineFixedStrategy(width_pct=50.0, price_deviation_bps=50, rebalance_cooldown_hours=24),
        'Dynamic-Vol': DynamicVolatilityStrategy(vol_estimator="ewma", k_width=1.5, price_deviation_bps=50, rebalance_cooldown_hours=24),
        'Dynamic-Inventory': DynamicInventoryStrategy(skew_threshold_pct=15.0, fee_density_window_h=24, reinvest_frequency_h=48)
    }
    
    # Display strategy information
    for name, strategy in strategies.items():
        print(f"\n📊 {name}")
        print("-" * 40)
        info = strategy.get_strategy_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return strategies

def demo_uniswap_math():
    """Demonstrate Uniswap V3 math calculations."""
    print("\n🔢 Uniswap V3 Math Demo")
    print("=" * 40)
    
    # Test price conversions
    price = 2500.0
    sqrt_price_x96 = UniswapV3Math.price_to_sqrt_price_x96(price)
    converted_price = UniswapV3Math.sqrt_price_x96_to_price(sqrt_price_x96)
    
    print(f"Original price: ${price:,.2f}")
    print(f"Sqrt price X96: {sqrt_price_x96}")
    print(f"Converted price: ${converted_price:,.2f}")
    print(f"Accuracy: {abs(price - converted_price):.6f}")
    
    # Test position calculations
    print(f"\nPosition calculations for ETH at ${price:,.2f}:")
    ranges = [(2400.0, 2600.0), (2450.0, 2550.0), (2480.0, 2520.0)]
    
    for i, (lower, upper) in enumerate(ranges):
        width_pct = ((upper - lower) / price) * 100
        print(f"  Range {i+1}: ${lower:,.2f} - ${upper:,.2f} (Width: {width_pct:.1f}%)")
    
    return price, ranges

def demo_friction_costs():
    """Demonstrate friction cost modeling."""
    print("\n💰 Friction Costs Demo")
    print("=" * 40)
    
    friction_model = FrictionModel(gas_cost=5.0, slippage_bps=1.0)
    
    # Gas costs
    operations = ['add_liquidity', 'remove_liquidity', 'swap', 'rebalance']
    for op in operations:
        gas_cost = friction_model.calculate_gas_cost(op)
        print(f"  {op}: ${gas_cost:.2f}")
    
    # Slippage costs
    trade_sizes = [1000, 10000, 100000]  # USD
    price = 2500.0
    
    print(f"\nSlippage costs at ${price:,.2f}:")
    for size in trade_sizes:
        slippage = friction_model.calculate_slippage(size, price)
        slippage_pct = (slippage / size) * 100
        print(f"  ${size:,} trade: ${slippage:.2f} ({slippage_pct:.3f}%)")
    
    return friction_model

def demo_strategy_comparison():
    """Demonstrate strategy comparison."""
    print("\n📈 Strategy Performance Comparison")
    print("=" * 50)
    
    # Simulated performance data
    performance_data = {
        'Strategy': ['Baseline-Static', 'Baseline-Fixed', 'Dynamic-Vol', 'Dynamic-Inventory'],
        'APR (%)': [5.2, 8.1, 12.3, 14.7],
        'MDD (%)': [15.2, 12.8, 9.5, 8.2],
        'Sharpe': [0.8, 1.2, 1.8, 2.1],
        'Calmar': [0.34, 0.63, 1.29, 1.79],
        'Rebalances': [2, 15, 28, 35],
        'Gas Cost ($)': [10, 75, 140, 175]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Display table
    print(df.to_string(index=False))
    
    # Analysis
    print(f"\n🏆 Best Strategy by Metric:")
    print(f"  Highest APR: {df.loc[df['APR (%)'].idxmax(), 'Strategy']} ({df['APR (%)'].max():.1f}%)")
    print(f"  Lowest MDD: {df.loc[df['MDD (%)'].idxmin(), 'Strategy']} ({df['MDD (%)'].min():.1f}%)")
    print(f"  Best Sharpe: {df.loc[df['Sharpe'].idxmax(), 'Strategy']} ({df['Sharpe'].max():.2f})")
    print(f"  Best Calmar: {df.loc[df['Calmar'].idxmax(), 'Strategy']} ({df['Calmar'].max():.2f})")
    
    return df

def demo_il_lvr_analysis():
    """Demonstrate IL and LVR analysis."""
    print("\n📊 IL and LVR Analysis Demo")
    print("=" * 40)
    
    # Price ratios for IL calculation
    price_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    print("Impermanent Loss Analysis:")
    print("  Price Ratio | IL (%) | Description")
    print("  ------------|---------|-------------")
    
    for ratio in price_ratios:
        # Simplified IL calculation (v2 formula)
        il_pct = (2 * np.sqrt(ratio) / (1 + ratio) - 1) * 100
        if ratio < 1.0:
            desc = "Token0 depreciated"
        elif ratio > 1.0:
            desc = "Token0 appreciated"
        else:
            desc = "No change"
        
        print(f"  {ratio:>11.1f} | {il_pct:>6.2f} | {desc}")
    
    # LVR estimation
    print(f"\nLVR (Loss Versus Rebalancing) Estimation:")
    print(f"  Passive strategy typically loses 0.1-0.5% vs active rebalancing")
    print(f"  Higher in volatile markets and wide positions")
    print(f"  Lower with frequent rebalancing and narrow positions")
    
    return price_ratios

def main():
    """Run the complete demo."""
    print("🎯 AMM Dynamic Rebalancing Backtester - Complete Demo")
    print("=" * 70)
    
    try:
        # Run all demos
        strategies = demo_strategies()
        price, ranges = demo_uniswap_math()
        friction_model = demo_friction_costs()
        performance_df = demo_strategy_comparison()
        price_ratios = demo_il_lvr_analysis()
        
        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("\n📋 Summary:")
        print(f"  • {len(strategies)} strategies implemented")
        print(f"  • Uniswap V3 math functions working")
        print(f"  • Friction cost modeling active")
        print(f"  • Performance comparison generated")
        print(f"  • IL/LVR analysis completed")
        
        print("\n🚀 Next Steps:")
        print("  1. Run 'python run.py quick --pool ETHUSDC' for quick test")
        print("  2. Run 'python run.py full --pool ETHUSDC' for full analysis")
        print("  3. Check 'results/' and 'reports/figs/' for outputs")
        print("  4. Modify 'configs/experiment_default.yaml' for customization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
