#!/usr/bin/env python3
"""
Example usage of the Steer Intent Backtester.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample price data for demonstration."""
    # Generate 30 days of hourly data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data with trend and volatility
    base_price = 2000.0
    trend = 0.0001  # Slight upward trend
    volatility = 0.01  # 1% hourly volatility
    
    returns = np.random.normal(trend, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
        'high': prices * (1 + abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates)),
        'quote_volume': prices * np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    return data

def run_basic_backtest():
    """Run a basic backtest with Bollinger Bands strategy."""
    from steerbt.backtester import Backtester
    from steerbt.reports import ReportGenerator
    
    # Create sample data
    price_data = create_sample_data()
    print(f"Generated {len(price_data)} data points")
    
    # Configure backtest
    config = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "strategy": "bollinger",
        "strategy_params": {
            "n": 20,  # 20-period SMA
            "k": 2.0,  # 2 standard deviations
            "curve_type": "uniform",
            "max_positions": 1
        },
        "initial_cash": 10000.0,
        "fee_bps": 5,      # 0.05% trading fees
        "slippage_bps": 1, # 0.01% slippage
        "gas_cost": 0.0,   # No gas costs for demo
        "liq_share": 0.002 # 0.2% liquidity share
    }
    
    # Run backtest
    print("Running backtest...")
    backtester = Backtester(config)
    results = backtester.run(price_data)
    
    # Show summary
    summary = backtester.get_summary()
    print("\nBacktest Summary:")
    print(f"  Total Return: {summary.get('total_return_pct', 0):.2f}%")
    print(f"  Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  Rebalance Count: {summary.get('rebalance_count', 0)}")
    
    # Generate reports
    print("\nGenerating reports...")
    report_generator = ReportGenerator(results, "reports")
    report_files = report_generator.generate_all_reports()
    
    print("Reports generated:")
    for report_type, filepath in report_files.items():
        if filepath:
            print(f"  {report_type}: {filepath}")
    
    return results

def run_multiple_strategies():
    """Compare multiple strategies."""
    from steerbt.backtester import Backtester
    
    price_data = create_sample_data()
    strategies = [
        ("bollinger", {"n": 20, "k": 2.0}),
        ("keltner", {"n": 20, "m": 2.0}),
        ("donchian", {"n": 20}),
        ("classic", {"width_mode": "percent", "width_value": 10.0, "placement_mode": "center"})
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        print(f"\nTesting {strategy_name} strategy...")
        
        config = {
            "pair": "ETHUSDC",
            "interval": "1h",
            "strategy": strategy_name,
            "strategy_params": params,
            "initial_cash": 10000.0,
            "fee_bps": 5,
            "slippage_bps": 1,
            "gas_cost": 0.0,
            "liq_share": 0.002
        }
        
        backtester = Backtester(config)
        strategy_results = backtester.run(price_data)
        
        summary = backtester.get_summary()
        results[strategy_name] = {
            "total_return": summary.get('total_return_pct', 0),
            "max_drawdown": summary.get('max_drawdown_pct', 0),
            "sharpe_ratio": summary.get('sharpe_ratio', 0),
            "rebalance_count": summary.get('rebalance_count', 0)
        }
    
    # Compare results
    print("\nStrategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Return %':<10} {'Max DD %':<10} {'Sharpe':<10} {'Rebalances':<10}")
    print("-" * 80)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<15} {metrics['total_return']:<10.2f} {metrics['max_drawdown']:<10.2f} "
              f"{metrics['sharpe_ratio']:<10.2f} {metrics['rebalance_count']:<10}")
    
    return results

def demonstrate_curves():
    """Demonstrate different liquidity distribution curves."""
    from steerbt.curves import CurveFactory
    
    print("Liquidity Distribution Curves Demo")
    print("=" * 50)
    
    curves = ['linear', 'gaussian', 'sigmoid', 'logarithmic', 'bid_ask', 'uniform']
    center_price = 2000.0
    width_pct = 20.0
    total_liquidity = 10000.0
    
    for curve_type in curves:
        print(f"\n{curve_type.upper()} Curve:")
        curve = CurveFactory.create_curve(curve_type)
        distribution = curve.generate_distribution(center_price, width_pct, total_liquidity)
        
        print(f"  Positions: {len(distribution)}")
        for i, (lower, upper, liquidity) in enumerate(distribution):
            print(f"    Position {i+1}: {lower:.2f} - {upper:.2f}, Liquidity: {liquidity:.2f}")

def demonstrate_triggers():
    """Demonstrate rebalancing triggers."""
    from steerbt.triggers import (
        GapFromCenterTrigger, RangeInactiveTrigger, 
        PercentDriftTrigger, ElapsedTimeTrigger, TriggerManager
    )
    from datetime import timedelta
    
    print("Rebalancing Triggers Demo")
    print("=" * 40)
    
    # Create trigger manager
    trigger_manager = TriggerManager()
    
    # Add different triggers
    trigger_manager.add_trigger("gap", GapFromCenterTrigger(gap_bps=100))
    trigger_manager.add_trigger("range", RangeInactiveTrigger())
    trigger_manager.add_trigger("drift", PercentDriftTrigger(drift_threshold_pct=5.0))
    trigger_manager.add_trigger("time", ElapsedTimeTrigger(timedelta(hours=24)))
    
    # Test triggers
    current_state = {
        "current_price": 2000.0,
        "position_center": 1950.0,
        "lower_price": 1900.0,
        "upper_price": 2100.0,
        "position_value": 10000.0,
        "current_timestamp": datetime.now()
    }
    
    should_trigger, triggered_names = trigger_manager.should_trigger_any(current_state)
    
    print(f"Should trigger: {should_trigger}")
    if triggered_names:
        print(f"Triggered: {', '.join(triggered_names)}")
    
    # Show trigger stats
    stats = trigger_manager.get_trigger_stats()
    print("\nTrigger Statistics:")
    for name, stat in stats.items():
        print(f"  {name}: {stat}")

if __name__ == "__main__":
    print("Steer Intent Backtester - Example Usage")
    print("=" * 50)
    
    try:
        # Run basic backtest
        print("\n1. Running Basic Backtest")
        results = run_basic_backtest()
        
        # Compare strategies
        print("\n2. Comparing Multiple Strategies")
        strategy_results = run_multiple_strategies()
        
        # Demonstrate curves
        print("\n3. Liquidity Distribution Curves")
        demonstrate_curves()
        
        # Demonstrate triggers
        print("\n4. Rebalancing Triggers")
        demonstrate_triggers()
        
        print("\nExample completed successfully!")
        print("Check the 'reports' directory for generated charts and data.")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
