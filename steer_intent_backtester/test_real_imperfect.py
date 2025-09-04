#!/usr/bin/env python3
"""
Test script for real imperfect strategy implementation.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imperfect_strategy():
    """Test the imperfect strategy implementation."""
    print("🚀 Testing Real Imperfect Strategy Implementation")
    print("=" * 80)
    
    # Load ETHUSDC data
    data_file = "data/ETHUSDC_1h.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 3 months for testing
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=90)
    
    # Filter data
    data = data[data.index >= start_date]
    
    print(f"📈 Using limited data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Test configurations - using the same config as the CSV file that produced MDD
    test_configs = {
        'classic_perfect': {
            "strategy": "classic",
            "strategy_params": {
                "width_mode": "percent",
                "width_value": 5.0,  # Same as CSV file
                "placement_mode": "center",
                "curve_type": "uniform",
                "liquidity_scale": 0.0001,  # Same as CSV file
                "max_positions": 3,
                "gap_bps": 1000,
                "drift_threshold_pct": 20.0,
                "time_threshold_hours": 72
            }
        },
        'imperfect_classic_low': {
            "strategy": "imperfect_classic",
            "strategy_params": {
                "width_mode": "percent",
                "width_value": 5.0,  # Same as CSV file
                "placement_mode": "center",
                "curve_type": "uniform",
                "liquidity_scale": 0.0001,  # Same as CSV file
                "max_positions": 3,
                "gap_bps": 1000,
                "drift_threshold_pct": 20.0,
                "time_threshold_hours": 72,
                "imperfection_level": 0.3,
                "rebalance_failure_rate": 0.2,
                "forced_no_rebalance_chance": 0.15
            }
        },
        'imperfect_classic_medium': {
            "strategy": "imperfect_classic",
            "strategy_params": {
                "width_mode": "percent",
                "width_value": 5.0,  # Same as CSV file
                "placement_mode": "center",
                "curve_type": "uniform",
                "liquidity_scale": 0.0001,  # Same as CSV file
                "max_positions": 3,
                "gap_bps": 1000,
                "drift_threshold_pct": 20.0,
                "time_threshold_hours": 72,
                "imperfection_level": 0.6,
                "rebalance_failure_rate": 0.4,
                "forced_no_rebalance_chance": 0.3
            }
        },
        'imperfect_classic_high': {
            "strategy": "imperfect_classic",
            "strategy_params": {
                "width_mode": "percent",
                "width_value": 5.0,  # Same as CSV file
                "placement_mode": "center",
                "curve_type": "uniform",
                "liquidity_scale": 0.0001,  # Same as CSV file
                "max_positions": 3,
                "gap_bps": 1000,
                "drift_threshold_pct": 20.0,
                "time_threshold_hours": 72,
                "imperfection_level": 0.9,
                "rebalance_failure_rate": 0.6,
                "forced_no_rebalance_chance": 0.5
            }
        }
    }
    
    # Base configuration - using the same config as the CSV file
    config_base = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,  # Same as CSV file
        "slippage_bps": 1,  # Same as CSV file
        "gas_cost": 0.0,  # Same as CSV file
        "liq_share": 0.001,  # Same as CSV file
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # Run tests
    print("\n🔄 Running imperfect strategy tests...")
    print("-" * 80)
    
    all_results = {}
    all_summaries = {}
    
    for config_name, config in test_configs.items():
        print(f"\n📊 Testing {config_name}...")
        
        # Merge configs
        test_config = config_base.copy()
        test_config.update(config)
        
        try:
            # Run backtest
            backtester = Backtester(test_config)
            results = backtester.run(data)
            
            # Get summary
            summary = backtester.get_summary()
            
            # Check if we got realistic MDD
            mdd = summary.get('max_drawdown_pct', 0)
            if mdd < 0:
                print(f"✅ {config_name} produced realistic MDD: {mdd:.2f}%")
            elif mdd == 0:
                print(f"⚠️  {config_name} still has 0% MDD")
            else:
                print(f"✅ {config_name} MDD: {mdd:.2f}%")
            
            print(f"   Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"   Max DD: {mdd:.2f}%")
            print(f"   Sharpe: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"   Rebalances: {summary.get('rebalance_count', 0)}")
            
            all_results[config_name] = results
            all_summaries[config_name] = summary
            
        except Exception as e:
            print(f"❌ {config_name} failed: {e}")
            all_results[config_name] = None
            all_summaries[config_name] = None
    
    # Generate comparison chart
    print("\n📊 Generating comparison chart...")
    chart_file = plot_comparison(all_results)
    
    # Print summary
    print("\n📋 Results Summary")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Return %':<15} {'Max DD %':<15} {'Sharpe':<12} {'Rebalances':<15}")
    print("-" * 80)
    
    successful_count = 0
    mdd_strategies = 0
    
    for config_name, summary in all_summaries.items():
        if summary:
            return_pct = summary.get('total_return_pct', 0)
            max_dd = summary.get('max_drawdown_pct', 0)
            sharpe = summary.get('sharpe_ratio', 0)
            rebalances = summary.get('rebalance_count', 0)
            
            # Add indicators for MDD
            indicator = ""
            if max_dd < 0:
                indicator = " ✅"  # Has MDD
                mdd_strategies += 1
            elif max_dd == 0:
                indicator = " ⚠️"  # No MDD
            else:
                indicator = " ❌"  # Positive MDD (error)
            
            print(f"{config_name:<25} {return_pct:<15.2f} {max_dd:<15.2f} {sharpe:<12.2f} {rebalances:<15}{indicator}")
            successful_count += 1
        else:
            print(f"{config_name:<25} {'FAILED':<15} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
    
    print(f"\n📊 Final Results:")
    print(f"   Successfully tested: {successful_count}/{len(test_configs)} strategies")
    print(f"   Strategies with MDD: {mdd_strategies}/{successful_count}")
    print(f"   Chart saved: {chart_file}")
    
    # Final analysis
    print(f"\n💡 Analysis:")
    if mdd_strategies > 0:
        print(f"   ✅ Successfully generated {mdd_strategies} strategies with realistic MDD!")
        print(f"   🔍 The real logic modifications worked!")
    else:
        print(f"   ❌ No strategies generated realistic MDD")
        print(f"   🔍 May need further strategy logic modifications")

def plot_comparison(all_results, output_dir="reports"):
    """Plot comparison of all strategies."""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot equity curves
    successful_strategies = []
    for i, (strategy_name, results) in enumerate(all_results.items()):
        if results is None:
            continue
            
        equity_curves = results.get("equity_curves", {})
        if "strategy" in equity_curves:
            strategy_df = pd.DataFrame(equity_curves["strategy"])
            if not strategy_df.empty:
                strategy_df["timestamp"] = pd.to_datetime(strategy_df["timestamp"])
                
                # Plot equity curve
                ax1.plot(strategy_df["timestamp"], strategy_df["total_value"], 
                        label=strategy_name, linewidth=2.5, color=colors[i % len(colors)])
                
                # Plot drawdown
                if "drawdown" in strategy_df.columns:
                    ax2.plot(strategy_df["timestamp"], strategy_df["drawdown"], 
                            label=strategy_name, linewidth=2.5, color=colors[i % len(colors)])
                
                successful_strategies.append(strategy_name)
    
    # Customize charts
    ax1.set_title("Real Imperfect Strategy - MDD Generation Test", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Drawdown Comparison (MDD Focus)", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_imperfect_test_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison chart saved: {filepath}")
    return filepath

if __name__ == "__main__":
    test_imperfect_strategy()
