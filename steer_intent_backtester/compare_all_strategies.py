#!/usr/bin/env python3
"""
Compare all preset strategies on ETHUSDC data.
This script runs backtests for all available strategies and plots the results together.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester
from steerbt.reports import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_strategy_backtest(strategy_name, strategy_params, data, config_base):
    """Run backtest for a specific strategy."""
    logger.info(f"Running {strategy_name} strategy...")
    
    # Create strategy-specific config
    config = config_base.copy()
    config["strategy"] = strategy_name
    config["strategy_params"] = strategy_params
    
    try:
        # Run backtest
        backtester = Backtester(config)
        results = backtester.run(data)
        
        # Get summary
        summary = backtester.get_summary()
        
        logger.info(f"✅ {strategy_name} completed: {summary.get('total_return_pct', 0):.2f}% return, {summary.get('max_drawdown_pct', 0):.2f}% max DD")
        
        # Debug drawdown calculation
        if summary.get('max_drawdown_pct', 0) == 0:
            logger.warning(f"⚠️  {strategy_name} has 0% drawdown - this seems suspicious!")
            # Check equity curve data
            if results and "equity_curves" in results:
                strategy_equity = results["equity_curves"].get("strategy", [])
                if strategy_equity:
                    df = pd.DataFrame(strategy_equity)
                    if "total_value" in df.columns:
                        values = df["total_value"]
                        running_max = values.expanding().max()
                        drawdowns = (values / running_max - 1) * 100
                        min_dd = drawdowns.min()
                        logger.info(f"   Debug: min drawdown from equity curve: {min_dd:.4f}%")
                        logger.info(f"   Debug: equity curve length: {len(values)}")
                        logger.info(f"   Debug: value range: {values.min():.2f} to {values.max():.2f}")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"❌ {strategy_name} failed: {e}")
        return None, None

def plot_all_strategies_comparison(all_results, output_dir="reports"):
    """Plot all strategies' equity curves on the same chart."""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color palette for strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot equity curves
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
                        label=strategy_name, linewidth=2, color=colors[i % len(colors)])
                
                # Plot drawdown
                if "drawdown" in strategy_df.columns:
                    ax2.plot(strategy_df["timestamp"], strategy_df["drawdown"], 
                            label=strategy_name, linewidth=2, color=colors[i % len(colors)])
    
    # Customize equity chart
    ax1.set_title("All Strategies - Equity Curves Comparison", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    # Format x-axis - 减少标签密度
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个标签
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Customize drawdown chart
    ax2.set_title("All Strategies - Drawdown Comparison", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(True, alpha=0.3)
    # Format x-axis - 减少标签密度
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个标签
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add performance summary table
    summary_data = []
    for strategy_name, results in all_results.items():
        if results is not None:
            summary = results.get("performance", {})
            summary_data.append({
                "Strategy": strategy_name,
                "Return (%)": f"{summary.get('total_return_pct', 0):.2f}",
                "Max DD (%)": f"{summary.get('max_drawdown_pct', 0):.2f}",
                "Sharpe": f"{summary.get('sharpe_ratio', 0):.2f}",
                "Rebalances": summary.get('rebalance_count', 0)
            })
    
    if summary_data:
        # Create summary table
        summary_df = pd.DataFrame(summary_data)
        
        # Add table to the plot
        table_ax = fig.add_axes([0.02, 0.02, 0.4, 0.15])
        table_ax.axis('tight')
        table_ax.axis('off')
        
        table = table_ax.table(cellText=summary_df.values, 
                              colLabels=summary_df.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color header row
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_strategies_comparison_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison chart saved: {filepath}")
    return filepath

def main():
    """Main function to run all strategy comparisons."""
    print("🚀 Starting All Strategies Comparison on ETHUSDC")
    print("=" * 60)
    
    # Load ETHUSDC data
    data_file = "data/ETHUSDC_1h.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run: python cli.py fetch --source binance --symbol ETHUSDC --interval 1h --start 2024-01-01 --end 2024-01-31 --out data/ETHUSDC_1h.csv")
        return
    
    print(f"📊 Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    print(f"📈 Data loaded: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Define all strategies and their parameters with more conservative settings
    strategies = {
        'classic': {
            "width_mode": "percent",
            "width_value": 5.0,  # Reduced from 10% to 5%
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.0001  # More conservative scaling
        },
        'channel_multiplier': {
            "width_pct": 5.0,  # Reduced from 10% to 5%
            "liquidity_scale": 0.0001
        },
        'bollinger': {
            "n": 20,
            "k": 1.5,  # Reduced from 2.0 to 1.5
            "liquidity_scale": 0.0001
        },
        'keltner': {
            "n": 20,
            "m": 1.5,  # Reduced from 2.0 to 1.5
            "liquidity_scale": 0.0001
        },
        'donchian': {
            "n": 20,
            "width_multiplier": 0.8,  # Reduced from 1.0 to 0.8
            "liquidity_scale": 0.0001
        },
        'stable': {
            "peg_method": "sma",
            "peg_period": 20,
            "width_pct": 10.0,  # Reduced from 15% to 10%
            "curve_type": "gaussian",
            "bin_count": 3,  # Reduced from 5 to 3
            "liquidity_scale": 0.0001
        },
        'fluid': {
            "ideal_ratio": 1.0,
            "acceptable_ratio": 0.15,  # Increased from 0.1 to 0.15
            "sprawl_type": "dynamic",
            "tail_weight": 0.15,  # Reduced from 0.2 to 0.15
            "liquidity_scale": 0.0001
        }
    }
    
    # Base configuration with more conservative settings
    config_base = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.001,  # Reduced from 0.002 to 0.001
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # Run all strategies
    print("\n🔄 Running backtests for all strategies...")
    print("-" * 60)
    
    all_results = {}
    all_summaries = {}
    
    for strategy_name, strategy_params in strategies.items():
        results, summary = run_strategy_backtest(strategy_name, strategy_params, data, config_base)
        all_results[strategy_name] = results
        all_summaries[strategy_name] = summary
    
    # Generate comparison chart
    print("\n📊 Generating comparison chart...")
    chart_file = plot_all_strategies_comparison(all_results)
    
    # Print summary table
    print("\n📋 Strategy Performance Summary")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Return %':<12} {'Max DD %':<12} {'Sharpe':<10} {'Rebalances':<12}")
    print("-" * 80)
    
    for strategy_name, summary in all_summaries.items():
        if summary:
            return_pct = summary.get('total_return_pct', 0)
            max_dd = summary.get('max_drawdown_pct', 0)
            sharpe = summary.get('sharpe_ratio', 0)
            rebalances = summary.get('rebalance_count', 0)
            
            print(f"{strategy_name:<20} {return_pct:<12.2f} {max_dd:<12.2f} {sharpe:<10.2f} {rebalances:<12}")
    
    print("\n🎉 All strategies comparison completed!")
    print(f"📊 Comparison chart saved: {chart_file}")
    
    # Save detailed results to CSV
    csv_file = os.path.join("reports", f"all_strategies_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    summary_data = []
    for strategy_name, summary in all_summaries.items():
        if summary:
            summary_data.append({
                "strategy": strategy_name,
                "total_return_pct": summary.get('total_return_pct', 0),
                "max_drawdown_pct": summary.get('max_drawdown_pct', 0),
                "sharpe_ratio": summary.get('sharpe_ratio', 0),
                "rebalance_count": summary.get('rebalance_count', 0),
                "total_fees_paid": summary.get('total_fees_paid', 0),
                "final_value": summary.get('final_value', 0)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"📄 Detailed results saved: {csv_file}")

if __name__ == "__main__":
    main()
