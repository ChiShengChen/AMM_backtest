#!/usr/bin/env python3
"""
Improved strategy comparison script with better liquidity scaling and error handling.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester
from steerbt.reports import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_strategy_backtest(strategy_name, strategy_params, data, config_base):
    """Run backtest for a specific strategy with better error handling."""
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
        
        # Validate results
        if summary and summary.get('total_return_pct', 0) > 10000:  # > 1000% is suspicious
            logger.warning(f"⚠️  {strategy_name} has unusually high return: {summary.get('total_return_pct', 0):.2f}%")
        
        logger.info(f"✅ {strategy_name} completed: {summary.get('total_return_pct', 0):.2f}% return, {summary.get('max_drawdown_pct', 0):.2f}% max DD")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"❌ {strategy_name} failed: {e}")
        return None, None

def plot_all_strategies_comparison(all_results, output_dir="reports"):
    """Plot all strategies' equity curves on the same chart with better formatting."""
    
    # Create figure with better layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    
    # Color palette for strategies
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
    
    # Customize equity chart
    ax1.set_title("All Strategies - Equity Curves Comparison (ETHUSDC)", fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=14)
    ax1.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Customize drawdown chart
    ax2.set_title("All Strategies - Drawdown Comparison (ETHUSDC)", fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel("Date", fontsize=14)
    ax2.set_ylabel("Drawdown (%)", fontsize=14)
    ax2.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.02, 0.02))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
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
        
        # Add table to the plot with better positioning
        table_ax = fig.add_axes([0.02, 0.02, 0.45, 0.12])
        table_ax.axis('tight')
        table_ax.axis('off')
        
        table = table_ax.table(cellText=summary_df.values, 
                              colLabels=summary_df.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color header row
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternate rows for better readability
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    # Add note about successful strategies
    if successful_strategies:
        note_text = f"Successfully tested strategies: {', '.join(successful_strategies)}"
        fig.text(0.02, 0.95, note_text, fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_strategies_comparison_v2_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison chart saved: {filepath}")
    return filepath

def main():
    """Main function to run improved strategy comparisons."""
    print("🚀 Starting Improved All Strategies Comparison on ETHUSDC")
    print("=" * 70)
    
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
    
    # Use more conservative liquidity scaling for all strategies
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
    print("\n🔄 Running backtests for all strategies with conservative settings...")
    print("-" * 70)
    
    all_results = {}
    all_summaries = {}
    
    for strategy_name, strategy_params in strategies.items():
        results, summary = run_strategy_backtest(strategy_name, strategy_params, data, config_base)
        all_results[strategy_name] = results
        all_summaries[strategy_name] = summary
    
    # Generate comparison chart
    print("\n📊 Generating improved comparison chart...")
    chart_file = plot_all_strategies_comparison(all_results)
    
    # Print summary table
    print("\n📋 Strategy Performance Summary (Conservative Settings)")
    print("=" * 90)
    print(f"{'Strategy':<20} {'Return %':<15} {'Max DD %':<15} {'Sharpe':<12} {'Rebalances':<15}")
    print("-" * 90)
    
    successful_count = 0
    for strategy_name, summary in all_summaries.items():
        if summary:
            return_pct = summary.get('total_return_pct', 0)
            max_dd = summary.get('max_drawdown_pct', 0)
            sharpe = summary.get('sharpe_ratio', 0)
            rebalances = summary.get('rebalance_count', 0)
            
            # Add warning for unusually high returns
            warning = ""
            if return_pct > 1000:
                warning = " ⚠️"
            elif return_pct > 100:
                warning = " ⚡"
            
            print(f"{strategy_name:<20} {return_pct:<15.2f} {max_dd:<15.2f} {sharpe:<12.2f} {rebalances:<15}{warning}")
            successful_count += 1
        else:
            print(f"{strategy_name:<20} {'FAILED':<15} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
    
    print(f"\n📊 Successfully tested: {successful_count}/{len(strategies)} strategies")
    print("\n🎉 Improved strategy comparison completed!")
    print(f"📊 Comparison chart saved: {chart_file}")
    
    # Save detailed results to CSV
    csv_file = os.path.join("reports", f"all_strategies_comparison_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
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
                "final_value": summary.get('final_value', 0),
                "status": "SUCCESS"
            })
        else:
            summary_data.append({
                "strategy": strategy_name,
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "rebalance_count": 0,
                "total_fees_paid": 0,
                "final_value": 0,
                "status": "FAILED"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"📄 Detailed results saved: {csv_file}")

if __name__ == "__main__":
    main()
