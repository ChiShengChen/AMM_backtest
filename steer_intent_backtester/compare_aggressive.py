#!/usr/bin/env python3
"""
Aggressive strategy comparison with more realistic drawdowns.
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

def run_aggressive_backtest(strategy_name, strategy_params, data, config_base):
    """Run backtest with aggressive settings."""
    logger.info(f"Running {strategy_name} strategy (aggressive)...")
    
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
        if summary and summary.get('max_drawdown_pct', 0) == 0:
            logger.warning(f"⚠️  {strategy_name} still has 0% drawdown - too conservative!")
        elif summary and summary.get('max_drawdown_pct', 0) > 0:
            logger.info(f"✅ {strategy_name} has realistic drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        
        logger.info(f"✅ {strategy_name} completed: {summary.get('total_return_pct', 0):.2f}% return, {summary.get('max_drawdown_pct', 0):.2f}% max DD")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"❌ {strategy_name} failed: {e}")
        return None, None

def plot_aggressive_comparison(all_results, output_dir="reports"):
    """Plot comparison with better formatting."""
    
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
    ax1.set_title("Aggressive Strategy Comparison - ETHUSDC (Realistic Drawdowns)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Drawdown Comparison", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis - 减少标签密度
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add summary table
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
        summary_df = pd.DataFrame(summary_data)
        
        # Add table
        table_ax = fig.add_axes([0.02, 0.02, 0.4, 0.15])
        table_ax.axis('tight')
        table_ax.axis('off')
        
        table = table_ax.table(cellText=summary_df.values, 
                              colLabels=summary_df.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aggressive_strategies_comparison_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Aggressive comparison chart saved: {filepath}")
    return filepath

def main():
    """Main function for aggressive strategy comparison."""
    print("🚀 Starting Aggressive Strategy Comparison on ETHUSDC")
    print("=" * 70)
    
    # Load ETHUSDC data
    data_file = "data/ETHUSDC_1h.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 6 months for more realistic results
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=180)
    
    # Filter data
    data = data[data.index >= start_date]
    
    print(f"📈 Using limited data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # More aggressive strategy parameters
    strategies = {
        'classic': {
            "width_mode": "percent",
            "width_value": 10.0,  # Wider range: 10%
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.01  # More aggressive scaling
        },
        'channel_multiplier': {
            "width_pct": 10.0,  # Wider range: 10%
            "liquidity_scale": 0.01
        },
        'bollinger': {
            "n": 20,  # Shorter period
            "k": 2.0,  # More aggressive: 2.0
            "liquidity_scale": 0.01
        },
        'keltner': {
            "n": 20,  # Shorter period
            "m": 2.0,  # More aggressive: 2.0
            "liquidity_scale": 0.01
        },
        'donchian': {
            "n": 20,  # Shorter period
            "width_multiplier": 1.0,  # More aggressive: 1.0
            "liquidity_scale": 0.01
        },
        'stable': {
            "peg_method": "sma",
            "peg_period": 20,  # Shorter period
            "width_pct": 10.0,  # Wider: 10%
            "curve_type": "gaussian",
            "bin_count": 3,  # More bins
            "liquidity_scale": 0.01
        },
        'fluid': {
            "ideal_ratio": 1.0,
            "acceptable_ratio": 0.3,  # Less tolerant
            "sprawl_type": "dynamic",  # Dynamic instead of static
            "tail_weight": 0.2,  # Higher tail weight
            "liquidity_scale": 0.01
        }
    }
    
    # More aggressive base configuration
    config_base = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.01,  # Higher liquidity share
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # Run all strategies
    print("\n🔄 Running aggressive backtests...")
    print("-" * 70)
    
    all_results = {}
    all_summaries = {}
    
    for strategy_name, strategy_params in strategies.items():
        results, summary = run_aggressive_backtest(strategy_name, strategy_params, data, config_base)
        all_results[strategy_name] = results
        all_summaries[strategy_name] = summary
    
    # Generate comparison chart
    print("\n📊 Generating aggressive comparison chart...")
    chart_file = plot_aggressive_comparison(all_results)
    
    # Print summary
    print("\n📋 Aggressive Strategy Performance Summary")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Return %':<15} {'Max DD %':<15} {'Sharpe':<12} {'Rebalances':<15}")
    print("-" * 80)
    
    successful_count = 0
    for strategy_name, summary in all_summaries.items():
        if summary:
            return_pct = summary.get('total_return_pct', 0)
            max_dd = summary.get('max_drawdown_pct', 0)
            sharpe = summary.get('sharpe_ratio', 0)
            rebalances = summary.get('rebalance_count', 0)
            
            # Add indicators for drawdown
            indicator = ""
            if max_dd == 0:
                indicator = " ⚠️"
            elif max_dd > -20:
                indicator = " ✅"
            elif max_dd > -50:
                indicator = " ⚡"
            else:
                indicator = " 🚨"
            
            print(f"{strategy_name:<20} {return_pct:<15.2f} {max_dd:<15.2f} {sharpe:<12.2f} {rebalances:<15}{indicator}")
            successful_count += 1
        else:
            print(f"{strategy_name:<20} {'FAILED':<15} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
    
    print(f"\n📊 Successfully tested: {successful_count}/{len(strategies)} strategies")
    print(f"📊 Chart saved: {chart_file}")
    
    # Save results
    csv_file = os.path.join("reports", f"aggressive_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    summary_data = []
    for strategy_name, summary in all_summaries.items():
        if summary:
            summary_data.append({
                "strategy": strategy_name,
                "total_return_pct": summary.get('total_return_pct', 0),
                "max_drawdown_pct": summary.get('max_drawdown_pct', 0),
                "sharpe_ratio": summary.get('sharpe_ratio', 0),
                "rebalance_count": summary.get('rebalance_count', 0),
                "status": "SUCCESS"
            })
        else:
            summary_data.append({
                "strategy": strategy_name,
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "rebalance_count": 0,
                "status": "FAILED"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"📄 Results saved: {csv_file}")

if __name__ == "__main__":
    main()
