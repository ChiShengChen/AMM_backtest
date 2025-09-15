#!/usr/bin/env python3
"""
Fixed All Strategies Comparison Script
修復版本的所有策略比較腳本 - 使用修復後的Portfolio和Backtester
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append('.')

from steerbt.backtester_fixed import Backtester

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data."""
    print(f"📊 Loading data from: {file_path}")
    
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"📈 Data loaded: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    return data

def run_strategy_backtest(data: pd.DataFrame, strategy_name: str, strategy_params: dict, config_base: dict) -> dict:
    """Run backtest for a single strategy."""
    logger.info(f"Running {strategy_name} strategy...")
    
    config = config_base.copy()
    config['strategy'] = strategy_name
    config['strategy_params'] = strategy_params
    
    backtester = Backtester(config)
    results = backtester.run(data)
    
    if results and 'metrics' in results:
        strategy_metrics = results['metrics']['strategy']
        return {
            'strategy': strategy_name,
            'return_pct': strategy_metrics.get('annual_return', 0) * 100,
            'max_drawdown_pct': strategy_metrics.get('max_drawdown', 0) * 100,
            'sharpe_ratio': strategy_metrics.get('sharpe_ratio', 0),
            'rebalance_count': strategy_metrics.get('rebalance_count', 0),
            'final_value': strategy_metrics.get('final_value', 0),
            'total_fees_paid': strategy_metrics.get('total_fees_paid', 0),
            'results': results
        }
    else:
        logger.warning(f"⚠️ {strategy_name} failed to produce results")
        return {
            'strategy': strategy_name,
            'return_pct': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'rebalance_count': 0,
            'final_value': 0,
            'total_fees_paid': 0,
            'results': None
        }

def create_comparison_chart(results_df: pd.DataFrame, output_path: str):
    """Create comparison chart."""
    print("📊 Generating comparison chart...")
    
    # Filter out failed strategies
    valid_results = results_df[results_df['return_pct'] != 0].copy()
    
    if valid_results.empty:
        print("⚠️ No valid results to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Steer Strategies Performance Comparison (Fixed)', fontsize=16, fontweight='bold')
    
    # 1. Return vs Drawdown scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(valid_results['max_drawdown_pct'], valid_results['return_pct'], 
                         c=valid_results['sharpe_ratio'], cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Max Drawdown (%)')
    ax1.set_ylabel('Annual Return (%)')
    ax1.set_title('Risk vs Return')
    ax1.grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, row in valid_results.iterrows():
        ax1.annotate(row['strategy'], (row['max_drawdown_pct'], row['return_pct']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
    
    # 2. Sharpe Ratio comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(valid_results['strategy'], valid_results['sharpe_ratio'], 
                   color='skyblue', alpha=0.7)
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, valid_results['sharpe_ratio']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Rebalance count comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(valid_results['strategy'], valid_results['rebalance_count'], 
                   color='lightcoral', alpha=0.7)
    ax3.set_ylabel('Rebalance Count')
    ax3.set_title('Rebalance Frequency')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, valid_results['rebalance_count']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_results['rebalance_count'])*0.01, 
                f'{int(value)}', ha='center', va='bottom', fontsize=8)
    
    # 4. Final value comparison
    ax4 = axes[1, 1]
    bars = ax4.bar(valid_results['strategy'], valid_results['final_value'], 
                   color='lightgreen', alpha=0.7)
    ax4.set_ylabel('Final Value ($)')
    ax4.set_title('Final Portfolio Value')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for bar, value in zip(bars, valid_results['final_value']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_results['final_value'])*0.01, 
                f'${value:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison chart saved: {output_path}")

def main():
    """Main function."""
    print("🚀 Starting Fixed All Strategies Comparison on ETHUSDC")
    print("=" * 60)
    
    # Load data
    data = load_data('data/ETHUSDC_1h.csv')
    
    # Define all strategies and their parameters with conservative settings
    strategies = {
        'classic': {
            "width_mode": "percent",
            "width_value": 5.0,
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.01  # 修復：使用更保守的流動性規模
        },
        'channel_multiplier': {
            "width_pct": 5.0,
            "liquidity_scale": 0.01
        },
        'bollinger': {
            "n": 20,
            "k": 1.5,
            "liquidity_scale": 0.01
        },
        'keltner': {
            "n": 20,
            "m": 1.5,
            "liquidity_scale": 0.01
        },
        'donchian': {
            "n": 20,
            "width_multiplier": 0.8,
            "liquidity_scale": 0.01
        },
        'stable': {
            "peg_method": "sma",
            "peg_period": 20,
            "width_pct": 10.0,
            "curve_type": "gaussian",
            "bin_count": 3,
            "liquidity_scale": 0.01
        },
        'fluid': {
            "ideal_ratio": 1.0,
            "acceptable_ratio": 0.15,
            "sprawl_type": "dynamic",
            "tail_weight": 0.15,
            "liquidity_scale": 0.01
        }
    }
    
    # Base configuration with conservative settings
    config_base = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.001,
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # Run backtests for all strategies
    print("\n🔄 Running backtests for all strategies...")
    print("-" * 60)
    
    results = []
    for strategy_name, strategy_params in strategies.items():
        result = run_strategy_backtest(data, strategy_name, strategy_params, config_base)
        results.append(result)
        
        # Log completion
        if result['results']:
            logger.info(f"✅ {strategy_name} completed: {result['return_pct']:.2f}% return, {result['max_drawdown_pct']:.2f}% max DD")
        else:
            logger.warning(f"❌ {strategy_name} failed")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create comparison chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"reports/all_strategies_comparison_fixed_{timestamp}.png"
    create_comparison_chart(results_df, chart_path)
    
    # Print summary table
    print("\n📋 Strategy Performance Summary")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Return %':<15} {'Max DD %':<12} {'Sharpe':<10} {'Rebalances':<12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<20} {row['return_pct']:<15.2f} {row['max_drawdown_pct']:<12.2f} "
              f"{row['sharpe_ratio']:<10.2f} {row['rebalance_count']:<12.0f}")
    
    # Save detailed results
    csv_path = f"reports/all_strategies_comparison_fixed_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n🎉 All strategies comparison completed!")
    print(f"📊 Comparison chart saved: {chart_path}")
    print(f"📄 Detailed results saved: {csv_path}")

if __name__ == "__main__":
    main()
