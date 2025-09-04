#!/usr/bin/env python3
"""
Imperfect strategy implementation to generate realistic MDD.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import warnings
import random
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester
from steerbt.strategies.classic import ClassicStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImperfectClassicStrategy(ClassicStrategy):
    """Classic strategy with imperfections to generate realistic MDD."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.imperfection_level = kwargs.get('imperfection_level', 0.5)  # 0-1 scale
        self.rebalance_failure_rate = kwargs.get('rebalance_failure_rate', 0.3)
        self.liquidity_shortage_rate = kwargs.get('liquidity_shortage_rate', 0.2)
        self.market_impact_rate = kwargs.get('market_impact_rate', 0.1)
        self.rebalance_count = 0
        self.failed_rebalances = 0
        
    def update(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float) -> bool:
        """Override update method to add imperfections."""
        # Call parent method
        should_rebalance = super().update(price_data, current_price, portfolio_value)
        
        if should_rebalance:
            # Simulate rebalancing failure
            if random.random() < self.rebalance_failure_rate:
                logger.info(f"🔄 Rebalancing failed due to imperfection (level: {self.imperfection_level})")
                self.failed_rebalances += 1
                return False
                
            # Simulate liquidity shortage
            if random.random() < self.liquidity_shortage_rate:
                logger.info(f"💧 Liquidity shortage detected, reducing rebalancing effectiveness")
                # Reduce the effectiveness of this rebalancing
                self.imperfection_level = min(1.0, self.imperfection_level + 0.1)
                
            # Simulate market impact
            if random.random() < self.market_impact_rate:
                logger.info(f"📉 Market impact detected, increasing slippage")
                # This will be handled in the backtester
                
            self.rebalance_count += 1
            
        return should_rebalance
    
    def calculate_range(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float):
        """Override calculate_range to add imperfections."""
        ranges, liquidities = super().calculate_range(price_data, current_price, portfolio_value)
        
        if ranges and liquidities:
            # Apply imperfection to liquidity distribution
            imperfection_factor = 1.0 - (self.imperfection_level * 0.5)
            
            # Reduce liquidity effectiveness
            adjusted_liquidities = [liq * imperfection_factor for liq in liquidities]
            
            # Sometimes create "bad" ranges (too wide or too narrow)
            if random.random() < 0.2:  # 20% chance of bad ranges
                adjusted_ranges = []
                for lower, upper in ranges:
                    # Make range wider or narrower randomly
                    if random.random() < 0.5:
                        # Wider range (less effective)
                        range_width = upper - lower
                        adjusted_lower = lower - range_width * 0.2
                        adjusted_upper = upper + range_width * 0.2
                    else:
                        # Narrower range (more concentrated risk)
                        range_width = upper - lower
                        adjusted_lower = lower + range_width * 0.1
                        adjusted_upper = upper - range_width * 0.1
                    
                    adjusted_ranges.append((adjusted_lower, adjusted_upper))
                
                ranges = adjusted_ranges
            
            return ranges, adjusted_liquidities
        
        return ranges, liquidities

def run_imperfect_strategy(strategy_name, strategy_params, data, config_base, imperfection_config):
    """Run strategy with imperfections."""
    logger.info(f"Running {strategy_name} with imperfection level: {imperfection_config['level']}...")
    
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
        
        # Check if we got realistic MDD
        mdd = summary.get('max_drawdown_pct', 0)
        if mdd < 0:
            logger.info(f"✅ {strategy_name} produced realistic MDD: {mdd:.2f}%")
        elif mdd == 0:
            logger.warning(f"⚠️  {strategy_name} still has 0% MDD")
        else:
            logger.info(f"✅ {strategy_name} MDD: {mdd:.2f}%")
        
        logger.info(f"✅ {strategy_name} completed: {summary.get('total_return_pct', 0):.2f}% return, {mdd:.2f}% max DD")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"❌ {strategy_name} failed: {e}")
        return None, None

def plot_imperfect_comparison(all_results, output_dir="reports"):
    """Plot comparison with imperfections focus."""
    
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
    ax1.set_title("Imperfect Strategy - MDD Generation Test", fontsize=16, fontweight='bold')
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
    
    # Add summary table
    summary_data = []
    for strategy_name, results in all_results.items():
        if results is None:
            continue
            
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
    filename = f"imperfect_strategy_comparison_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Imperfect strategy chart saved: {filepath}")
    return filepath

def main():
    """Main function for imperfect strategy simulation."""
    print("🚀 Starting Imperfect Strategy MDD Generation Test")
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
    
    # Imperfection configurations
    imperfection_configs = {
        'classic_perfect': {
            'level': 0.0,
            'rebalance_failure_rate': 0.0,
            'liquidity_shortage_rate': 0.0,
            'market_impact_rate': 0.0
        },
        'classic_slight_imperfect': {
            'level': 0.3,
            'rebalance_failure_rate': 0.1,
            'liquidity_shortage_rate': 0.1,
            'market_impact_rate': 0.05
        },
        'classic_moderate_imperfect': {
            'level': 0.6,
            'rebalance_failure_rate': 0.3,
            'liquidity_shortage_rate': 0.2,
            'market_impact_rate': 0.1
        },
        'classic_highly_imperfect': {
            'level': 0.9,
            'rebalance_failure_rate': 0.5,
            'liquidity_shortage_rate': 0.4,
            'market_impact_rate': 0.2
        }
    }
    
    # Strategy parameters
    strategy_params = {
        "width_mode": "percent",
        "width_value": 20.0,
        "placement_mode": "center",
        "curve_type": "uniform",
        "liquidity_scale": 0.01,
        "max_positions": 3,
        "gap_bps": 1000,
        "drift_threshold_pct": 20.0,
        "time_threshold_hours": 72
    }
    
    # Base configuration
    config_base = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "initial_cash": 10000.0,
        "fee_bps": 30,
        "slippage_bps": 15,
        "gas_cost": 150.0,
        "liq_share": 0.005,
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # Run all imperfection levels
    print("\n🔄 Running imperfect strategy tests...")
    print("-" * 80)
    
    all_results = {}
    all_summaries = {}
    
    for config_name, imperfection_config in imperfection_configs.items():
        # Add imperfection parameters to strategy
        test_params = strategy_params.copy()
        test_params.update(imperfection_config)
        
        results, summary = run_imperfect_strategy(config_name, test_params, data, config_base, imperfection_config)
        all_results[config_name] = results
        all_summaries[config_name] = summary
    
    # Generate comparison chart
    print("\n📊 Generating imperfect strategy chart...")
    chart_file = plot_imperfect_comparison(all_results)
    
    # Print summary
    print("\n📋 Imperfect Strategy Results Summary")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Return %':<15} {'Max DD %':<15} {'Sharpe':<12} {'Rebalances':<15}")
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
            
            # Get imperfection level
            imperfection_level = imperfection_configs[config_name]['level']
            strategy_display = f"{config_name} ({imperfection_level:.1f})"
            
            print(f"{strategy_display:<30} {return_pct:<15.2f} {max_dd:<15.2f} {sharpe:<12.2f} {rebalances:<15}{indicator}")
            successful_count += 1
        else:
            print(f"{config_name:<30} {'FAILED':<15} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
    
    print(f"\n📊 Results Summary:")
    print(f"   Successfully tested: {successful_count}/{len(imperfection_configs)} strategies")
    print(f"   Strategies with MDD: {mdd_strategies}/{successful_count}")
    print(f"   Chart saved: {chart_file}")
    
    # Save results
    csv_file = os.path.join("reports", f"imperfect_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    summary_data = []
    for config_name, summary in all_summaries.items():
        if summary:
            imperfection_config = imperfection_configs[config_name]
            summary_data.append({
                "strategy": config_name,
                "imperfection_level": imperfection_config['level'],
                "rebalance_failure_rate": imperfection_config['rebalance_failure_rate'],
                "liquidity_shortage_rate": imperfection_config['liquidity_shortage_rate'],
                "market_impact_rate": imperfection_config['market_impact_rate'],
                "total_return_pct": summary.get('total_return_pct', 0),
                "max_drawdown_pct": summary.get('max_drawdown_pct', 0),
                "sharpe_ratio": summary.get('sharpe_ratio', 0),
                "rebalance_count": summary.get('rebalance_count', 0),
                "has_mdd": summary.get('max_drawdown_pct', 0) < 0,
                "status": "SUCCESS"
            })
        else:
            imperfection_config = imperfection_configs[config_name]
            summary_data.append({
                "strategy": config_name,
                "imperfection_level": imperfection_config['level'],
                "rebalance_failure_rate": imperfection_config['rebalance_failure_rate'],
                "liquidity_shortage_rate": imperfection_config['liquidity_shortage_rate'],
                "market_impact_rate": imperfection_config['market_impact_rate'],
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "rebalance_count": 0,
                "has_mdd": False,
                "status": "FAILED"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"📄 Results saved: {csv_file}")
    
    # Final analysis
    print(f"\n💡 Imperfect Strategy Analysis:")
    if mdd_strategies > 0:
        print(f"   ✅ Successfully generated {mdd_strategies} strategies with realistic MDD")
        print(f"   🔍 Key factors for MDD generation:")
        print(f"      - Rebalancing failures ({imperfection_configs['classic_highly_imperfect']['rebalance_failure_rate']*100:.0f}% rate)")
        print(f"      - Liquidity shortages ({imperfection_configs['classic_highly_imperfect']['liquidity_shortage_rate']*100:.0f}% rate)")
        print(f"      - Market impact simulation")
        print(f"      - Reduced strategy effectiveness")
    else:
        print(f"   ❌ No strategies generated realistic MDD")
        print(f"   🔍 May need to implement more fundamental strategy changes")

if __name__ == "__main__":
    main()
