#!/usr/bin/env python3
"""
Example script demonstrating the new organized report system.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_sample_data():
    """Create sample price data for demonstration."""
    # Create 30 days of hourly data
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=30*24, freq='H')
    
    # Generate realistic price data with some volatility
    np.random.seed(42)
    base_price = 2000.0
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'quote_volume': volume * price
        })
    
    return pd.DataFrame(data)

def run_demo_backtest():
    """Run a demonstration backtest with organized reports."""
    print("Steer Intent Backtester - Organized Reports Demo")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample price data...")
    price_data = create_sample_data()
    price_data = price_data.set_index('timestamp')
    
    print(f"✓ Generated {len(price_data)} data points")
    print(f"  Date range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"  Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
    
    # Create backtest configuration
    config = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "strategy": "bollinger",
        "strategy_params": {
            "n": 20,
            "k": 2.0
        },
        "initial_cash": 10000.0,
        "fee_bps": 5,
        "slippage_bps": 1,
        "gas_cost": 0.0,
        "liq_share": 0.002,
        "start_date": price_data.index[0],
        "end_date": price_data.index[-1]
    }
    
    print(f"\nRunning backtest with configuration:")
    print(f"  Pair: {config['pair']}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Parameters: {config['strategy_params']}")
    print(f"  Initial cash: ${config['initial_cash']:,.2f}")
    
    try:
        # Import and run backtester
        from steerbt.backtester import Backtester
        
        # Create and run backtester
        backtester = Backtester(config)
        results = backtester.run(price_data)
        
        print("✓ Backtest completed successfully")
        
        # Generate organized reports
        print("\nGenerating organized reports...")
        report_files = backtester.generate_reports("demo_reports")
        
        print("✓ Reports generated successfully!")
        print(f"\nExperiment directory structure:")
        print(f"  {report_files.get('index', '').replace('/index_', '/').replace('_', '/')}")
        
        # Show file structure
        experiment_dir = os.path.dirname(report_files.get('index', ''))
        if os.path.exists(experiment_dir):
            print(f"\nGenerated files:")
            for root, dirs, files in os.walk(experiment_dir):
                level = root.replace(experiment_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_size = os.path.getsize(os.path.join(root, file))
                    print(f"{subindent}{file} ({file_size:,} bytes)")
        
        # Show performance summary
        performance = results.get("performance", {})
        print(f"\nPerformance Summary:")
        print(f"  Total Return: {performance.get('total_return_pct', 0):.2f}%")
        print(f"  Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"  Rebalance Count: {performance.get('rebalance_count', 0)}")
        
        # Show baseline comparison
        baselines = results.get("baselines", {})
        if baselines:
            print(f"\nBaseline Comparison:")
            if "hodl_50_50" in baselines:
                hodl = baselines["hodl_50_50"]
                print(f"  HODL 50:50 Return: {hodl.get('total_return_pct', 0):.2f}%")
            if "single_asset" in baselines:
                single = baselines["single_asset"]
                print(f"  Single Asset Return: {single.get('total_return_pct', 0):.2f}%")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Open the index file in your browser to view all results:")
        print(f"   {report_files.get('index', '')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_demo_backtest()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ The new organized report system is working perfectly!")
        print("📋 Key features demonstrated:")
        print("   • Automatic experiment directory creation")
        print("   • Organized file structure (figs/, data/, logs/)")
        print("   • HTML index page for easy browsing")
        print("   • Complete experiment configuration tracking")
        print("   • Performance metrics and baseline comparisons")
        print("\n🚀 You can now run real backtests with:")
        print("   python cli.py backtest --pair ETHUSDC --strategy bollinger --n 20 --k 2")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
