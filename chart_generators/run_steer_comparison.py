#!/usr/bin/env python3
"""
運行steer回測比較，生成圖表和表格
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# 添加路徑
sys.path.append('/Users/michael/Desktop/Omnis_bt/steer_intent_backtester')
sys.path.append('/Users/michael/Desktop/Omnis_bt/cleaned_project/backtesters/steer_intent_backtester')

from steerbt.backtester import Backtester
from steerbt.strategies.classic import ClassicStrategy
from steerbt.portfolio import Portfolio

# 設置英文字體和樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_points=5000):
    """創建測試數據"""
    logger.info(f"🔧 創建 {n_points} 個數據點的測試數據...")
    
    # 創建模擬的ETH價格數據
    base_price = 2000.0
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
    
    # 生成價格數據（帶有趨勢和波動）
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n_points)  # 2% 標準差
    trend = np.linspace(0, 0.3, n_points)  # 30% 總趨勢
    prices = base_price * np.exp(np.cumsum(returns) + trend)
    
    # 添加一些波動事件
    for i in range(500, n_points, 1000):
        if i < n_points:
            prices[i:i+20] *= 0.9  # 10% 下跌
        if i + 500 < n_points:
            prices[i+500:i+520] *= 1.1  # 10% 上漲
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_points),
        'quote_volume': np.random.uniform(1000000, 10000000, n_points)
    }, index=dates)
    
    return data

def run_backtest(config, data, name):
    """運行單個回測"""
    logger.info(f"🔄 運行 {name} 回測...")
    
    try:
        backtester = Backtester(config)
        results = backtester.run(data)
        
        if results is None:
            logger.error(f"❌ {name} 回測失敗")
            return None
        
        portfolio = backtester.portfolio
        
        # 獲取權益曲線
        equity_df = portfolio.get_equity_dataframe()
        
        # 計算性能指標
        final_value = portfolio.get_total_value(data['close'].iloc[-1])
        initial_cash = config['initial_cash']
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        # 計算最大回撤
        if not equity_df.empty:
            cumulative = (1 + equity_df['total_value'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0
        
        # 計算夏普比率
        if not equity_df.empty and len(equity_df) > 1:
            returns = equity_df['total_value'].pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365.25 * 24)  # 年化
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'name': name,
            'equity_curve': equity_df,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'rebalance_count': len(portfolio.transaction_history),
            'total_fees': portfolio.total_fees_paid,
            'final_cash': portfolio.cash,
            'portfolio': portfolio
        }
        
    except Exception as e:
        logger.error(f"❌ {name} 回測失敗: {e}")
        return None

def create_comparison_plots(results, output_dir="steer_comparison_results"):
    """創建比較圖表"""
    logger.info("📊 創建比較圖表...")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Portfolio Value Comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Portfolio Value
    plt.subplot(2, 2, 1)
    for result in results:
        if result and not result['equity_curve'].empty:
            plt.plot(result['equity_curve'].index, 
                    result['equity_curve']['total_value'], 
                    label=result['name'], linewidth=2)
    
    plt.title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Drawdown Comparison
    plt.subplot(2, 2, 2)
    for result in results:
        if result and not result['equity_curve'].empty:
            equity_df = result['equity_curve']
            cumulative = (1 + equity_df['total_value'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            plt.plot(equity_df.index, drawdown, label=result['name'], linewidth=2)
    
    plt.title('Drawdown Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cash Balance Comparison
    plt.subplot(2, 2, 3)
    for result in results:
        if result and not result['equity_curve'].empty:
            plt.plot(result['equity_curve'].index, 
                    result['equity_curve']['cash'], 
                    label=result['name'], linewidth=2)
    
    plt.title('Cash Balance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cash Balance ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Cumulative Fees
    plt.subplot(2, 2, 4)
    for result in results:
        if result and not result['equity_curve'].empty:
            plt.plot(result['equity_curve'].index, 
                    result['equity_curve']['total_costs'], 
                    label=result['name'], linewidth=2)
    
    plt.title('Cumulative Fees Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Fees ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Performance Metrics Comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    metrics_data = []
    for result in results:
        if result:
            metrics_data.append({
                'Strategy': result['name'],
                'Total Return (%)': result['total_return'],
                'Max Drawdown (%)': result['max_drawdown'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Rebalance Count': result['rebalance_count'],
                'Total Fees ($)': result['total_fees']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Total Return
    axes[0, 0].bar(metrics_df['Strategy'], metrics_df['Total Return (%)'], 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[0, 0].set_title('Total Return Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Total Return (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Max Drawdown
    axes[0, 1].bar(metrics_df['Strategy'], metrics_df['Max Drawdown (%)'], 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[0, 1].set_title('Max Drawdown Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Max Drawdown (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sharpe Ratio
    axes[0, 2].bar(metrics_df['Strategy'], metrics_df['Sharpe Ratio'], 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[0, 2].set_title('Sharpe Ratio Comparison', fontweight='bold')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Rebalance Count
    axes[1, 0].bar(metrics_df['Strategy'], metrics_df['Rebalance Count'], 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[1, 0].set_title('Rebalance Count Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Rebalance Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Total Fees
    axes[1, 1].bar(metrics_df['Strategy'], metrics_df['Total Fees ($)'], 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[1, 1].set_title('Total Fees Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Total Fees ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Efficiency Metric (Return/Fees)
    efficiency = metrics_df['Total Return (%)'] / (metrics_df['Total Fees ($)'] / 10000 * 100)
    axes[1, 2].bar(metrics_df['Strategy'], efficiency, 
                   color=['#2E8B57', '#DC143C', '#4169E1'])
    axes[1, 2].set_title('Efficiency (Return/Fees)', fontweight='bold')
    axes[1, 2].set_ylabel('Efficiency Ratio')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df

def create_rebalance_table(results, output_dir="steer_comparison_results"):
    """創建重新平衡次數比較表格"""
    logger.info("📋 創建重新平衡次數比較表格...")
    
    # 準備表格數據
    table_data = []
    for result in results:
        if result:
            portfolio = result['portfolio']
            
            # 計算重新平衡統計
            rebalance_history = portfolio.transaction_history
            rebalance_count = len(rebalance_history)
            
            # 計算重新平衡頻率
            if result['equity_curve'].empty:
                rebalance_frequency = 0
            else:
                try:
                    total_hours = (result['equity_curve'].index[-1] - result['equity_curve'].index[0]).total_seconds() / 3600
                    rebalance_frequency = rebalance_count / total_hours * 24  # 每日重新平衡次數
                except:
                    # 如果時間計算失敗，使用數據點數估算
                    total_hours = len(result['equity_curve'])  # 假設每小時一個數據點
                    rebalance_frequency = rebalance_count / total_hours * 24
            
            # 計算平均重新平衡成本
            avg_rebalance_cost = result['total_fees'] / rebalance_count if rebalance_count > 0 else 0
            
            table_data.append({
                'Strategy': result['name'],
                'Total Rebalances': rebalance_count,
                'Daily Rebalance Rate': f"{rebalance_frequency:.2f}",
                'Avg Rebalance Cost ($)': f"{avg_rebalance_cost:.2f}",
                'Total Fees ($)': f"{result['total_fees']:.2f}",
                'Final Cash ($)': f"{result['final_cash']:.2f}",
                'Cash Ratio (%)': f"{(result['final_cash'] / result['final_value'] * 100):.1f}",
                'Final Value ($)': f"{result['final_value']:.2f}",
                'Total Return (%)': f"{result['total_return']:.2f}"
            })
    
    # 創建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存為CSV
    df.to_csv(f'{output_dir}/rebalance_comparison_table.csv', index=False)
    
    # 創建HTML表格
    html_table = df.to_html(index=False, classes='table table-striped table-hover')
    
    with open(f'{output_dir}/rebalance_comparison_table.html', 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Steer Rebalance Comparison</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h2>Steer Rebalance Comparison Table</h2>
            {html_table}
        </body>
        </html>
        """)
    
    # Print table
    print("\n" + "="*100)
    print("REBALANCE COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df

def main():
    """主函數"""
    logger.info("🚀 開始運行steer回測比較...")
    
    # 創建測試數據
    data = create_test_data(5000)
    logger.info(f"📊 數據範圍: {data.index[0]} 到 {data.index[-1]}")
    logger.info(f"💰 價格範圍: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Define test configurations
    configs = [
        {
            'name': 'Original (Before Fix)',
            'config': {
                "pair": "ETHUSDC",
                "interval": "1h",
                "strategy": "classic",
                "strategy_params": {
                    "width_mode": "percent",
                    "width_value": 5.0,
                    "placement_mode": "center",
                    "curve_type": "uniform",
                    "liquidity_scale": 0.01
                },
                "initial_cash": 10000.0,
                "fee_bps": 5,
                "slippage_bps": 1,
                "gas_cost": 0.0,
                "liq_share": 0.001,
                "start_date": data.index[0],
                "end_date": data.index[-1]
            }
        },
        {
            'name': 'Fixed (Conservative)',
            'config': {
                "pair": "ETHUSDC",
                "interval": "1h",
                "strategy": "classic",
                "strategy_params": {
                    "width_mode": "percent",
                    "width_value": 10.0,
                    "placement_mode": "center",
                    "curve_type": "uniform",
                    "liquidity_scale": 0.001
                },
                "initial_cash": 10000.0,
                "fee_bps": 5,
                "slippage_bps": 1,
                "gas_cost": 0.0,
                "liq_share": 0.001,
                "start_date": data.index[0],
                "end_date": data.index[-1]
            }
        },
        {
            'name': 'Fixed (Moderate)',
            'config': {
                "pair": "ETHUSDC",
                "interval": "1h",
                "strategy": "classic",
                "strategy_params": {
                    "width_mode": "percent",
                    "width_value": 5.0,
                    "placement_mode": "center",
                    "curve_type": "uniform",
                    "liquidity_scale": 0.01
                },
                "initial_cash": 10000.0,
                "fee_bps": 5,
                "slippage_bps": 1,
                "gas_cost": 0.0,
                "liq_share": 0.001,
                "start_date": data.index[0],
                "end_date": data.index[-1]
            }
        }
    ]
    
    # 運行回測
    results = []
    for config_info in configs:
        result = run_backtest(config_info['config'], data, config_info['name'])
        if result:
            results.append(result)
    
    if not results:
        logger.error("❌ 所有回測都失敗了")
        return
    
    # 創建比較圖表
    metrics_df = create_comparison_plots(results)
    
    # 創建重新平衡表格
    rebalance_df = create_rebalance_table(results)
    
    # 生成總結報告
    logger.info("\n" + "="*80)
    logger.info("回測比較總結")
    logger.info("="*80)
    
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"  最終價值: ${result['final_value']:.2f}")
        logger.info(f"  總回報率: {result['total_return']:.2f}%")
        logger.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        logger.info(f"  重新平衡次數: {result['rebalance_count']}")
        logger.info(f"  總手續費: ${result['total_fees']:.2f}")
        logger.info(f"  最終現金: ${result['final_cash']:.2f}")
    
    logger.info(f"\n📊 圖表和表格已保存到 'steer_comparison_results' 目錄")
    logger.info("✅ 回測比較完成！")

if __name__ == "__main__":
    main()
