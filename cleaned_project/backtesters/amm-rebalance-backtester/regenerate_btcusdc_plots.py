#!/usr/bin/env python3
"""
重新生成 BTC/USDC 圖表腳本
"""

import sys
import os
sys.path.append('src')

from src.reporting.plots import PlotGenerator
from src.reporting.strategy_recorder import StrategyRecorder
import yaml

def main():
    # 載入 BTC/USDC 配置
    with open('configs/btcusdc_experiment.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 創建圖表生成器
    plot_generator = PlotGenerator(config)
    
    # 模擬 BTC/USDC 結果
    results = {
        'strategies': {
            'Baseline-Static': {
                'apr': 5.2, 'mdd': 15.2, 'sharpe': 0.8, 'calmar': 0.34,
                'rebalance_count': 2
            },
            'Baseline-Fixed': {
                'apr': 8.1, 'mdd': 12.8, 'sharpe': 1.2, 'calmar': 0.63,
                'rebalance_count': 15
            },
            'Dynamic-Vol': {
                'apr': 12.3, 'mdd': 9.5, 'sharpe': 1.8, 'calmar': 1.29,
                'rebalance_count': 28
            },
            'Dynamic-Inventory': {
                'apr': 14.7, 'mdd': 8.2, 'sharpe': 2.1, 'calmar': 1.79,
                'rebalance_count': 35
            }
        }
    }
    
    # 生成所有圖表
    print("正在生成 BTC/USDC 圖表...")
    
    # 確保目錄存在
    os.makedirs('reports/figs/btcusdc', exist_ok=True)
    
    # 生成圖表
    # Generate plots with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pool = "BTCUSDC"
    pool_dir = f"reports/figs/{pool.lower()}"
    os.makedirs(pool_dir, exist_ok=True)
    plot_generator.plot_equity_curves(results, f"{pool_dir}/{pool}_equity_curves_{timestamp}.png")
    plot_generator.plot_apr_mdd_scatter(results, f"{pool_dir}/{pool}_apr_mdd_scatter_{timestamp}.png")
    plot_generator.plot_fee_vs_price_pnl(results, f"{pool_dir}/{pool}_fee_vs_price_pnl_{timestamp}.png")
    plot_generator.plot_sensitivity_heatmap(results, f"{pool_dir}/{pool}_sensitivity_heatmap_{timestamp}.png")
    plot_generator.plot_gas_frequency_contour(results, f"{pool_dir}/{pool}_gas_frequency_contour_{timestamp}.png")
    plot_generator.plot_il_curve(results, f"{pool_dir}/{pool}_il_curve_{timestamp}.png")
    plot_generator.plot_lvr_estimates(results, f"{pool_dir}/{pool}_lvr_estimates_{timestamp}.png")
    
    print("✅ BTC/USDC 圖表生成完成！")
    print("📁 圖表保存在: reports/figs/btcusdc/")

if __name__ == "__main__":
    main()
