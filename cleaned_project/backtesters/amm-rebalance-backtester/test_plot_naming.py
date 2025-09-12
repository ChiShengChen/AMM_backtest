#!/usr/bin/env python3
"""
測試圖表命名格式
"""

import os
import sys
from datetime import datetime
from src.reporting.plots import PlotGenerator
from src.io.schema import ValidationConfig
from src.io.loader import DataLoader

def test_plot_naming():
    """測試圖表命名格式"""
    
    print("🧪 Testing Plot Naming Format")
    print("=" * 40)
    
    # 配置
    pool = "BTCUSDC"
    frequency = "1d"
    
    # 加載數據
    print("📈 Loading test data...")
    config = ValidationConfig()
    loader = DataLoader("data", config)
    price_data, _ = loader.load_pool_data(pool, frequency)
    
    # 創建測試結果
    print("🔧 Creating test results...")
    test_results = {
        'summary': {
            'Baseline-Static': {'apr': 5.0, 'mdd': 10.0, 'sharpe': 0.5},
            'Baseline-Fixed': {'apr': 8.0, 'mdd': 15.0, 'sharpe': 0.53},
            'Dynamic-Vol': {'apr': 12.0, 'mdd': 20.0, 'sharpe': 0.6},
            'Dynamic-Inventory': {'apr': 10.0, 'mdd': 18.0, 'sharpe': 0.56}
        },
        'price_data_info': {
            'total_days': len(price_data),
            'start_date': price_data.index[0].strftime('%Y-%m-%d'),
            'end_date': price_data.index[-1].strftime('%Y-%m-%d')
        }
    }
    
    # 生成圖表
    print("📊 Generating test plots...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pool_dir = f"reports/figs/{pool.lower()}"
    os.makedirs(pool_dir, exist_ok=True)
    
    config_data = {'pool': pool, 'frequency': frequency}
    plot_generator = PlotGenerator(config_data)
    
    # 生成所有圖表
    plots = [
        ('equity_curves', 'Equity Curves'),
        ('apr_mdd_scatter', 'APR vs MDD Scatter'),
        ('fee_vs_price_pnl', 'Fee vs Price PnL'),
        ('sensitivity_heatmap', 'Sensitivity Heatmap'),
        ('gas_frequency_contour', 'Gas Frequency Contour'),
        ('il_curve', 'IL Curve'),
        ('lvr_estimates', 'LVR Estimates')
    ]
    
    generated_files = []
    
    for plot_type, plot_name in plots:
        try:
            filename = f"{pool_dir}/{pool}_{plot_type}_{timestamp}.png"
            print(f"  🔧 Generating {plot_name}...")
            
            if plot_type == 'equity_curves':
                plot_generator.plot_equity_curves(test_results, filename)
            elif plot_type == 'apr_mdd_scatter':
                plot_generator.plot_apr_mdd_scatter(test_results, filename)
            elif plot_type == 'fee_vs_price_pnl':
                plot_generator.plot_fee_vs_price_pnl(test_results, filename)
            elif plot_type == 'sensitivity_heatmap':
                plot_generator.plot_sensitivity_heatmap(test_results, filename)
            elif plot_type == 'gas_frequency_contour':
                plot_generator.plot_gas_frequency_contour(test_results, filename)
            elif plot_type == 'il_curve':
                plot_generator.plot_il_curve(test_results, filename)
            elif plot_type == 'lvr_estimates':
                plot_generator.plot_lvr_estimates(test_results, filename)
            
            generated_files.append(filename)
            print(f"    ✅ Saved: {filename}")
            
        except Exception as e:
            print(f"    ❌ Error generating {plot_name}: {e}")
    
    # 檢查生成的文件
    print(f"\n📁 Generated Files:")
    print("-" * 30)
    
    for file_path in generated_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} (not found)")
    
    # 顯示目錄結構
    print(f"\n📂 Directory Structure:")
    print("-" * 25)
    
    if os.path.exists(pool_dir):
        files = os.listdir(pool_dir)
        for file in sorted(files):
            if file.endswith('.png'):
                print(f"  📊 {file}")
    
    print(f"\n🎉 Plot naming test completed!")
    print(f"📊 Generated {len(generated_files)} plots with format: {{pool}}_{{plot_type}}_{{timestamp}}.png")

if __name__ == "__main__":
    test_plot_naming()
