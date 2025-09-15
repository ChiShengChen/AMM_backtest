#!/usr/bin/env python3
"""
創建steer回測比較的總結視覺化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 設置英文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
df = pd.read_csv('/Users/michael/Desktop/Omnis_bt/steer_comparison_results/rebalance_comparison_table.csv')

# Create summary charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Steer Backtest Fix Effect Summary', fontsize=16, fontweight='bold')

# 1. Rebalance Count Comparison
axes[0, 0].bar(df['Strategy'], df['Total Rebalances'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Rebalance Count Comparison', fontweight='bold')
axes[0, 0].set_ylabel('Rebalance Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Total Fees Comparison
axes[0, 1].bar(df['Strategy'], df['Total Fees ($)'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_title('Total Fees Comparison', fontweight='bold')
axes[0, 1].set_ylabel('Total Fees ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Cash Ratio Comparison
axes[1, 0].bar(df['Strategy'], df['Cash Ratio (%)'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 0].set_title('Cash Ratio Comparison', fontweight='bold')
axes[1, 0].set_ylabel('Cash Ratio (%)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Total Return Comparison
axes[1, 1].bar(df['Strategy'], df['Total Return (%)'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 1].set_title('Total Return Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Total Return (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/michael/Desktop/Omnis_bt/steer_comparison_results/summary_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Create efficiency comparison chart
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate efficiency metric (Return / Fee Rate)
efficiency = df['Total Return (%)'] / (df['Total Fees ($)'] / 10000 * 100)
efficiency = efficiency.fillna(0)

bars = ax.bar(df['Strategy'], efficiency, 
              color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Strategy Efficiency Comparison (Return/Fee Rate)', fontsize=14, fontweight='bold')
ax.set_ylabel('Efficiency Ratio')
ax.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars, efficiency):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/Users/michael/Desktop/Omnis_bt/steer_comparison_results/efficiency_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("📊 Summary visualization charts generated:")
print("   - summary_comparison.png: Main metrics comparison")
print("   - efficiency_comparison.png: Efficiency comparison")
