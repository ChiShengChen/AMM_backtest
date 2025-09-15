#!/usr/bin/env python3
"""
Improved Comparison Charts for Ultimate Steer Strategies
改進的終極steer策略比較圖表
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 設置英文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedComparisonCharts:
    def __init__(self, output_dir="simplified_ultimate_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 顏色配置
        self.colors = {
            'steer_original': '#E74C3C',      # 紅色
            'steer_fixed': '#27AE60',         # 綠色
            'steer_core': '#3498DB',          # 藍色
            'steer_specialized': '#E67E22',   # 橙色
            'steer_ml': '#9B59B6',            # 紫色
            'steer_quantum': '#F39C12',       # 黃色
            'classical_ml': '#2ECC71',        # 綠色
            'quantum_ml': '#8E44AD',          # 深紫色
            'hybrid_ml': '#E91E63'            # 粉紅色
        }
    
    def load_strategy_data(self):
        """載入策略數據"""
        logger.info("📊 Loading strategy data...")
        
        # 載入現有的排名數據
        csv_file = self.output_dir / 'final_ranking_table.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            logger.info("✅ Loaded existing ranking data")
        else:
            # 如果沒有現有數據，創建模擬數據
            logger.warning("⚠️ No existing data found, creating simulated data")
            df = self.create_simulated_data()
        
        return df
    
    def create_simulated_data(self):
        """創建模擬數據"""
        strategies = [
            {'name': 'Quantum Bollinger Strategy', 'type': 'steer_quantum', 'category': 'Steer Quantum', 'accuracy': 0.8200, 'rebalance_count': 30, 'total_fees': 120.0, 'cash_ratio': 0.950, 'annual_return': 0.250, 'sharpe_ratio': 1.85, 'max_drawdown': 0.070},
            {'name': 'Random Forest', 'type': 'classical_ml', 'category': 'ML/QML', 'accuracy': 0.9948, 'rebalance_count': 41, 'total_fees': 205.0, 'cash_ratio': 0.800, 'annual_return': 0.284, 'sharpe_ratio': 1.79, 'max_drawdown': 0.166},
            {'name': 'ML Bollinger Strategy', 'type': 'steer_ml', 'category': 'Steer ML', 'accuracy': 0.8500, 'rebalance_count': 35, 'total_fees': 150.0, 'cash_ratio': 0.920, 'annual_return': 0.220, 'sharpe_ratio': 1.75, 'max_drawdown': 0.080},
            {'name': 'Stable Strategy', 'type': 'steer_core', 'category': 'Steer Core', 'accuracy': 0.7654, 'rebalance_count': 45, 'total_fees': 160.0, 'cash_ratio': 0.880, 'annual_return': 0.200, 'sharpe_ratio': 1.62, 'max_drawdown': 0.100},
            {'name': 'Bollinger Strategy', 'type': 'steer_core', 'category': 'Steer Core', 'accuracy': 0.7456, 'rebalance_count': 42, 'total_fees': 180.0, 'cash_ratio': 0.900, 'annual_return': 0.190, 'sharpe_ratio': 1.58, 'max_drawdown': 0.110},
            {'name': 'Gradient Boosting', 'type': 'classical_ml', 'category': 'ML/QML', 'accuracy': 0.9948, 'rebalance_count': 38, 'total_fees': 190.0, 'cash_ratio': 0.800, 'annual_return': 0.073, 'sharpe_ratio': 0.79, 'max_drawdown': 0.239},
            {'name': 'QuantumRWKV', 'type': 'hybrid_ml', 'category': 'ML/QML', 'accuracy': 0.8251, 'rebalance_count': 33, 'total_fees': 165.0, 'cash_ratio': 0.800, 'annual_return': 0.250, 'sharpe_ratio': 0.99, 'max_drawdown': 0.255},
            {'name': 'Classic Strategy', 'type': 'steer_core', 'category': 'Steer Core', 'accuracy': 0.7234, 'rebalance_count': 28, 'total_fees': 200.0, 'cash_ratio': 0.850, 'annual_return': 0.180, 'sharpe_ratio': 1.45, 'max_drawdown': 0.120},
            {'name': 'Fixed (Conservative)', 'type': 'steer_fixed', 'category': 'Steer Fix', 'accuracy': 0.6986, 'rebalance_count': 2880, 'total_fees': 47.88, 'cash_ratio': 1.000, 'annual_return': -0.0048, 'sharpe_ratio': 1.50, 'max_drawdown': 0.150},
            {'name': 'QASA Hybrid', 'type': 'hybrid_ml', 'category': 'ML/QML', 'accuracy': 0.6425, 'rebalance_count': 41, 'total_fees': 220.0, 'cash_ratio': 0.850, 'annual_return': 0.297, 'sharpe_ratio': 2.38, 'max_drawdown': 0.205},
            {'name': 'Fixed (Moderate)', 'type': 'steer_fixed', 'category': 'Steer Fix', 'accuracy': 0.6977, 'rebalance_count': 2414, 'total_fees': 562.02, 'cash_ratio': 0.951, 'annual_return': -0.0075, 'sharpe_ratio': 1.30, 'max_drawdown': 0.200},
            {'name': 'Original (Before Fix)', 'type': 'steer_original', 'category': 'Steer Fix', 'accuracy': 0.6977, 'rebalance_count': 2414, 'total_fees': 562.02, 'cash_ratio': 0.951, 'annual_return': -0.0075, 'sharpe_ratio': 1.20, 'max_drawdown': 0.250}
        ]
        
        return pd.DataFrame(strategies)
    
    def create_improved_efficiency_analysis(self, df):
        """創建改進的效率分析圖表 - 每個點都標上模型名稱"""
        logger.info("📊 Creating improved efficiency analysis with model names...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
        
        strategies = df['Strategy'].tolist()
        types = df['Type'].tolist()
        colors = [self.colors[t] for t in types]
        
        # 1. 準確率 vs 重新平衡次數
        accuracies = df['Accuracy'].tolist()
        rebalance_counts = df['Rebalance Count'].tolist()
        
        scatter1 = ax1.scatter(rebalance_counts, accuracies, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
        ax1.set_xlabel('Rebalance Count', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax1.set_title('Accuracy vs Rebalance Frequency', fontweight='bold', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # 添加策略標籤 - 調整位置避免重疊
        for i, strategy in enumerate(strategies):
            # 根據位置調整標籤位置
            if rebalance_counts[i] > 1000:  # 高重新平衡次數的策略
                offset_x, offset_y = 50, 0.01
            else:  # 低重新平衡次數的策略
                offset_x, offset_y = 5, 0.01
            
            ax1.annotate(strategy, (rebalance_counts[i], accuracies[i]), 
                        xytext=(offset_x, offset_y), textcoords='offset points', 
                        fontsize=9, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 2. 夏普比率 vs 最大回撤
        sharpe_ratios = df['Sharpe Ratio'].tolist()
        max_drawdowns = df['Max Drawdown'].tolist()
        
        scatter2 = ax2.scatter(max_drawdowns, sharpe_ratios, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
        ax2.set_xlabel('Max Drawdown', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Sharpe Ratio', fontweight='bold', fontsize=12)
        ax2.set_title('Risk-Return Profile', fontweight='bold', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # 添加策略標籤
        for i, strategy in enumerate(strategies):
            ax2.annotate(strategy, (max_drawdowns[i], sharpe_ratios[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 3. 年化回報率 vs 波動率
        annual_returns = df['Annual Return'].tolist()
        volatilities = [abs(ret) * 0.3 for ret in annual_returns]  # 估算波動率
        
        scatter3 = ax3.scatter(volatilities, annual_returns, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
        ax3.set_xlabel('Volatility', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Annual Return', fontweight='bold', fontsize=12)
        ax3.set_title('Return vs Risk', fontweight='bold', fontsize=16)
        ax3.grid(True, alpha=0.3)
        
        # 添加策略標籤
        for i, strategy in enumerate(strategies):
            ax3.annotate(strategy, (volatilities[i], annual_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 4. 策略類型統計
        type_counts = {t: types.count(t) for t in set(types)}
        
        wedges, texts, autotexts = ax4.pie(type_counts.values(), labels=type_counts.keys(), 
                                          colors=[self.colors[t] for t in type_counts.keys()],
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Strategy Type Distribution', fontweight='bold', fontsize=16)
        
        # 調整標籤字體大小
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Improved efficiency analysis saved")
    
    def create_improved_performance_heatmap(self, df):
        """創建改進的性能熱力圖 - 數值不標準化"""
        logger.info("📊 Creating improved performance heatmap without normalization...")
        
        # 準備數據 - 不進行標準化
        metrics = ['Accuracy', 'Rebalance Count', 'Total Fees ($)', 'Cash Ratio', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
        
        # 創建數據矩陣 - 使用原始數值
        data_matrix = []
        for _, row in df.iterrows():
            row_data = [
                row['Accuracy'],
                row['Rebalance Count'],
                row['Total Fees ($)'],
                row['Cash Ratio'],
                row['Annual Return'],
                row['Sharpe Ratio'],
                row['Max Drawdown']
            ]
            data_matrix.append(row_data)
        
        df_heatmap = pd.DataFrame(data_matrix, index=df['Strategy'], columns=metrics)
        
        # 創建熱力圖
        plt.figure(figsize=(14, 12))
        sns.heatmap(df_heatmap, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Performance Value'})
        
        plt.title('Steer Strategies vs ML/QML Models - Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Strategies/Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Improved performance heatmap saved")
    
    def create_improved_rebalance_comparison(self, df):
        """創建改進的重新平衡次數比較圖表 - 按數字大到小排序"""
        logger.info("📊 Creating improved rebalance comparison with sorting...")
        
        # 按重新平衡次數從大到小排序
        df_sorted = df.sort_values('Rebalance Count', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # 左圖：條形圖比較 - 已排序
        strategies = df_sorted['Strategy'].tolist()
        rebalance_counts = df_sorted['Rebalance Count'].tolist()
        types = df_sorted['Type'].tolist()
        colors = [self.colors[t] for t in types]
        
        bars = ax1.bar(range(len(strategies)), rebalance_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Strategies', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Rebalance Count', fontweight='bold', fontsize=12)
        ax1.set_title('Steer Strategies vs ML/QML Models - Rebalance Frequency Comparison (Sorted)', fontweight='bold', fontsize=16)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, count in zip(bars, rebalance_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(rebalance_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 右圖：按類型分組的箱線圖
        type_groups = {}
        for strategy_type in set(types):
            type_groups[strategy_type] = [count for count, t in zip(rebalance_counts, types) if t == strategy_type]
        
        box_data = list(type_groups.values())
        box_labels = list(type_groups.keys())
        box_colors = [self.colors[t] for t in box_labels]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Rebalance Count', fontweight='bold', fontsize=12)
        ax2.set_title('Rebalance Frequency Distribution by Strategy Type', fontweight='bold', fontsize=16)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加統計信息
        for i, (data_group, label) in enumerate(zip(box_data, box_labels)):
            if len(data_group) > 0:
                mean_reb = np.mean(data_group)
                ax2.text(i+1, mean_reb + max([max(g) for g in box_data])*0.01, f'Mean: {mean_reb:.1f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_rebalance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Improved rebalance comparison saved")
    
    def run_analysis(self):
        """運行完整分析"""
        logger.info("🚀 Starting improved comparison charts generation...")
        
        # 載入策略數據
        df = self.load_strategy_data()
        
        # 創建改進的圖表
        self.create_improved_efficiency_analysis(df)
        self.create_improved_performance_heatmap(df)
        self.create_improved_rebalance_comparison(df)
        
        logger.info(f"✅ Improved comparison charts completed! Results saved in: {self.output_dir}")
        logger.info("📊 Generated improved charts:")
        logger.info("  - steer_vs_ml_efficiency_analysis.png (with model names)")
        logger.info("  - steer_vs_ml_performance_heatmap.png (without normalization)")
        logger.info("  - steer_vs_ml_rebalance_comparison.png (sorted by rebalance count)")

def main():
    """主函數"""
    print("🚀 Improved Comparison Charts Generator")
    print("=" * 50)
    
    # 創建分析器
    analyzer = ImprovedComparisonCharts()
    
    # 運行分析
    analyzer.run_analysis()
    
    print("\n✅ Improved comparison charts completed!")
    print(f"📁 Results saved in: {analyzer.output_dir}")

if __name__ == "__main__":
    main()
