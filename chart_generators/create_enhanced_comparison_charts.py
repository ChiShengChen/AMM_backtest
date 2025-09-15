#!/usr/bin/env python3
"""
Enhanced Comparison Charts for Ultimate Steer Strategies
為終極steer策略比較創建增強版比較圖表
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

class EnhancedComparisonCharts:
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
            {'name': 'QASA Hybrid', 'type': 'hybrid_ml', 'category': 'ML/QML', 'accuracy': 0.6425, 'rebalance_count': 41, 'total_fees': 220.0, 'cash_ratio': 0.850, 'annual_return': 0.297, 'sharpe_ratio': 2.38, 'max_drawdown': 0.205},
            {'name': 'Bollinger Strategy', 'type': 'steer_core', 'category': 'Steer Core', 'accuracy': 0.7456, 'rebalance_count': 42, 'total_fees': 180.0, 'cash_ratio': 0.900, 'annual_return': 0.190, 'sharpe_ratio': 1.58, 'max_drawdown': 0.110},
            {'name': 'QuantumRWKV', 'type': 'hybrid_ml', 'category': 'ML/QML', 'accuracy': 0.8251, 'rebalance_count': 33, 'total_fees': 165.0, 'cash_ratio': 0.800, 'annual_return': 0.250, 'sharpe_ratio': 0.99, 'max_drawdown': 0.255},
            {'name': 'Classic Strategy', 'type': 'steer_core', 'category': 'Steer Core', 'accuracy': 0.7234, 'rebalance_count': 28, 'total_fees': 200.0, 'cash_ratio': 0.850, 'annual_return': 0.180, 'sharpe_ratio': 1.45, 'max_drawdown': 0.120},
            {'name': 'Gradient Boosting', 'type': 'classical_ml', 'category': 'ML/QML', 'accuracy': 0.9948, 'rebalance_count': 38, 'total_fees': 190.0, 'cash_ratio': 0.800, 'annual_return': 0.073, 'sharpe_ratio': 0.79, 'max_drawdown': 0.239},
            {'name': 'Fixed (Conservative)', 'type': 'steer_fixed', 'category': 'Steer Fix', 'accuracy': 0.6986, 'rebalance_count': 2880, 'total_fees': 47.88, 'cash_ratio': 1.000, 'annual_return': -0.0048, 'sharpe_ratio': 1.50, 'max_drawdown': 0.150},
            {'name': 'Original (Before Fix)', 'type': 'steer_original', 'category': 'Steer Fix', 'accuracy': 0.6977, 'rebalance_count': 2414, 'total_fees': 562.02, 'cash_ratio': 0.951, 'annual_return': -0.0075, 'sharpe_ratio': 1.20, 'max_drawdown': 0.250},
            {'name': 'Fixed (Moderate)', 'type': 'steer_fixed', 'category': 'Steer Fix', 'accuracy': 0.6977, 'rebalance_count': 2414, 'total_fees': 562.02, 'cash_ratio': 0.951, 'annual_return': -0.0075, 'sharpe_ratio': 1.30, 'max_drawdown': 0.200}
        ]
        
        return pd.DataFrame(strategies)
    
    def create_accuracy_comparison_chart(self, df):
        """創建準確率比較圖表"""
        logger.info("📊 Creating accuracy comparison chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左圖：條形圖比較
        strategies = df['Strategy'].tolist()
        accuracies = df['Accuracy'].tolist()
        types = df['Type'].tolist()
        colors = [self.colors[t] for t in types]
        
        bars = ax1.bar(range(len(strategies)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Strategies', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Steer Strategies vs ML/QML Models - Accuracy Comparison', fontweight='bold', fontsize=16)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 右圖：按類型分組的箱線圖
        type_groups = {}
        for strategy_type in set(types):
            type_groups[strategy_type] = [acc for acc, t in zip(accuracies, types) if t == strategy_type]
        
        box_data = list(type_groups.values())
        box_labels = list(type_groups.keys())
        box_colors = [self.colors[t] for t in box_labels]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('Accuracy Distribution by Strategy Type', fontweight='bold', fontsize=16)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加統計信息
        for i, (data_group, label) in enumerate(zip(box_data, box_labels)):
            if len(data_group) > 0:
                mean_acc = np.mean(data_group)
                ax2.text(i+1, mean_acc + 0.05, f'Mean: {mean_acc:.3f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Accuracy comparison chart saved")
    
    def create_rebalance_comparison_chart(self, df):
        """創建重新平衡次數比較圖表"""
        logger.info("📊 Creating rebalance comparison chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左圖：條形圖比較
        strategies = df['Strategy'].tolist()
        rebalance_counts = df['Rebalance Count'].tolist()
        types = df['Type'].tolist()
        colors = [self.colors[t] for t in types]
        
        bars = ax1.bar(range(len(strategies)), rebalance_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Strategies', fontweight='bold')
        ax1.set_ylabel('Rebalance Count', fontweight='bold')
        ax1.set_title('Steer Strategies vs ML/QML Models - Rebalance Frequency Comparison', fontweight='bold', fontsize=16)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, count in zip(bars, rebalance_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(rebalance_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
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
        
        ax2.set_ylabel('Rebalance Count', fontweight='bold')
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
        
        logger.info("✅ Rebalance comparison chart saved")
    
    def create_equity_curve_with_uncertainty(self, df):
        """創建權益曲線與不確定性帶圖表"""
        logger.info("📊 Creating equity curve with uncertainty chart...")
        
        # 生成時間序列數據
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 1, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 創建圖表
        plt.figure(figsize=(16, 10))
        
        # 為每個策略生成權益曲線
        for i, (_, row) in enumerate(df.iterrows()):
            strategy_name = row['Strategy']
            annual_return = row['Annual Return']
            volatility = abs(annual_return) * 0.3  # 估算波動率
            sharpe_ratio = row['Sharpe Ratio']
            
            # 生成模擬的累積回報曲線
            np.random.seed(hash(strategy_name) % 2**32)  # 確保可重現性
            daily_returns = np.random.normal(annual_return/365, volatility/365, len(dates))
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            # 生成不確定性帶
            uncertainty_std = volatility * np.sqrt(np.arange(len(dates)) / 365)
            upper_bound = cumulative_returns + 1.96 * uncertainty_std
            lower_bound = cumulative_returns - 1.96 * uncertainty_std
            
            # 選擇顏色
            color = self.colors[row['Type']]
            
            # 繪製主線
            plt.plot(dates, cumulative_returns * 100, 
                    label=strategy_name, color=color, linewidth=2, alpha=0.8)
            
            # 繪製不確定性帶
            plt.fill_between(dates, lower_bound * 100, upper_bound * 100, 
                           color=color, alpha=0.2)
        
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Cumulative Return (%)', fontweight='bold')
        plt.title('All Strategies: Cumulative Return Comparison with Uncertainty Bands', fontweight='bold', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加零線
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_strategies_equity_curve_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Equity curve with uncertainty chart saved")
    
    def create_performance_heatmap(self, df):
        """創建性能熱力圖"""
        logger.info("📊 Creating performance heatmap...")
        
        # 準備數據
        metrics = ['accuracy', 'rebalance_count', 'total_fees', 'cash_ratio', 'annual_return', 'sharpe_ratio', 'max_drawdown']
        
        # 創建數據矩陣
        data_matrix = []
        for _, row in df.iterrows():
            row_data = [
                row['Accuracy'],
                row['Rebalance Count'] / 100,  # 標準化
                row['Total Fees ($)'] / 1000,      # 標準化
                row['Cash Ratio'],
                row['Annual Return'],
                row['Sharpe Ratio'] / 3,       # 標準化
                1 - row['Max Drawdown']        # 轉換為正向指標
            ]
            data_matrix.append(row_data)
        
        df_heatmap = pd.DataFrame(data_matrix, index=df['Strategy'], columns=metrics)
        
        # 創建熱力圖
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_heatmap, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Normalized Performance'})
        
        plt.title('Steer Strategies vs ML/QML Models - Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Strategies/Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Performance heatmap saved")
    
    def create_efficiency_analysis(self, df):
        """創建效率分析圖表"""
        logger.info("📊 Creating efficiency analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        strategies = df['Strategy'].tolist()
        types = df['Type'].tolist()
        colors = [self.colors[t] for t in types]
        
        # 1. 準確率 vs 重新平衡次數
        accuracies = df['Accuracy'].tolist()
        rebalance_counts = df['Rebalance Count'].tolist()
        
        scatter1 = ax1.scatter(rebalance_counts, accuracies, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Rebalance Count', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy vs Rebalance Frequency', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加策略標籤
        for i, strategy in enumerate(strategies):
            ax1.annotate(strategy, (rebalance_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. 夏普比率 vs 最大回撤
        sharpe_ratios = df['Sharpe Ratio'].tolist()
        max_drawdowns = df['Max Drawdown'].tolist()
        
        scatter2 = ax2.scatter(max_drawdowns, sharpe_ratios, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Max Drawdown', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax2.set_title('Risk-Return Profile', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. 年化回報率 vs 波動率
        annual_returns = df['Annual Return'].tolist()
        volatilities = [abs(ret) * 0.3 for ret in annual_returns]  # 估算波動率
        
        scatter3 = ax3.scatter(volatilities, annual_returns, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Volatility', fontweight='bold')
        ax3.set_ylabel('Annual Return', fontweight='bold')
        ax3.set_title('Return vs Risk', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. 策略類型統計
        type_counts = {t: types.count(t) for t in set(types)}
        
        wedges, texts, autotexts = ax4.pie(type_counts.values(), labels=type_counts.keys(), 
                                          colors=[self.colors[t] for t in type_counts.keys()],
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Strategy Type Distribution', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steer_vs_ml_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Efficiency analysis saved")
    
    def run_analysis(self):
        """運行完整分析"""
        logger.info("🚀 Starting enhanced comparison charts generation...")
        
        # 載入策略數據
        df = self.load_strategy_data()
        
        # 創建各種比較圖表
        self.create_accuracy_comparison_chart(df)
        self.create_rebalance_comparison_chart(df)
        self.create_equity_curve_with_uncertainty(df)
        self.create_performance_heatmap(df)
        self.create_efficiency_analysis(df)
        
        logger.info(f"✅ Enhanced comparison charts completed! Results saved in: {self.output_dir}")
        logger.info("📊 Generated charts:")
        logger.info("  - steer_vs_ml_accuracy_comparison.png")
        logger.info("  - steer_vs_ml_rebalance_comparison.png")
        logger.info("  - all_strategies_equity_curve_with_uncertainty.png")
        logger.info("  - steer_vs_ml_performance_heatmap.png")
        logger.info("  - steer_vs_ml_efficiency_analysis.png")

def main():
    """主函數"""
    print("🚀 Enhanced Comparison Charts Generator")
    print("=" * 50)
    
    # 創建分析器
    analyzer = EnhancedComparisonCharts()
    
    # 運行分析
    analyzer.run_analysis()
    
    print("\n✅ Enhanced comparison charts completed!")
    print(f"📁 Results saved in: {analyzer.output_dir}")

if __name__ == "__main__":
    main()
