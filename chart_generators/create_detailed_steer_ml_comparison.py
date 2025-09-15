#!/usr/bin/env python3
"""
Detailed Steer Strategies vs ML/QML Models Comparison
詳細比較steer策略與ML/QML模型的rebalance次數、APR和曲線圖
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設置英文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedSteerMLComparison:
    def __init__(self, output_dir="detailed_steer_ml_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 顏色配置
        self.colors = {
            'steer_original': '#E74C3C',      # 紅色 - 原始steer
            'steer_fixed': '#27AE60',         # 綠色 - 修正後steer
            'classical_ml': '#3498DB',        # 藍色 - 經典ML
            'quantum_ml': '#9B59B6',          # 紫色 - 量子ML
            'hybrid_ml': '#F39C12'            # 橙色 - 混合ML
        }
    
    def load_data(self):
        """載入所有數據"""
        logger.info("📊 Loading all data...")
        
        # 載入steer策略數據
        steer_csv = Path("steer_comparison_results/rebalance_comparison_table.csv")
        if steer_csv.exists():
            steer_df = pd.read_csv(steer_csv)
            logger.info("✅ Loaded steer strategies data")
        else:
            logger.warning("⚠️ Steer CSV not found, using simulated data")
            steer_df = pd.DataFrame({
                'Strategy': ['Original (Before Fix)', 'Fixed (Conservative)', 'Fixed (Moderate)'],
                'Total Rebalances': [2414, 2880, 2414],
                'Total Return (%)': [-0.75, -0.48, -0.75],
                'Total Fees ($)': [562.02, 47.88, 562.02],
                'Final Cash ($)': [9437.98, 9952.12, 9437.98],
                'Cash Ratio (%)': [95.1, 100.0, 95.1]
            })
        
        # 載入ML模型數據
        ml_data = {
            'Random Forest': {'accuracy': 0.9948, 'rebalance_count': 41, 'annual_return': 0.284, 'sharpe_ratio': 1.79, 'max_drawdown': 0.166, 'strategy_type': 'classical_ml'},
            'Gradient Boosting': {'accuracy': 0.9948, 'rebalance_count': 38, 'annual_return': 0.073, 'sharpe_ratio': 0.79, 'max_drawdown': 0.239, 'strategy_type': 'classical_ml'},
            'Logistic Regression': {'accuracy': 0.6373, 'rebalance_count': 48, 'annual_return': 0.150, 'sharpe_ratio': 1.47, 'max_drawdown': 0.056, 'strategy_type': 'classical_ml'},
            'VQE Classifier': {'accuracy': 0.5440, 'rebalance_count': 52, 'annual_return': 0.074, 'sharpe_ratio': 2.05, 'max_drawdown': 0.277, 'strategy_type': 'quantum_ml'},
            'QNN': {'accuracy': 0.3731, 'rebalance_count': 53, 'annual_return': 0.143, 'sharpe_ratio': 0.73, 'max_drawdown': 0.210, 'strategy_type': 'quantum_ml'},
            'QSVM': {'accuracy': 0.5130, 'rebalance_count': 50, 'annual_return': 0.082, 'sharpe_ratio': 1.94, 'max_drawdown': 0.229, 'strategy_type': 'quantum_ml'},
            'QASA Hybrid': {'accuracy': 0.6425, 'rebalance_count': 41, 'annual_return': 0.297, 'sharpe_ratio': 2.38, 'max_drawdown': 0.205, 'strategy_type': 'hybrid_ml'},
            'QuantumRWKV': {'accuracy': 0.8251, 'rebalance_count': 33, 'annual_return': 0.250, 'sharpe_ratio': 0.99, 'max_drawdown': 0.255, 'strategy_type': 'hybrid_ml'},
            'LSTM_QNN': {'accuracy': 0.6448, 'rebalance_count': 37, 'annual_return': 0.094, 'sharpe_ratio': 0.83, 'max_drawdown': 0.271, 'strategy_type': 'hybrid_ml'},
            'QASA Sequence': {'accuracy': 0.6448, 'rebalance_count': 34, 'annual_return': 0.266, 'sharpe_ratio': 0.97, 'max_drawdown': 0.215, 'strategy_type': 'hybrid_ml'}
        }
        
        return steer_df, ml_data
    
    def create_rebalance_comparison(self, steer_df, ml_data):
        """創建rebalance次數比較圖"""
        logger.info("📊 Creating rebalance count comparison...")
        
        # 準備數據
        steer_strategies = steer_df['Strategy'].tolist()
        steer_rebalances = steer_df['Total Rebalances'].tolist()
        steer_types = ['steer_original' if 'Original' in s else 'steer_fixed' for s in steer_strategies]
        
        ml_strategies = list(ml_data.keys())
        ml_rebalances = [ml_data[s]['rebalance_count'] for s in ml_strategies]
        ml_types = [ml_data[s]['strategy_type'] for s in ml_strategies]
        
        # 合併數據
        all_strategies = steer_strategies + ml_strategies
        all_rebalances = steer_rebalances + ml_rebalances
        all_types = steer_types + ml_types
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左圖：所有策略的rebalance次數比較
        colors = [self.colors[t] for t in all_types]
        bars = ax1.bar(range(len(all_strategies)), all_rebalances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Strategies/Models', fontweight='bold')
        ax1.set_ylabel('Rebalance Count', fontweight='bold')
        ax1.set_title('Rebalance Count Comparison: Steer vs ML/QML', fontweight='bold')
        ax1.set_xticks(range(len(all_strategies)))
        ax1.set_xticklabels(all_strategies, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, count in zip(bars, all_rebalances):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(all_rebalances)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 右圖：按類型分組的箱線圖
        type_groups = {
            'Steer Original': [r for r, t in zip(all_rebalances, all_types) if t == 'steer_original'],
            'Steer Fixed': [r for r, t in zip(all_rebalances, all_types) if t == 'steer_fixed'],
            'Classical ML': [r for r, t in zip(all_rebalances, all_types) if t == 'classical_ml'],
            'Quantum ML': [r for r, t in zip(all_rebalances, all_types) if t == 'quantum_ml'],
            'Hybrid ML': [r for r, t in zip(all_rebalances, all_types) if t == 'hybrid_ml']
        }
        
        # 只顯示有數據的組
        filtered_groups = {k: v for k, v in type_groups.items() if len(v) > 0}
        
        box_data = list(filtered_groups.values())
        box_labels = list(filtered_groups.keys())
        box_colors = [self.colors['steer_original'] if 'Steer Original' in label else
                     self.colors['steer_fixed'] if 'Steer Fixed' in label else
                     self.colors['classical_ml'] if 'Classical' in label else
                     self.colors['quantum_ml'] if 'Quantum' in label else
                     self.colors['hybrid_ml'] for label in box_labels]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Rebalance Count', fontweight='bold')
        ax2.set_title('Rebalance Count Distribution by Strategy Type', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加統計信息
        for i, (data_group, label) in enumerate(zip(box_data, box_labels)):
            if len(data_group) > 0:
                mean_reb = np.mean(data_group)
                ax2.text(i+1, mean_reb + max([max(g) for g in box_data])*0.01, f'Mean: {mean_reb:.1f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rebalance_count_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Rebalance count comparison saved")
    
    def create_apr_comparison(self, steer_df, ml_data):
        """創建APR比較圖"""
        logger.info("📊 Creating APR comparison...")
        
        # 準備數據
        steer_strategies = steer_df['Strategy'].tolist()
        steer_returns = steer_df['Total Return (%)'].tolist()
        steer_types = ['steer_original' if 'Original' in s else 'steer_fixed' for s in steer_strategies]
        
        ml_strategies = list(ml_data.keys())
        ml_returns = [ml_data[s]['annual_return'] * 100 for s in ml_strategies]  # 轉換為百分比
        ml_types = [ml_data[s]['strategy_type'] for s in ml_strategies]
        
        # 合併數據
        all_strategies = steer_strategies + ml_strategies
        all_returns = steer_returns + ml_returns
        all_types = steer_types + ml_types
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左圖：所有策略的APR比較
        colors = [self.colors[t] for t in all_types]
        bars = ax1.bar(range(len(all_strategies)), all_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Strategies/Models', fontweight='bold')
        ax1.set_ylabel('Annual Percentage Return (%)', fontweight='bold')
        ax1.set_title('APR Comparison: Steer vs ML/QML', fontweight='bold')
        ax1.set_xticks(range(len(all_strategies)))
        ax1.set_xticklabels(all_strategies, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, ret in zip(bars, all_returns):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (max(all_returns) - min(all_returns))*0.01,
                    f'{ret:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 右圖：按類型分組的箱線圖
        type_groups = {
            'Steer Original': [r for r, t in zip(all_returns, all_types) if t == 'steer_original'],
            'Steer Fixed': [r for r, t in zip(all_returns, all_types) if t == 'steer_fixed'],
            'Classical ML': [r for r, t in zip(all_returns, all_types) if t == 'classical_ml'],
            'Quantum ML': [r for r, t in zip(all_returns, all_types) if t == 'quantum_ml'],
            'Hybrid ML': [r for r, t in zip(all_returns, all_types) if t == 'hybrid_ml']
        }
        
        # 只顯示有數據的組
        filtered_groups = {k: v for k, v in type_groups.items() if len(v) > 0}
        
        box_data = list(filtered_groups.values())
        box_labels = list(filtered_groups.keys())
        box_colors = [self.colors['steer_original'] if 'Steer Original' in label else
                     self.colors['steer_fixed'] if 'Steer Fixed' in label else
                     self.colors['classical_ml'] if 'Classical' in label else
                     self.colors['quantum_ml'] if 'Quantum' in label else
                     self.colors['hybrid_ml'] for label in box_labels]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Annual Percentage Return (%)', fontweight='bold')
        ax2.set_title('APR Distribution by Strategy Type', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加統計信息
        for i, (data_group, label) in enumerate(zip(box_data, box_labels)):
            if len(data_group) > 0:
                mean_ret = np.mean(data_group)
                ax2.text(i+1, mean_ret + (max([max(g) for g in box_data]) - min([min(g) for g in box_data]))*0.01, f'Mean: {mean_ret:.2f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'apr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ APR comparison saved")
    
    def create_performance_curves(self, steer_df, ml_data):
        """創建性能曲線圖"""
        logger.info("📊 Creating performance curves...")
        
        # 模擬性能曲線數據
        np.random.seed(42)
        days = np.arange(0, 365)
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Steer策略性能曲線
        ax1.set_title('Steer Strategies Performance Curves', fontweight='bold', fontsize=14)
        
        for i, (_, row) in enumerate(steer_df.iterrows()):
            strategy = row['Strategy']
            annual_return = row['Total Return (%)'] / 100
            
            # 生成模擬的累積回報曲線
            daily_returns = np.random.normal(annual_return/365, 0.02, 365)
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            color = self.colors['steer_original'] if 'Original' in strategy else self.colors['steer_fixed']
            ax1.plot(days, cumulative_returns * 100, label=strategy, color=color, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Days', fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. ML模型性能曲線
        ax2.set_title('ML/QML Models Performance Curves', fontweight='bold', fontsize=14)
        
        for strategy, data in ml_data.items():
            annual_return = data['annual_return']
            strategy_type = data['strategy_type']
            
            # 生成模擬的累積回報曲線
            daily_returns = np.random.normal(annual_return/365, 0.02, 365)
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            color = self.colors[strategy_type]
            ax2.plot(days, cumulative_returns * 100, label=strategy, color=color, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Days', fontweight='bold')
        ax2.set_ylabel('Cumulative Return (%)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. 回撤曲線比較
        ax3.set_title('Drawdown Curves Comparison', fontweight='bold', fontsize=14)
        
        # Steer策略回撤
        for i, (_, row) in enumerate(steer_df.iterrows()):
            strategy = row['Strategy']
            annual_return = row['Total Return (%)'] / 100
            
            # 生成模擬的回撤曲線
            daily_returns = np.random.normal(annual_return/365, 0.02, 365)
            cumulative_returns = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max * 100
            
            color = self.colors['steer_original'] if 'Original' in strategy else self.colors['steer_fixed']
            ax3.plot(days, drawdown, label=f"Steer: {strategy}", color=color, linewidth=2, alpha=0.8)
        
        # 選擇幾個代表性的ML模型
        selected_ml = ['Random Forest', 'QuantumRWKV', 'QASA Hybrid']
        for strategy in selected_ml:
            if strategy in ml_data:
                data = ml_data[strategy]
                annual_return = data['annual_return']
                strategy_type = data['strategy_type']
                
                # 生成模擬的回撤曲線
                daily_returns = np.random.normal(annual_return/365, 0.02, 365)
                cumulative_returns = np.cumprod(1 + daily_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max * 100
                
                color = self.colors[strategy_type]
                ax3.plot(days, drawdown, label=f"ML: {strategy}", color=color, linewidth=2, alpha=0.8, linestyle='--')
        
        ax3.set_xlabel('Days', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 風險回報散點圖
        ax4.set_title('Risk-Return Scatter Plot', fontweight='bold', fontsize=14)
        
        # 準備數據
        all_strategies = []
        all_returns = []
        all_volatilities = []
        all_colors = []
        
        # Steer策略
        for _, row in steer_df.iterrows():
            strategy = row['Strategy']
            annual_return = row['Total Return (%)'] / 100
            volatility = abs(annual_return) * 0.3  # 估算波動率
            
            all_strategies.append(f"Steer: {strategy}")
            all_returns.append(annual_return * 100)
            all_volatilities.append(volatility * 100)
            all_colors.append(self.colors['steer_original'] if 'Original' in strategy else self.colors['steer_fixed'])
        
        # ML模型
        for strategy, data in ml_data.items():
            annual_return = data['annual_return']
            volatility = data.get('volatility', 0.2)  # 使用預設波動率
            
            all_strategies.append(f"ML: {strategy}")
            all_returns.append(annual_return * 100)
            all_volatilities.append(volatility * 100)
            all_colors.append(self.colors[data['strategy_type']])
        
        # 散點圖
        scatter = ax4.scatter(all_volatilities, all_returns, c=all_colors, s=100, alpha=0.7, edgecolors='black')
        
        # 添加標籤
        for i, strategy in enumerate(all_strategies):
            ax4.annotate(strategy, (all_volatilities[i], all_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Volatility (%)', fontweight='bold')
        ax4.set_ylabel('Annual Return (%)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Performance curves saved")
    
    def create_efficiency_metrics(self, steer_df, ml_data):
        """創建效率指標比較"""
        logger.info("📊 Creating efficiency metrics comparison...")
        
        # 準備數據
        data_list = []
        
        # Steer策略數據
        for _, row in steer_df.iterrows():
            strategy = row['Strategy']
            strategy_type = 'steer_original' if 'Original' in strategy else 'steer_fixed'
            
            data_list.append({
                'Strategy': strategy,
                'Type': strategy_type,
                'Rebalance Count': row['Total Rebalances'],
                'APR (%)': row['Total Return (%)'],
                'Total Fees ($)': row['Total Fees ($)'],
                'Cash Ratio (%)': row['Cash Ratio (%)'],
                'Efficiency (APR/Fees)': row['Total Return (%)'] / (row['Total Fees ($)'] / 1000) if row['Total Fees ($)'] > 0 else 0,
                'Rebalance Efficiency (APR/Rebalance)': row['Total Return (%)'] / row['Total Rebalances'] if row['Total Rebalances'] > 0 else 0
            })
        
        # ML模型數據
        for strategy, data in ml_data.items():
            annual_return = data['annual_return'] * 100
            rebalance_count = data['rebalance_count']
            strategy_type = data['strategy_type']
            
            # 估算費用和現金比例
            estimated_fees = rebalance_count * 5  # 假設每次rebalance 5美元
            estimated_cash_ratio = 0.8  # 假設80%現金比例
            
            data_list.append({
                'Strategy': strategy,
                'Type': strategy_type,
                'Rebalance Count': rebalance_count,
                'APR (%)': annual_return,
                'Total Fees ($)': estimated_fees,
                'Cash Ratio (%)': estimated_cash_ratio * 100,
                'Efficiency (APR/Fees)': annual_return / (estimated_fees / 1000) if estimated_fees > 0 else 0,
                'Rebalance Efficiency (APR/Rebalance)': annual_return / rebalance_count if rebalance_count > 0 else 0
            })
        
        df = pd.DataFrame(data_list)
        
        # 創建效率比較圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 效率指標 (APR/Fees)
        colors = [self.colors[t] for t in df['Type']]
        bars1 = ax1.bar(range(len(df)), df['Efficiency (APR/Fees)'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Strategies/Models', fontweight='bold')
        ax1.set_ylabel('Efficiency (APR/Fees)', fontweight='bold')
        ax1.set_title('Efficiency: APR per Dollar of Fees', fontweight='bold')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Strategy'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Rebalance效率 (APR/Rebalance)
        bars2 = ax2.bar(range(len(df)), df['Rebalance Efficiency (APR/Rebalance)'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Strategies/Models', fontweight='bold')
        ax2.set_ylabel('Rebalance Efficiency (APR/Rebalance)', fontweight='bold')
        ax2.set_title('Rebalance Efficiency: APR per Rebalance', fontweight='bold')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 費用比較
        bars3 = ax3.bar(range(len(df)), df['Total Fees ($)'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Strategies/Models', fontweight='bold')
        ax3.set_ylabel('Total Fees ($)', fontweight='bold')
        ax3.set_title('Total Fees Comparison', fontweight='bold')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['Strategy'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 現金比例比較
        bars4 = ax4.bar(range(len(df)), df['Cash Ratio (%)'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Strategies/Models', fontweight='bold')
        ax4.set_ylabel('Cash Ratio (%)', fontweight='bold')
        ax4.set_title('Cash Ratio Comparison', fontweight='bold')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(df['Strategy'], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存數據
        df.to_csv(self.output_dir / 'efficiency_metrics.csv', index=False)
        
        logger.info("✅ Efficiency metrics saved")
    
    def run_analysis(self):
        """運行完整分析"""
        logger.info("🚀 Starting detailed Steer vs ML/QML comparison analysis...")
        
        # 載入數據
        steer_df, ml_data = self.load_data()
        
        # 創建各種圖表
        self.create_rebalance_comparison(steer_df, ml_data)
        self.create_apr_comparison(steer_df, ml_data)
        self.create_performance_curves(steer_df, ml_data)
        self.create_efficiency_metrics(steer_df, ml_data)
        
        logger.info(f"✅ Analysis completed! Results saved in: {self.output_dir}")
        logger.info("📊 Generated charts:")
        logger.info("  - rebalance_count_comparison.png")
        logger.info("  - apr_comparison.png")
        logger.info("  - performance_curves.png")
        logger.info("  - efficiency_metrics.png")
        logger.info("  - efficiency_metrics.csv")

def main():
    """主函數"""
    print("🚀 Detailed Steer Strategies vs ML/QML Models Comparison")
    print("=" * 60)
    
    # 創建分析器
    analyzer = DetailedSteerMLComparison()
    
    # 運行分析
    analyzer.run_analysis()
    
    print("\n✅ Analysis completed!")
    print(f"📁 Results saved in: {analyzer.output_dir}")

if __name__ == "__main__":
    main()
