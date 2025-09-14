#!/usr/bin/env python3
"""
創建帶有誤差條和標準差陰影的圖表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class UncertaintyChartGenerator:
    """不確定性圖表生成器"""
    
    def __init__(self, output_dir="reports/uncertainty_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置顏色方案
        self.colors = {
            'classic_ml': '#2E86AB',      # 藍色
            'quantum_ml': '#A23B72',      # 紫色
            'hybrid_quantum': '#2CA02C',  # 綠色
            'transformer': '#FF6B6B',     # 紅色
            'baseline': '#6C757D'         # 灰色
        }
        
        # 模型分組
        self.model_groups = {
            'Classic ML': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
            'Quantum ML': ['VQE Classifier', 'QNN', 'QSVM'],
            'Hybrid Quantum': ['QASA Hybrid', 'LSTM_QNN', 'QuantumRWKV'],
            'Transformer': ['Transformer'],
            'Baseline': ['Static Baseline', 'Fixed Baseline']
        }
    
    def load_multiple_runs_data(self):
        """載入多輪運行數據"""
        try:
            # 載入統計數據
            stats_df = pd.read_csv('reports/improved_charts/multiple_runs_statistics.csv', index_col=0)
            
            # 載入時間序列數據
            with open('reports/improved_charts/multiple_runs_time_series.json', 'r') as f:
                time_series_data = json.load(f)
            
            # 載入原始結果
            raw_results = pd.read_csv('reports/improved_charts/multiple_runs_raw_results.csv')
            
            return stats_df, time_series_data, raw_results
            
        except FileNotFoundError as e:
            print(f"❌ 未找到多輪運行數據: {e}")
            return None, None, None
    
    def create_equity_curves_with_uncertainty(self, time_series_data):
        """創建帶有不確定性的資金曲線圖"""
        # 只使用有數據的模型組
        available_groups = {}
        for model_type, models in self.model_groups.items():
            available_models = [m for m in models if m in time_series_data]
            if available_models:
                available_groups[model_type] = available_models
        
        # 根據可用組數調整布局
        num_groups = len(available_groups)
        if num_groups <= 4:
            rows, cols = 2, 2
        elif num_groups <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
        fig.suptitle('Model Performance: Equity Curves with Uncertainty Bands', fontsize=16, fontweight='bold')
        
        # 確保axes是二維數組
        if num_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 生成時間索引 - 確保與數據長度匹配
        # 數據長度為252個交易日，所以我們需要生成252個工作日
        time_index = pd.bdate_range('2024-01-01', '2024-12-31', freq='B')
        # 確保長度為252
        if len(time_index) > 252:
            time_index = time_index[:252]
        elif len(time_index) < 252:
            # 如果不足252天，用最後一個日期填充
            last_date = time_index[-1]
            additional_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=252-len(time_index), freq='B')
            time_index = time_index.append(additional_dates)
        days = len(time_index)
        
        # 定義顏色方案
        color_schemes = {
            'Classic ML': ['#1f77b4', '#aec7e8', '#ff7f0e'],
            'Quantum ML': ['#d62728', '#ff9896', '#ffbb78'],
            'Hybrid Quantum': ['#2ca02c', '#98df8a', '#ff7f0e'],
            'Transformer': ['#ff6b6b'],
            'Baseline': ['#9467bd', '#c5b0d5']
        }
        
        for idx, (model_type, models) in enumerate(available_groups.items()):
            if idx >= rows * cols:
                break
                
            ax = axes[idx // cols, idx % cols]
            colors = color_schemes.get(model_type, ['#666666'])
            
            for i, model in enumerate(models):
                if model in time_series_data:
                    data = time_series_data[model]
                    
                    # 獲取數據
                    mean_returns = np.array(data['cumulative_returns_mean']) * 100
                    std_returns = np.array(data['cumulative_returns_std']) * 100
                    
                    # 確保數據長度匹配
                    if len(mean_returns) != len(time_index):
                        if len(mean_returns) > len(time_index):
                            mean_returns = mean_returns[:len(time_index)]
                            std_returns = std_returns[:len(time_index)]
                        else:
                            # 如果數據不足，用最後一個值填充
                            last_mean = mean_returns[-1] if len(mean_returns) > 0 else 0
                            last_std = std_returns[-1] if len(std_returns) > 0 else 0
                            mean_returns = np.concatenate([
                                mean_returns, 
                                np.full(len(time_index) - len(mean_returns), last_mean)
                            ])
                            std_returns = np.concatenate([
                                std_returns, 
                                np.full(len(time_index) - len(std_returns), last_std)
                            ])
                    
                    # 計算上下界
                    upper_bound = mean_returns + std_returns
                    lower_bound = mean_returns - std_returns
                    
                    # 繪製陰影區域
                    color = colors[i % len(colors)]
                    ax.fill_between(time_index, lower_bound, upper_bound, 
                                   alpha=0.2, color=color, label=f'{model} ±1σ')
                    
                    # 繪製平均線
                    ax.plot(time_index, mean_returns, 
                           color=color, linewidth=2.5, alpha=0.9, label=model)
            
            ax.set_title(f'{model_type} Models', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # 隱藏空的子圖
        for idx in range(num_groups, rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'equity_curves_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_all_models_equity_curve_with_uncertainty(self, time_series_data):
        """創建所有模型的資金曲線圖（帶不確定性）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('All Models: Cumulative Return Comparison with Uncertainty Bands', fontsize=18, fontweight='bold')
        
        # 生成時間索引 - 確保與數據長度匹配
        # 數據長度為252個交易日，所以我們需要生成252個工作日
        time_index = pd.bdate_range('2024-01-01', '2024-12-31', freq='B')
        # 確保長度為252
        if len(time_index) > 252:
            time_index = time_index[:252]
        elif len(time_index) < 252:
            # 如果不足252天，用最後一個日期填充
            last_date = time_index[-1]
            additional_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=252-len(time_index), freq='B')
            time_index = time_index.append(additional_dates)
        days = len(time_index)
        
        # 定義所有模型的顏色方案
        model_colors = {
            'Random Forest': '#1f77b4',
            'Gradient Boosting': '#aec7e8', 
            'Logistic Regression': '#ff7f0e',
            'VQE Classifier': '#d62728',
            'QNN': '#ff9896',
            'QSVM': '#ffbb78',
            'QASA Hybrid': '#2ca02c',
            'LSTM_QNN': '#98df8a',
            'QuantumRWKV': '#ff7f0e',
            'Transformer': '#ff6b6b',
            'Static Baseline': '#9467bd',
            'Fixed Baseline': '#c5b0d5'
        }
        
        # 為每個模型生成資金曲線
        for model_type, models in self.model_groups.items():
            for model in models:
                if model in time_series_data:
                    data = time_series_data[model]
                    
                    # 獲取數據
                    mean_returns = np.array(data['cumulative_returns_mean']) * 100
                    std_returns = np.array(data['cumulative_returns_std']) * 100
                    
                    # 計算上下界
                    upper_bound = mean_returns + std_returns
                    lower_bound = mean_returns - std_returns
                    
                    # 獲取模型顏色
                    color = model_colors.get(model, '#666666')
                    
                    # 根據模型類型設置線型
                    linestyle = '-'
                    if 'Baseline' in model_type:
                        linestyle = '--'
                    elif 'Quantum' in model_type and 'Hybrid' not in model_type:
                        linestyle = '-.'
                    elif 'Transformer' in model_type:
                        linestyle = ':'
                    
                    # 繪製陰影區域
                    ax.fill_between(time_index, lower_bound, upper_bound, 
                                   alpha=0.15, color=color)
                    
                    # 繪製平均線
                    ax.plot(time_index, mean_returns, 
                           color=color, linewidth=2.5, alpha=0.9, 
                           linestyle=linestyle, label=model)
        
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加水平零線
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_models_equity_curve_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_apr_comparison_with_error_bars(self, stats_df):
        """創建帶誤差條的APR比較圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 準備數據
        models = stats_df.index.tolist()
        apr_means = stats_df['annual_return_mean'].values * 100
        apr_stds = stats_df['annual_return_std'].values * 100
        
        # 獲取模型類型顏色
        colors = []
        for model in models:
            model_type = self._get_model_type(model)
            color = self.colors.get(model_type.lower().replace(' ', '_'), '#666666')
            colors.append(color)
        
        # 1. 柱狀圖比較（帶誤差條）
        x_pos = np.arange(len(models))
        bars = ax1.bar(x_pos, apr_means, yerr=apr_stds, color=colors, alpha=0.8, 
                      capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        
        ax1.set_title('Annual Percentage Rate (APR) Comparison\n(with Error Bars)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('APR (%)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for i, (bar, mean, std) in enumerate(zip(bars, apr_means, apr_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                    f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. 按模型類型分組的箱線圖
        # 創建分組數據
        grouped_data = []
        for model in models:
            model_type = self._get_model_type(model)
            apr_mean = stats_df.loc[model, 'annual_return_mean'] * 100
            apr_std = stats_df.loc[model, 'annual_return_std'] * 100
            
            # 模擬數據點（基於均值和標準差）
            simulated_data = np.random.normal(apr_mean, apr_std, 100)
            for value in simulated_data:
                grouped_data.append({'Model_Type': model_type, 'APR': value})
        
        grouped_df = pd.DataFrame(grouped_data)
        
        sns.boxplot(data=grouped_df, x='Model_Type', y='APR', ax=ax2)
        ax2.set_title('APR Distribution by Model Type\n(with Uncertainty)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('APR (%)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'apr_comparison_with_error_bars.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_risk_return_scatter_with_uncertainty(self, stats_df):
        """創建帶不確定性的風險收益散點圖"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 準備數據
        models = stats_df.index.tolist()
        
        # 為每個模型分配不同顏色
        colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            color = colors[i]
            
            # 獲取數據
            apr_mean = stats_df.loc[model, 'annual_return_mean'] * 100
            apr_std = stats_df.loc[model, 'annual_return_std'] * 100
            vol_mean = stats_df.loc[model, 'volatility_mean'] * 100
            vol_std = stats_df.loc[model, 'volatility_std'] * 100
            sharpe_mean = stats_df.loc[model, 'sharpe_ratio_mean']
            sharpe_std = stats_df.loc[model, 'sharpe_ratio_std']
            
            # 繪製誤差橢圓
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((vol_mean, apr_mean), vol_std*2, apr_std*2, 
                            alpha=0.2, color=color)
            ax.add_patch(ellipse)
            
            # 繪製散點
            ax.scatter(vol_mean, apr_mean, s=sharpe_mean*100, c=[color], alpha=0.8, 
                      label=model, edgecolors='black', linewidth=0.5)
            
            # 添加誤差條
            ax.errorbar(vol_mean, apr_mean, xerr=vol_std, yerr=apr_std, 
                       fmt='none', color=color, alpha=0.5, capsize=3)
            
            # 添加模型名稱標籤
            ax.annotate(model, (vol_mean, apr_mean), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))
        
        # 添加Sharpe ratio參考線
        # 獲取x軸範圍來繪製參考線
        x_min, x_max = ax.get_xlim()
        x_range = np.linspace(x_min, x_max, 100)
        
        # Sharpe ratio = (Return - Risk-free rate) / Volatility
        # 假設無風險利率為0，所以 Return = Sharpe_ratio * Volatility
        sharpe_ratios = [0, 0.5, 1.0, 1.5, 2.0]
        
        for sr in sharpe_ratios:
            y_sharpe = sr * x_range
            ax.plot(x_range, y_sharpe, color='lightgray', linestyle='--', alpha=0.6, linewidth=1)
            
            # 在線的右端添加文字標籤
            ax.text(x_max * 0.95, sr * x_max * 0.95, f'Sharpe={sr}', 
                   fontsize=10, color='gray', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile: Model Comparison with Uncertainty\n(Size = Sharpe Ratio, Ellipses = ±1σ)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 設置圖例，分兩列顯示以節省空間
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', 
                 ncol=1, fontsize=8, title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_return_scatter_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_heatmap_with_uncertainty(self, stats_df):
        """創建帶不確定性的性能熱力圖"""
        # 選擇關鍵指標
        metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'volatility', 'win_rate']
        
        # 創建數據透視表（使用均值）
        pivot_data = pd.DataFrame(index=stats_df.index)
        for metric in metrics:
            pivot_data[metric] = stats_df[f'{metric}_mean']
        
        pivot_data['max_drawdown'] = abs(pivot_data['max_drawdown'])  # 轉換為正數
        
        # 轉換為百分比顯示
        display_data = pivot_data.copy()
        display_data['annual_return'] = display_data['annual_return'] * 100
        display_data['volatility'] = display_data['volatility'] * 100
        display_data['max_drawdown'] = display_data['max_drawdown'] * 100
        display_data['win_rate'] = display_data['win_rate'] * 100
        
        # 創建不確定性數據
        uncertainty_data = pd.DataFrame(index=stats_df.index)
        for metric in metrics:
            uncertainty_data[metric] = stats_df[f'{metric}_std']
        
        uncertainty_data['max_drawdown'] = uncertainty_data['max_drawdown']  # 保持原樣
        uncertainty_data['annual_return'] = uncertainty_data['annual_return'] * 100
        uncertainty_data['volatility'] = uncertainty_data['volatility'] * 100
        uncertainty_data['max_drawdown'] = uncertainty_data['max_drawdown'] * 100
        uncertainty_data['win_rate'] = uncertainty_data['win_rate'] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 均值熱力圖
        sns.heatmap(display_data.T, annot=True, cmap='RdYlBu_r', 
                   fmt='.1f', ax=ax1, cbar_kws={'label': 'Mean Value'})
        ax1.set_title('Model Performance Heatmap\n(Mean Values)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Performance Metrics')
        
        # 2. 標準差熱力圖
        sns.heatmap(uncertainty_data.T, annot=True, cmap='Reds', 
                   fmt='.2f', ax=ax2, cbar_kws={'label': 'Standard Deviation'})
        ax2.set_title('Model Performance Uncertainty\n(Standard Deviations)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Performance Metrics')
        
        # 設置Y軸標籤
        y_labels = ['APR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Volatility (%)', 'Win Rate (%)']
        ax1.set_yticklabels(y_labels)
        ax2.set_yticklabels(y_labels)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _get_model_type(self, model_name):
        """獲取模型類型"""
        for model_type, models in self.model_groups.items():
            if any(m in model_name for m in models):
                return model_type
        return 'Other'
    
    def generate_all_uncertainty_charts(self):
        """生成所有不確定性圖表"""
        print("🎯 開始生成不確定性圖表...")
        
        # 載入數據
        stats_df, time_series_data, raw_results = self.load_multiple_runs_data()
        
        if stats_df is None:
            print("❌ 無法載入多輪運行數據")
            return
        
        print(f"📊 載入 {len(stats_df)} 個模型的統計數據")
        
        # 生成各種圖表
        print("📈 生成帶不確定性的資金曲線比較圖...")
        self.create_equity_curves_with_uncertainty(time_series_data)
        
        print("📈 生成所有模型資金曲線圖（帶不確定性）...")
        self.create_all_models_equity_curve_with_uncertainty(time_series_data)
        
        print("📊 生成帶誤差條的APR比較圖...")
        self.create_apr_comparison_with_error_bars(stats_df)
        
        print("🎯 生成帶不確定性的風險收益散點圖...")
        self.create_risk_return_scatter_with_uncertainty(stats_df)
        
        print("🔥 生成帶不確定性的性能熱力圖...")
        self.create_performance_heatmap_with_uncertainty(stats_df)
        
        print(f"✅ 所有不確定性圖表已生成完成！")
        print(f"📁 輸出目錄: {self.output_dir}")

def main():
    """主函數"""
    generator = UncertaintyChartGenerator()
    generator.generate_all_uncertainty_charts()

if __name__ == "__main__":
    main()
