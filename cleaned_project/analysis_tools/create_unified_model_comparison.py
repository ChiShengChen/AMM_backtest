#!/usr/bin/env python3
"""
統一模型比較圖表生成器
整合專案中所有模型的回測結果，生成統一的比較圖表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_results():
    """載入所有模型的回測結果"""
    results = []
    
    # 1. AMM Rebalance Backtester 結果
    amm_file = "amm-rebalance-backtester/reports/5year_backtest/backtest_results.csv"
    if Path(amm_file).exists():
        amm_df = pd.read_csv(amm_file)
        amm_df['Project'] = 'AMM Rebalance'
        amm_df['Model_Components'] = amm_df['Strategy'].map({
            'QuantumBased': 'QNN_Rebalance_Predictor_(Qiskit_VQC)',
            'QuantumVolatility': 'QNN_Volatility_Predictor_(Qiskit_VQC)', 
            'QuantumHybrid': 'QNN_Rebalance_+_QSVM_Volatility_(Qiskit)',
            'MLBased': 'RandomForest_Classifier_(100_trees)',
            'MLVolatility': 'GradientBoosting_Regressor_(100_estimators)',
            'MLHybrid': 'RandomForest_+_GradientBoosting_Ensemble'
        })
        results.append(amm_df)
    
    # 2. Steer Intent Backtester 結果
    steer_file = "steer_intent_backtester/reports/5year_backtest/backtest_results.csv"
    if Path(steer_file).exists():
        steer_df = pd.read_csv(steer_file)
        steer_df['Project'] = 'Steer Intent'
        steer_df['Model_Components'] = steer_df['Strategy'].map({
            'QuantumBollinger': 'QNN_Bollinger_Bands_Predictor_(Qiskit_VQC)',
            'QuantumKeltner': 'QNN_Keltner_Channels_Predictor_(Qiskit_VQC)',
            'QuantumHybrid': 'QNN_Bollinger_+_Keltner_Hybrid_(Qiskit)',
            'MLBollinger': 'RandomForest_Classifier_(100_trees)_+_Bollinger_Bands',
            'MLKeltner': 'RandomForest_Classifier_(100_trees)_+_Keltner_Channels',
            'MLHybrid': 'RandomForest_+_GradientBoosting_Ensemble_+_Hybrid_Strategy'
        })
        results.append(steer_df)
    
    # 3. QASA Benchmark 結果
    qasa_file = "reports/simplified_qasa_benchmark/simplified_qasa_benchmark_results.csv"
    if Path(qasa_file).exists():
        qasa_df = pd.read_csv(qasa_file)
        qasa_df['Project'] = 'QASA Benchmark'
        qasa_df['Symbol'] = qasa_df['Symbol'].str.replace('_1d_20200905_20250903', '')
        results.append(qasa_df)
    
    # 4. PennyLane 結果
    pennylane_files = [
        "reports/final_pennylane_backtest/final_pennylane_backtest_results.csv",
        "reports/optimized_pennylane_backtest/optimized_pennylane_backtest_results.csv",
        "reports/simple_pennylane_backtest/simple_pennylane_backtest_results.csv"
    ]
    
    for file in pennylane_files:
        if Path(file).exists():
            pl_df = pd.read_csv(file)
            pl_df['Project'] = 'PennyLane Quantum'
            pl_df['Symbol'] = pl_df['Symbol'].str.replace('_1d_20200905_20250903', '')
            results.append(pl_df)
    
    # 合併所有結果
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def create_unified_comparison_charts(df):
    """創建統一的模型比較圖表"""
    
    # 設置圖表樣式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 創建大圖表
    fig = plt.figure(figsize=(20, 24))
    
    # 1. 收益比較 - 按資產分組
    ax1 = plt.subplot(4, 2, 1)
    asset_returns = df.groupby(['Symbol', 'Type'])['Return_Pct'].mean().unstack()
    asset_returns.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Average Returns by Asset (by Model Type)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Return (%)')
    ax1.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 夏普比率比較 - 按資產分組
    ax2 = plt.subplot(4, 2, 2)
    asset_sharpe = df.groupby(['Symbol', 'Type'])['Sharpe_Ratio'].mean().unstack()
    asset_sharpe.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Average Sharpe Ratio by Asset (by Model Type)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Sharpe Ratio')
    ax2.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 項目間比較 - 收益
    ax3 = plt.subplot(4, 2, 3)
    project_returns = df.groupby(['Project', 'Type'])['Return_Pct'].mean().unstack()
    project_returns.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Average Returns by Project', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Return (%)')
    ax3.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 項目間比較 - 夏普比率
    ax4 = plt.subplot(4, 2, 4)
    project_sharpe = df.groupby(['Project', 'Type'])['Sharpe_Ratio'].mean().unstack()
    project_sharpe.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Average Sharpe Ratio by Project', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Sharpe Ratio')
    ax4.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 散點圖 - 收益 vs 夏普比率
    ax5 = plt.subplot(4, 2, 5)
    colors = {'Classic ML': 'blue', 'Quantum': 'red', 'Simplified_QASA': 'green', 'Final_Quantum': 'orange'}
    for model_type in df['Type'].unique():
        if pd.notna(model_type):
            subset = df[df['Type'] == model_type]
            ax5.scatter(subset['Return_Pct'], subset['Sharpe_Ratio'], 
                       c=colors.get(model_type, 'gray'), label=model_type, alpha=0.7, s=60)
    
    ax5.set_xlabel('Return (%)')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Return vs Sharpe Ratio Scatter Plot', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 再平衡次數比較
    ax6 = plt.subplot(4, 2, 6)
    rebalance_data = df.groupby(['Symbol', 'Type'])['Rebalances'].mean().unstack()
    rebalance_data.plot(kind='bar', ax=ax6, width=0.8)
    ax6.set_title('Average Rebalances by Asset', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Average Rebalances')
    ax6.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. 模型性能排名 (前15名)
    ax7 = plt.subplot(4, 2, 7)
    top_models = df.nlargest(15, 'Return_Pct')[['Strategy', 'Symbol', 'Return_Pct', 'Type']]
    y_pos = np.arange(len(top_models))
    colors = [colors.get(t, 'gray') for t in top_models['Type']]
    
    bars = ax7.barh(y_pos, top_models['Return_Pct'], color=colors, alpha=0.7)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels([f"{row['Strategy']} ({row['Symbol']})" for _, row in top_models.iterrows()], fontsize=8)
    ax7.set_xlabel('Return (%)')
    ax7.set_title('Model Performance Ranking (Top 15)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. 模型類型統計
    ax8 = plt.subplot(4, 2, 8)
    type_stats = df.groupby('Type').agg({
        'Return_Pct': ['mean', 'std', 'count'],
        'Sharpe_Ratio': 'mean'
    }).round(2)
    
    # 創建統計表格
    stats_text = "Model Type Statistics:\n\n"
    for model_type in type_stats.index:
        if pd.notna(model_type):
            mean_return = type_stats.loc[model_type, ('Return_Pct', 'mean')]
            std_return = type_stats.loc[model_type, ('Return_Pct', 'std')]
            count = type_stats.loc[model_type, ('Return_Pct', 'count')]
            mean_sharpe = type_stats.loc[model_type, ('Sharpe_Ratio', 'mean')]
            stats_text += f"{model_type}:\n"
            stats_text += f"  Avg Return: {mean_return:.1f}% ± {std_return:.1f}%\n"
            stats_text += f"  Avg Sharpe: {mean_sharpe:.3f}\n"
            stats_text += f"  Model Count: {count}\n\n"
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Model Type Statistics Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_detailed_ranking_table(df):
    """創建詳細的模型排名表"""
    
    # 按收益排序
    df_sorted = df.sort_values('Return_Pct', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # 選擇重要列
    ranking_df = df_sorted[['Rank', 'Strategy', 'Symbol', 'Project', 'Type', 
                           'Return_Pct', 'Sharpe_Ratio', 'Rebalances', 'Model_Components']].copy()
    
    # 格式化數值
    ranking_df['Return_Pct'] = ranking_df['Return_Pct'].round(2)
    ranking_df['Sharpe_Ratio'] = ranking_df['Sharpe_Ratio'].round(3)
    ranking_df['Rebalances'] = ranking_df['Rebalances'].astype(int)
    
    return ranking_df

def main():
    """主函數"""
    print("🚀 Starting unified model comparison chart generation...")
    
    # 載入所有結果
    df = load_all_results()
    
    if df.empty:
        print("❌ No backtest result files found")
        return
    
    print(f"✅ Successfully loaded {len(df)} model results")
    print(f"📊 Projects included: {', '.join(df['Project'].unique())}")
    print(f"🎯 Assets included: {', '.join(df['Symbol'].unique())}")
    print(f"🤖 Model types included: {', '.join(df['Type'].unique())}")
    
    # 創建輸出目錄
    output_dir = Path("reports/unified_model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成統一比較圖表
    print("📈 Generating unified comparison charts...")
    fig = create_unified_comparison_charts(df)
    
    # 保存圖表
    chart_path = output_dir / "unified_model_comparison.png"
    fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Chart saved: {chart_path}")
    
    # 生成詳細排名表
    print("📋 Generating detailed ranking table...")
    ranking_df = create_detailed_ranking_table(df)
    
    # 保存排名表
    ranking_path = output_dir / "model_ranking_table.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding='utf-8')
    print(f"✅ Ranking table saved: {ranking_path}")
    
    # 生成統計摘要
    print("📊 Generating statistical summary...")
    summary_stats = {
        'total_models': len(df),
        'projects': df['Project'].nunique(),
        'assets': df['Symbol'].nunique(),
        'model_types': df['Type'].nunique(),
        'best_return': df['Return_Pct'].max(),
        'best_sharpe': df['Sharpe_Ratio'].max(),
        'avg_return': df['Return_Pct'].mean(),
        'avg_sharpe': df['Sharpe_Ratio'].mean()
    }
    
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("🎯 Unified Model Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Models: {summary_stats['total_models']}\n")
        f.write(f"Projects: {summary_stats['projects']}\n")
        f.write(f"Assets: {summary_stats['assets']}\n")
        f.write(f"Model Types: {summary_stats['model_types']}\n\n")
        f.write(f"Best Return: {summary_stats['best_return']:.2f}%\n")
        f.write(f"Best Sharpe Ratio: {summary_stats['best_sharpe']:.3f}\n")
        f.write(f"Average Return: {summary_stats['avg_return']:.2f}%\n")
        f.write(f"Average Sharpe Ratio: {summary_stats['avg_sharpe']:.3f}\n\n")
        
        f.write("🏆 Top 10 Models:\n")
        f.write("-" * 30 + "\n")
        for _, row in ranking_df.head(10).iterrows():
            f.write(f"{row['Rank']:2d}. {row['Strategy']} ({row['Symbol']}) - {row['Return_Pct']:.2f}%\n")
    
    print(f"✅ Statistical summary saved: {summary_path}")
    
    # 顯示前10名結果
    print("\n🏆 Top 10 Models:")
    print("-" * 80)
    for _, row in ranking_df.head(10).iterrows():
        print(f"{row['Rank']:2d}. {row['Strategy']:20s} ({row['Symbol']:8s}) - {row['Return_Pct']:6.2f}% (Sharpe: {row['Sharpe_Ratio']:5.3f})")
    
    print(f"\n✅ Unified model comparison completed! Results saved in: {output_dir}")
    
    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    main()
