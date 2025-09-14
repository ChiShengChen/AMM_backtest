#!/usr/bin/env python3
"""
生成所有模型性能比較圖表
整合回測結果並生成APR、資金曲線、風險等比較圖
"""

import sys
import os
from pathlib import Path

# 添加當前目錄到Python路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from integrate_backtest_results import BacktestResultsIntegrator
from create_model_performance_charts import ModelPerformanceChartGenerator

def main():
    """主函數"""
    print("🎯 開始生成所有模型性能比較圖表...")
    
    # 1. 整合回測結果
    print("\n📊 步驟1: 整合回測結果數據...")
    integrator = BacktestResultsIntegrator()
    
    # 加載各種結果
    integrator.load_amm_results()
    integrator.load_steer_results()
    integrator.load_unified_comparison_results()
    
    # 創建標準化結果
    df = integrator.create_standardized_results()
    
    # 保存整合結果
    csv_path, json_path = integrator.save_results(df)
    
    # 2. 生成性能圖表
    print("\n📈 步驟2: 生成模型性能比較圖表...")
    chart_generator = ModelPerformanceChartGenerator()
    
    # 使用整合的數據生成圖表
    print("📊 生成資金曲線比較圖...")
    chart_generator.create_equity_curves(df)
    
    print("📊 生成APR比較圖...")
    chart_generator.create_apr_comparison(df)
    
    print("🎯 生成風險收益散點圖...")
    chart_generator.create_risk_return_scatter(df)
    
    print("📉 生成回撤分析圖...")
    chart_generator.create_drawdown_analysis(df)
    
    print("🔥 生成性能熱力圖...")
    chart_generator.create_performance_heatmap(df)
    
    print("📋 生成綜合儀表板...")
    chart_generator.create_comprehensive_dashboard(df)
    
    # 3. 生成摘要報告
    print("\n📝 步驟3: 生成摘要報告...")
    generate_summary_report(df, chart_generator.output_dir)
    
    print(f"\n✅ 所有圖表生成完成！")
    print(f"📁 輸出目錄: {chart_generator.output_dir}")
    print(f"📊 數據文件: {csv_path}")
    
    return df

def generate_summary_report(df, output_dir):
    """生成摘要報告"""
    report_path = Path(output_dir) / "performance_summary_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 模型性能比較摘要報告\n\n")
        
        # 基本統計
        f.write("## 📊 基本統計\n\n")
        f.write(f"- **總模型數量**: {len(df)}\n")
        f.write(f"- **模型類型**: {', '.join(df['Model_Type'].unique())}\n")
        f.write(f"- **數據來源**: {', '.join(df['Source'].unique())}\n\n")
        
        # 模型類型分布
        f.write("## 🏷️ 模型類型分布\n\n")
        type_counts = df['Model_Type'].value_counts()
        for model_type, count in type_counts.items():
            f.write(f"- **{model_type}**: {count} 個模型\n")
        f.write("\n")
        
        # 性能摘要
        f.write("## 📈 性能摘要\n\n")
        summary = df.groupby('Model_Type')[['APR', 'Sharpe_Ratio', 'Max_Drawdown', 'Volatility']].mean()
        
        f.write("| 模型類型 | APR (%) | 夏普比率 | 最大回撤 (%) | 波動率 (%) |\n")
        f.write("|----------|---------|----------|--------------|------------|\n")
        
        for model_type, row in summary.iterrows():
            f.write(f"| {model_type} | {row['APR']*100:.2f} | {row['Sharpe_Ratio']:.2f} | {abs(row['Max_Drawdown'])*100:.2f} | {row['Volatility']*100:.2f} |\n")
        
        f.write("\n")
        
        # 最佳表現模型
        f.write("## 🏆 最佳表現模型\n\n")
        
        best_apr = df.loc[df['APR'].idxmax()]
        f.write(f"- **最高APR**: {best_apr['Model']} ({best_apr['APR']*100:.2f}%)\n")
        
        best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
        f.write(f"- **最高夏普比率**: {best_sharpe['Model']} ({best_sharpe['Sharpe_Ratio']:.2f})\n")
        
        best_drawdown = df.loc[df['Max_Drawdown'].idxmax()]  # 最大回撤（負值）的最大值
        f.write(f"- **最低回撤**: {best_drawdown['Model']} ({abs(best_drawdown['Max_Drawdown'])*100:.2f}%)\n")
        
        f.write("\n")
        
        # 圖表說明
        f.write("## 📊 生成的圖表\n\n")
        f.write("1. **equity_curves_comparison.png** - 資金曲線比較圖\n")
        f.write("2. **apr_comparison.png** - APR比較圖\n")
        f.write("3. **risk_return_scatter.png** - 風險收益散點圖\n")
        f.write("4. **drawdown_analysis.png** - 回撤分析圖\n")
        f.write("5. **performance_heatmap.png** - 性能熱力圖\n")
        f.write("6. **comprehensive_dashboard.png** - 綜合儀表板\n")
        
        f.write("\n")
        f.write("## 📁 文件說明\n\n")
        f.write("- **integrated_model_performance.csv** - 整合的性能數據\n")
        f.write("- **integrated_model_performance.json** - JSON格式的性能數據\n")
        f.write("- **performance_summary_report.md** - 本摘要報告\n")
    
    print(f"📝 摘要報告已保存到: {report_path}")

if __name__ == "__main__":
    main()
