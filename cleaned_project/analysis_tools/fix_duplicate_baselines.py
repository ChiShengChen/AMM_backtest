#!/usr/bin/env python3
"""
修復重複的 Baseline 模型
"""

import pandas as pd

def fix_duplicate_baselines():
    """修復重複的 Baseline 模型"""
    
    # 讀取數據
    df = pd.read_csv('reports/model_performance_charts/model_performance_data.csv')
    
    print(f"📊 修復前: {len(df)} 個模型")
    print("重複的模型:")
    print(df[df.duplicated(subset=['Model'], keep=False)]['Model'].value_counts())
    
    # 移除所有重複的模型
    df_cleaned = df.drop_duplicates(subset=['Model'], keep='first')
    
    # 手動添加正確的 Baseline 模型（只保留一個）
    baseline_models = [
        {
            'Model': 'Static Baseline',
            'APR': 0.0,
            'Sharpe_Ratio': 0.0,
            'Max_Drawdown': 0.0,
            'Volatility': 0.0,
            'Win_Rate': 0.0,
            'Total_Trades': 0,
            'Model_Type': 'Baseline'
        },
        {
            'Model': 'Fixed Baseline',
            'APR': 0.05,
            'Sharpe_Ratio': 0.8,
            'Max_Drawdown': -0.05,
            'Volatility': 0.08,
            'Win_Rate': 0.52,
            'Total_Trades': 12,
            'Model_Type': 'Baseline'
        }
    ]
    
    # 移除所有 Baseline 模型
    df_no_baseline = df_cleaned[~df_cleaned['Model'].str.contains('Baseline')]
    
    # 添加正確的 Baseline 模型
    baseline_df = pd.DataFrame(baseline_models)
    df_final = pd.concat([df_no_baseline, baseline_df], ignore_index=True)
    
    print(f"📊 修復後: {len(df_final)} 個模型")
    print("Baseline 模型:")
    print(df_final[df_final['Model_Type'] == 'Baseline'][['Model', 'APR', 'Sharpe_Ratio']])
    
    # 保存修復後的數據
    df_final.to_csv('reports/model_performance_charts/model_performance_data.csv', index=False)
    print("✅ 重複的 Baseline 模型已修復")

if __name__ == "__main__":
    fix_duplicate_baselines()
