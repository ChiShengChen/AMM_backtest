#!/usr/bin/env python3
"""
清理重複的模型數據
"""

import pandas as pd

def clean_duplicate_models():
    """清理重複的模型數據"""
    
    # 讀取數據
    df = pd.read_csv('reports/model_performance_charts/model_performance_data.csv')
    
    print(f"📊 清理前: {len(df)} 個模型")
    print("重複的模型:")
    print(df[df.duplicated(subset=['Model'], keep=False)]['Model'].value_counts())
    
    # 移除重複，保留第一個
    df_cleaned = df.drop_duplicates(subset=['Model'], keep='first')
    
    print(f"📊 清理後: {len(df_cleaned)} 個模型")
    
    # 保存清理後的數據
    df_cleaned.to_csv('reports/model_performance_charts/model_performance_data.csv', index=False)
    print("✅ 重複模型已清理")

if __name__ == "__main__":
    clean_duplicate_models()
