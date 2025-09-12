#!/usr/bin/env python3
"""
驗證下載的5年日線數據質量
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def validate_data_file(filepath):
    """驗證單個數據文件"""
    print(f"\n{'='*60}")
    print(f"驗證文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # 讀取數據
        df = pd.read_csv(filepath)
        
        # 基本信息
        print(f"📊 基本信息:")
        print(f"  記錄數量: {len(df):,}")
        print(f"  文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        # 時間範圍檢查
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_range = (df['timestamp'].min(), df['timestamp'].max())
        time_span = time_range[1] - time_range[0]
        
        print(f"\n⏰ 時間範圍:")
        print(f"  開始時間: {time_range[0]}")
        print(f"  結束時間: {time_range[1]}")
        print(f"  總跨度: {time_span.days} 天")
        
        # 數據完整性檢查
        print(f"\n🔍 數據完整性:")
        missing_values = df.isnull().sum().sum()
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        print(f"  缺失值: {missing_values}")
        print(f"  重複時間戳: {duplicate_timestamps}")
        
        # 價格數據檢查
        print(f"\n💰 價格數據:")
        price_range = (df['low'].min(), df['high'].max())
        print(f"  最低價: ${price_range[0]:.2f}")
        print(f"  最高價: ${price_range[1]:.2f}")
        print(f"  價格範圍: ${price_range[1] - price_range[0]:.2f}")
        
        # 交易量檢查
        print(f"\n📈 交易量數據:")
        total_volume = df['volume'].sum()
        avg_volume = df['volume'].mean()
        print(f"  總交易量: {total_volume:,.2f}")
        print(f"  平均交易量: {avg_volume:,.2f}")
        
        # 數據一致性檢查
        print(f"\n✅ 數據一致性:")
        price_consistency = all(df['low'] <= df['high']) and all(df['low'] <= df['close']) and all(df['close'] <= df['high'])
        volume_consistency = all(df['volume'] >= 0)
        print(f"  價格一致性: {'✅' if price_consistency else '❌'}")
        print(f"  交易量一致性: {'✅' if volume_consistency else '❌'}")
        
        # 時間間隔檢查
        time_gaps = df['timestamp'].diff().dropna()
        expected_gap = pd.Timedelta(days=1)
        gap_consistency = (time_gaps == expected_gap).mean()
        print(f"  時間間隔一致性: {gap_consistency:.1%}")
        
        # 計算年化收益率
        if len(df) > 1:
            start_price = df.iloc[0]['close']
            end_price = df.iloc[-1]['close']
            years = time_span.days / 365.25
            annual_return = ((end_price / start_price) ** (1/years) - 1) * 100
            print(f"\n📊 年化收益率: {annual_return:.2f}%")
        
        # 數據質量評分
        quality_score = 0
        if missing_values == 0: quality_score += 20
        if duplicate_timestamps == 0: quality_score += 20
        if price_consistency: quality_score += 20
        if volume_consistency: quality_score += 20
        if gap_consistency > 0.95: quality_score += 20
        
        print(f"\n🏆 數據質量評分: {quality_score}/100")
        
        if quality_score >= 80:
            print("  🎉 數據質量優秀！")
        elif quality_score >= 60:
            print("  ✅ 數據質量良好")
        else:
            print("  ⚠️ 數據質量需要改進")
        
        return {
            'filepath': filepath,
            'quality_score': quality_score,
            'record_count': len(df),
            'time_span_days': time_span.days,
            'missing_values': missing_values,
            'duplicate_timestamps': duplicate_timestamps
        }
        
    except Exception as e:
        print(f"❌ 驗證失敗: {e}")
        return None

def create_summary_report(validation_results):
    """創建總結報告"""
    print(f"\n{'='*80}")
    print("📋 數據驗證總結報告")
    print(f"{'='*80}")
    
    if not validation_results:
        print("❌ 沒有有效的驗證結果")
        return
    
    # 統計信息
    total_files = len(validation_results)
    successful_validations = len([r for r in validation_results if r is not None])
    avg_quality_score = np.mean([r['quality_score'] for r in validation_results if r is not None])
    total_records = sum([r['record_count'] for r in validation_results if r is not None])
    total_days = sum([r['time_span_days'] for r in validation_results if r is not None])
    
    print(f"📊 總體統計:")
    print(f"  總文件數: {total_files}")
    print(f"  成功驗證: {successful_validations}")
    print(f"  平均質量評分: {avg_quality_score:.1f}/100")
    print(f"  總記錄數: {total_records:,}")
    print(f"  總時間跨度: {total_days:,} 天")
    
    # 各文件詳細信息
    print(f"\n📁 文件詳細信息:")
    for result in validation_results:
        if result:
            filename = os.path.basename(result['filepath'])
            print(f"  {filename}:")
            print(f"    質量評分: {result['quality_score']}/100")
            print(f"    記錄數: {result['record_count']:,}")
            print(f"    時間跨度: {result['time_span_days']} 天")
    
    # 建議
    print(f"\n💡 建議:")
    if avg_quality_score >= 80:
        print("  🎉 所有數據質量優秀，可以直接用於回測！")
    elif avg_quality_score >= 60:
        print("  ✅ 數據質量良好，建議檢查個別問題後使用")
    else:
        print("  ⚠️ 數據質量需要改進，建議重新下載")

def main():
    """主函數"""
    print("🔍 開始驗證下載的5年日線數據...")
    
    # 數據目錄
    data_dir = "data/5year_daily"
    
    if not os.path.exists(data_dir):
        print(f"❌ 數據目錄不存在: {data_dir}")
        return
    
    # 獲取所有 CSV 文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ 在 {data_dir} 中沒有找到 CSV 文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 個數據文件")
    
    # 驗證每個文件
    validation_results = []
    for csv_file in sorted(csv_files):
        filepath = os.path.join(data_dir, csv_file)
        result = validate_data_file(filepath)
        validation_results.append(result)
    
    # 創建總結報告
    create_summary_report(validation_results)
    
    print(f"\n🎯 驗證完成！數據已準備好用於 AMM 回測")

if __name__ == "__main__":
    main()
