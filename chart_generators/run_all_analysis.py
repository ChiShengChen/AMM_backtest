#!/usr/bin/env python3
"""
Run All Analysis - 運行所有分析
一鍵運行所有圖表生成腳本
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description):
    """運行腳本並顯示結果"""
    print(f"\n🚀 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {description} - 完成")
            if result.stdout:
                print("輸出:", result.stdout[-200:])  # 顯示最後200個字符
        else:
            print(f"❌ {description} - 失敗")
            print("錯誤:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - 錯誤: {e}")
        return False
    
    return True

def main():
    """主函數"""
    print("🎯 運行所有圖表生成分析")
    print("=" * 60)
    
    # 定義腳本執行順序
    scripts = [
        ("run_steer_comparison.py", "Steer策略修復和基礎比較"),
        ("create_enhanced_comparison_charts.py", "增強版比較圖表"),
        ("create_improved_comparison_charts.py", "改進版圖表"),
        ("create_enhanced_efficiency_analysis.py", "最終版效率分析"),
        ("create_final_enhanced_report.py", "最終增強版報告")
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script, description in scripts:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"⚠️ 跳過後續腳本，因為 {script} 失敗")
            break
    
    print(f"\n📊 分析完成!")
    print(f"✅ 成功: {success_count}/{total_count}")
    print(f"❌ 失敗: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 所有分析都成功完成!")
        print("📁 結果保存在以下目錄:")
        print("  - steer_comparison_results/")
        print("  - simplified_ultimate_comparison/")
    else:
        print("\n⚠️ 部分分析失敗，請檢查錯誤信息")

if __name__ == "__main__":
    main()
