#!/usr/bin/env python3
"""
修復圖表生成器的腳本
"""

import re

def fix_plot_generator():
    """修復 plots.py 中的策略結果讀取問題"""
    
    # 讀取原始文件
    with open('src/reporting/plots.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復策略結果讀取邏輯
    old_pattern = r"""        strategies = results\.get\('strategies', \{\}\)
        if not strategies:
            logger\.warning\("No strategy results found"\)
            return"""
    
    new_pattern = """        # 嘗試從不同格式讀取策略結果
        strategies = results.get('strategies', {})
        if not strategies:
            # 嘗試從 summary DataFrame 讀取
            if 'summary' in results and isinstance(results['summary'], pd.DataFrame):
                strategies = {}
                for _, row in results['summary'].iterrows():
                    strategies[row['strategy']] = {
                        'apr': row['apr'],
                        'mdd': row['mdd'],
                        'sharpe': row.get('sharpe', 0),
                        'calmar': row.get('calmar', 0),
                        'rebalance_count': row.get('rebalance_count', 0)
                    }
        
        if not strategies:
            logger.warning("No strategy results found")
            return"""
    
    # 替換所有出現的地方
    fixed_content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)
    
    # 寫回文件
    with open('src/reporting/plots.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ 圖表生成器修復完成！")

if __name__ == "__main__":
    fix_plot_generator()
