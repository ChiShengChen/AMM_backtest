#!/usr/bin/env python3
"""
分析不同策略的label定義和rebalancing標準差異
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LabelDifferenceAnalyzer:
    """Label差異分析器"""
    
    def __init__(self, output_dir="reports/label_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_qasa_vs_others(self):
        """分析QASA與其他策略的label差異"""
        
        analysis = {
            "QASA Benchmark": {
                "label_type": "價格變化預測",
                "label_definition": "預測下一個時間步的價格變化百分比",
                "label_formula": "y = (next_price - current_price) / current_price",
                "rebalance_criteria": "預測的價格變化絕對值 > 1%",
                "threshold": "0.01 (1%)",
                "prediction_target": "連續值 (價格變化率)",
                "model_output": "回歸預測",
                "decision_logic": "abs(prediction) > threshold",
                "training_data": "歷史價格序列",
                "features": "12個技術指標特徵",
                "model_type": "量子-經典混合神經網絡"
            },
            
            "AMM Quantum Strategy": {
                "label_type": "再平衡分類",
                "label_definition": "基於價格偏差的分類標籤",
                "label_formula": "y = 1 if price_deviation > threshold else 0",
                "rebalance_criteria": "量子模型預測 > 閾值",
                "threshold": "0.3 (30%)",
                "prediction_target": "二分類 (是否再平衡)",
                "model_output": "分類預測",
                "decision_logic": "prediction > rebalance_threshold",
                "training_data": "特徵工程後的市場數據",
                "features": "8個量子特徵",
                "model_type": "量子神經網絡"
            },
            
            "PennyLane Quantum": {
                "label_type": "價格變化分類",
                "label_definition": "基於價格變化幅度的分類標籤",
                "label_formula": "y = 1 if abs(price_change) > threshold else 0",
                "rebalance_criteria": "量子預測 = 1 且 概率 > 0.6",
                "threshold": "0.1 (10%)",
                "prediction_target": "二分類 (是否再平衡)",
                "model_output": "分類預測 + 概率",
                "decision_logic": "prediction == 1 and probability > 0.6",
                "training_data": "價格數據序列",
                "features": "4個量子特徵",
                "model_type": "PennyLane量子分類器"
            },
            
            "Steer Intent Classic": {
                "label_type": "位置管理",
                "label_definition": "基於布林帶位置的分類標籤",
                "label_formula": "y = 1 if |BB_position - 0.5| > 0.3 else 0",
                "rebalance_criteria": "多個觸發條件 (間隙、漂移、時間)",
                "threshold": "動態 (基於波動率)",
                "prediction_target": "二分類 (是否調整位置)",
                "model_output": "觸發器組合",
                "decision_logic": "多個觸發器OR邏輯",
                "training_data": "價格和技術指標",
                "features": "布林帶、ATR、EMA等",
                "model_type": "規則基礎策略"
            },
            
            "AMM Baseline": {
                "label_type": "價格偏差",
                "label_definition": "基於價格與移動平均線的偏差",
                "label_formula": "y = 1 if |price/MA - 1| > threshold else 0",
                "rebalance_criteria": "價格偏差 > 2%",
                "threshold": "0.02 (2%)",
                "prediction_target": "二分類 (是否再平衡)",
                "model_output": "簡單規則",
                "decision_logic": "price_deviation > threshold",
                "training_data": "歷史價格數據",
                "features": "移動平均線",
                "model_type": "規則基礎策略"
            }
        }
        
        return analysis
    
    def create_comparison_table(self, analysis):
        """創建比較表格"""
        df = pd.DataFrame(analysis).T
        
        # 重新排列列順序
        columns_order = [
            'label_type', 'label_definition', 'label_formula', 
            'rebalance_criteria', 'threshold', 'prediction_target',
            'model_output', 'decision_logic', 'model_type'
        ]
        
        df = df[columns_order]
        return df
    
    def create_visual_comparison(self, analysis):
        """創建視覺化比較"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Label and Rebalancing Criteria Comparison', fontsize=16, fontweight='bold')
        
        # 1. Label類型分布
        ax1 = axes[0, 0]
        label_types = [analysis[strategy]['label_type'] for strategy in analysis.keys()]
        label_counts = pd.Series(label_types).value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax1.pie(label_counts.values, labels=label_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Label Type Distribution')
        
        # 2. 閾值比較
        ax2 = axes[0, 1]
        thresholds = []
        strategies = []
        for strategy, data in analysis.items():
            threshold_str = data['threshold']
            # 提取數值
            try:
                if '%' in threshold_str:
                    threshold_val = float(threshold_str.split('(')[1].split('%')[0])
                elif '(' in threshold_str and ')' in threshold_str:
                    threshold_val = float(threshold_str.split('(')[1].split(')')[0])
                else:
                    # 處理特殊情況
                    if '動態' in threshold_str:
                        threshold_val = 5.0  # 給動態閾值一個代表值
                    else:
                        threshold_val = 0.0
                thresholds.append(threshold_val)
                strategies.append(strategy)
            except (ValueError, IndexError):
                # 處理無法解析的閾值
                thresholds.append(0.0)
                strategies.append(strategy)
        
        bars = ax2.bar(range(len(strategies)), thresholds, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax2.set_title('Rebalancing Thresholds Comparison')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Threshold (%)')
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels([s.replace(' ', '\n') for s in strategies], rotation=45, ha='right')
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. 預測目標類型
        ax3 = axes[1, 0]
        prediction_targets = [analysis[strategy]['prediction_target'] for strategy in analysis.keys()]
        target_counts = pd.Series(prediction_targets).value_counts()
        
        bars = ax3.bar(target_counts.index, target_counts.values, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Prediction Target Types')
        ax3.set_xlabel('Target Type')
        ax3.set_ylabel('Number of Strategies')
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 4. 模型類型分布
        ax4 = axes[1, 1]
        model_types = [analysis[strategy]['model_type'] for strategy in analysis.keys()]
        model_counts = pd.Series(model_types).value_counts()
        
        wedges, texts, autotexts = ax4.pie(model_counts.values, labels=model_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Model Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_criteria_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_analysis_chart(self, analysis):
        """創建詳細分析圖表"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 創建策略比較矩陣
        strategies = list(analysis.keys())
        criteria = ['label_type', 'prediction_target', 'threshold', 'model_type']
        
        # 創建數據矩陣
        matrix_data = []
        for strategy in strategies:
            row = []
            for criterion in criteria:
                value = analysis[strategy][criterion]
                # 簡化值用於顯示
                if criterion == 'threshold':
                    value = value.split('(')[1].split(')')[0] if '(' in value else value
                elif criterion == 'model_type':
                    value = value.split(' ')[0]  # 只取第一個詞
                row.append(value)
            matrix_data.append(row)
        
        # 創建熱力圖
        df_matrix = pd.DataFrame(matrix_data, index=strategies, columns=criteria)
        
        # 創建數值編碼的矩陣用於熱力圖
        # 將字符串轉換為數值編碼
        df_numeric = df_matrix.copy()
        unique_values = {}
        value_counter = 0
        
        for col in df_numeric.columns:
            for idx in df_numeric.index:
                value = df_numeric.loc[idx, col]
                if value not in unique_values:
                    unique_values[value] = value_counter
                    value_counter += 1
                df_numeric.loc[idx, col] = unique_values[value]
        
        df_numeric = df_numeric.astype(float)
        
        # 使用seaborn創建熱力圖
        sns.heatmap(df_numeric, annot=df_matrix, fmt='s', cmap='Set3', 
                   cbar_kws={'label': 'Strategy Characteristics'}, ax=ax)
        
        ax.set_title('Strategy Label and Rebalancing Criteria Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Strategy')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_criteria_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self, analysis):
        """生成分析報告"""
        report_path = self.output_dir / "label_differences_analysis.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 策略Label定義和Rebalancing標準差異分析\n\n")
            
            f.write("## 🎯 核心發現\n\n")
            f.write("QASA Benchmark與其他策略在label定義和rebalancing標準上存在顯著差異：\n\n")
            
            f.write("### 1. Label類型差異\n")
            f.write("- **QASA**: 價格變化預測 (回歸問題)\n")
            f.write("- **其他策略**: 再平衡分類 (分類問題)\n\n")
            
            f.write("### 2. 預測目標差異\n")
            f.write("- **QASA**: 預測連續的價格變化率\n")
            f.write("- **其他策略**: 預測二分類的再平衡決策\n\n")
            
            f.write("### 3. 閾值設定差異\n")
            f.write("- **QASA**: 1% 價格變化閾值\n")
            f.write("- **AMM Quantum**: 30% 預測閾值\n")
            f.write("- **PennyLane**: 10% 價格變化 + 60% 概率閾值\n")
            f.write("- **Steer Intent**: 動態閾值 (基於波動率)\n")
            f.write("- **AMM Baseline**: 2% 價格偏差閾值\n\n")
            
            f.write("## 📊 詳細比較\n\n")
            
            for strategy, data in analysis.items():
                f.write(f"### {strategy}\n\n")
                f.write(f"- **Label類型**: {data['label_type']}\n")
                f.write(f"- **Label定義**: {data['label_definition']}\n")
                f.write(f"- **Label公式**: `{data['label_formula']}`\n")
                f.write(f"- **再平衡標準**: {data['rebalance_criteria']}\n")
                f.write(f"- **閾值**: {data['threshold']}\n")
                f.write(f"- **預測目標**: {data['prediction_target']}\n")
                f.write(f"- **模型輸出**: {data['model_output']}\n")
                f.write(f"- **決策邏輯**: `{data['decision_logic']}`\n")
                f.write(f"- **模型類型**: {data['model_type']}\n\n")
            
            f.write("## 🔍 關鍵差異分析\n\n")
            f.write("### QASA的特殊性\n")
            f.write("1. **回歸vs分類**: QASA是唯一使用回歸預測的策略\n")
            f.write("2. **連續預測**: 預測連續的價格變化而非離散的再平衡決策\n")
            f.write("3. **低閾值**: 1%的閾值比其他策略更敏感\n")
            f.write("4. **混合架構**: 量子-經典混合神經網絡\n\n")
            
            f.write("### 其他策略的共同點\n")
            f.write("1. **分類問題**: 都是二分類的再平衡決策\n")
            f.write("2. **規則基礎**: 大部分基於技術指標的規則\n")
            f.write("3. **較高閾值**: 2%-30%的閾值範圍\n")
            f.write("4. **特徵工程**: 基於技術指標的特徵\n\n")
            
            f.write("## 💡 影響分析\n\n")
            f.write("### 對比較結果的影響\n")
            f.write("1. **不公平比較**: 不同的label定義導致比較不公平\n")
            f.write("2. **閾值敏感性**: QASA的低閾值可能導致過度交易\n")
            f.write("3. **模型複雜度**: 回歸問題比分類問題更複雜\n")
            f.write("4. **特徵需求**: 不同策略需要不同的特徵工程\n\n")
            
            f.write("### 建議改進\n")
            f.write("1. **統一label定義**: 所有策略使用相同的label標準\n")
            f.write("2. **標準化閾值**: 使用相同的閾值進行比較\n")
            f.write("3. **公平比較**: 確保所有策略解決相同的問題\n")
            f.write("4. **特徵一致性**: 使用相同的特徵工程方法\n\n")
    
    def run_analysis(self):
        """運行完整分析"""
        print("🔍 開始分析策略label定義和rebalancing標準差異...")
        
        # 分析QASA與其他策略的差異
        analysis = self.analyze_qasa_vs_others()
        
        # 創建比較表格
        df = self.create_comparison_table(analysis)
        df.to_csv(self.output_dir / 'label_comparison_table.csv')
        print(f"📊 比較表格已保存: {self.output_dir / 'label_comparison_table.csv'}")
        
        # 創建視覺化比較
        print("📈 生成視覺化比較圖表...")
        self.create_visual_comparison(analysis)
        self.create_detailed_analysis_chart(analysis)
        
        # 生成分析報告
        print("📝 生成分析報告...")
        self.generate_analysis_report(analysis)
        
        print(f"✅ 分析完成！結果保存在: {self.output_dir}")
        
        return analysis, df

def main():
    """主函數"""
    analyzer = LabelDifferenceAnalyzer()
    analysis, df = analyzer.run_analysis()
    
    print("\n📋 策略比較摘要:")
    print(df[['label_type', 'prediction_target', 'threshold']].to_string())

if __name__ == "__main__":
    main()
