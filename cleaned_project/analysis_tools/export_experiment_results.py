#!/usr/bin/env python3
"""
導出實驗結果數據到各種格式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExperimentResultsExporter:
    """實驗結果導出器"""
    
    def __init__(self, output_dir="reports/exported_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """載入所有實驗數據"""
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
            print(f"❌ 未找到數據文件: {e}")
            return None, None, None
    
    def export_summary_statistics(self, stats_df):
        """導出摘要統計數據"""
        print("📊 導出摘要統計數據...")
        
        # 基本統計摘要
        summary_stats = {
            'total_models': len(stats_df),
            'total_runs': stats_df['n_runs'].sum(),
            'metrics': list(stats_df.columns),
            'models': list(stats_df.index)
        }
        
        # 按模型類型分組的統計
        model_groups = {
            'Classic ML': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
            'Quantum ML': ['VQE Classifier', 'QNN', 'QSVM'],
            'Hybrid Quantum': ['QASA Hybrid', 'LSTM_QNN', 'QuantumRWKV'],
            'Transformer': ['Transformer']
        }
        
        group_stats = {}
        for group_name, models in model_groups.items():
            available_models = [m for m in models if m in stats_df.index]
            if available_models:
                group_data = stats_df.loc[available_models]
                group_stats[group_name] = {
                    'model_count': len(available_models),
                    'avg_annual_return': group_data['annual_return_mean'].mean(),
                    'avg_volatility': group_data['volatility_mean'].mean(),
                    'avg_sharpe_ratio': group_data['sharpe_ratio_mean'].mean(),
                    'avg_max_drawdown': group_data['max_drawdown_mean'].mean(),
                    'avg_win_rate': group_data['win_rate_mean'].mean()
                }
        
        # 保存到JSON
        export_data = {
            'summary': summary_stats,
            'group_statistics': group_stats,
            'detailed_statistics': stats_df.to_dict('index')
        }
        
        with open(self.output_dir / 'experiment_summary.json', 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存到CSV (主要指標)
        main_metrics = ['annual_return_mean', 'annual_return_std', 'volatility_mean', 'volatility_std', 
                       'sharpe_ratio_mean', 'sharpe_ratio_std', 'max_drawdown_mean', 'max_drawdown_std',
                       'win_rate_mean', 'win_rate_std', 'calmar_ratio_mean', 'calmar_ratio_std']
        
        main_stats_df = stats_df[main_metrics].copy()
        main_stats_df.to_csv(self.output_dir / 'main_metrics_summary.csv')
        
        print(f"✅ 摘要統計已保存到: {self.output_dir}")
        return export_data
    
    def export_performance_rankings(self, stats_df):
        """導出性能排名"""
        print("🏆 導出性能排名...")
        
        # 定義排名指標
        ranking_metrics = {
            'annual_return': 'annual_return_mean',
            'sharpe_ratio': 'sharpe_ratio_mean',
            'calmar_ratio': 'calmar_ratio_mean',
            'win_rate': 'win_rate_mean',
            'max_drawdown': 'max_drawdown_mean'  # 注意：最大回撤越小越好
        }
        
        rankings = {}
        
        for metric_name, column_name in ranking_metrics.items():
            if column_name in stats_df.columns:
                if metric_name == 'max_drawdown':
                    # 最大回撤：數值越小排名越高
                    ranked = stats_df.sort_values(column_name, ascending=True)
                else:
                    # 其他指標：數值越大排名越高
                    ranked = stats_df.sort_values(column_name, ascending=False)
                
                rankings[metric_name] = {
                    'rankings': ranked[column_name].to_dict(),
                    'top_3': ranked.head(3)[column_name].to_dict()
                }
        
        # 綜合排名（基於Sharpe ratio）
        if 'sharpe_ratio_mean' in stats_df.columns:
            overall_ranking = stats_df.sort_values('sharpe_ratio_mean', ascending=False)
            rankings['overall'] = {
                'rankings': overall_ranking['sharpe_ratio_mean'].to_dict(),
                'top_5': overall_ranking.head(5)[['sharpe_ratio_mean', 'annual_return_mean', 'volatility_mean']].to_dict('index')
            }
        
        # 保存排名數據
        with open(self.output_dir / 'performance_rankings.json', 'w', encoding='utf-8') as f:
            json.dump(rankings, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存排名CSV
        ranking_df = pd.DataFrame()
        for metric_name, data in rankings.items():
            if 'rankings' in data:
                ranking_df[f'{metric_name}_rank'] = range(1, len(data['rankings']) + 1)
                ranking_df[f'{metric_name}_value'] = list(data['rankings'].values())
        
        ranking_df.index = list(rankings['overall']['rankings'].keys()) if 'overall' in rankings else stats_df.index
        ranking_df.to_csv(self.output_dir / 'performance_rankings.csv')
        
        print(f"✅ 性能排名已保存到: {self.output_dir}")
        return rankings
    
    def export_time_series_data(self, time_series_data):
        """導出時間序列數據"""
        print("📈 導出時間序列數據...")
        
        # 為每個模型創建CSV文件
        for model_name, data in time_series_data.items():
            model_df = pd.DataFrame({
                'day': data['days'],
                'cumulative_returns_mean': data['cumulative_returns_mean'],
                'cumulative_returns_std': data['cumulative_returns_std'],
                'cumulative_returns_min': data.get('cumulative_returns_min', []),
                'cumulative_returns_max': data.get('cumulative_returns_max', []),
                'cumulative_returns_median': data.get('cumulative_returns_median', [])
            })
            
            # 添加日期列
            model_df['date'] = pd.date_range('2024-01-01', periods=len(model_df), freq='B')
            
            # 保存到CSV
            model_df.to_csv(self.output_dir / f'{model_name.replace(" ", "_")}_time_series.csv', index=False)
        
        # 創建所有模型的合併時間序列
        all_models_df = pd.DataFrame()
        for model_name, data in time_series_data.items():
            all_models_df[f'{model_name}_cumulative_mean'] = data['cumulative_returns_mean']
            all_models_df[f'{model_name}_cumulative_std'] = data['cumulative_returns_std']
            all_models_df[f'{model_name}_cumulative_min'] = data.get('cumulative_returns_min', [])
            all_models_df[f'{model_name}_cumulative_max'] = data.get('cumulative_returns_max', [])
            all_models_df[f'{model_name}_cumulative_median'] = data.get('cumulative_returns_median', [])
        
        all_models_df['date'] = pd.date_range('2024-01-01', periods=len(all_models_df), freq='B')
        all_models_df.to_csv(self.output_dir / 'all_models_time_series.csv', index=False)
        
        print(f"✅ 時間序列數據已保存到: {self.output_dir}")
    
    def export_raw_results(self, raw_results):
        """導出原始結果數據"""
        print("📋 導出原始結果數據...")
        
        # 按模型分組保存
        for model in raw_results['Model'].unique():
            model_data = raw_results[raw_results['Model'] == model]
            model_data.to_csv(self.output_dir / f'{model.replace(" ", "_")}_raw_results.csv', index=False)
        
        # 保存所有原始結果
        raw_results.to_csv(self.output_dir / 'all_raw_results.csv', index=False)
        
        print(f"✅ 原始結果數據已保存到: {self.output_dir}")
    
    def export_excel_summary(self, stats_df, rankings):
        """導出Excel摘要報告"""
        print("📊 導出Excel摘要報告...")
        
        try:
            with pd.ExcelWriter(self.output_dir / 'experiment_results_summary.xlsx', engine='openpyxl') as writer:
                # 主要統計數據
                stats_df.to_excel(writer, sheet_name='Main_Statistics')
                
                # 性能排名
                if 'overall' in rankings:
                    overall_df = pd.DataFrame(rankings['overall']['rankings'], index=[0]).T
                    overall_df.columns = ['Sharpe_Ratio']
                    overall_df.to_excel(writer, sheet_name='Overall_Rankings')
                
                # 按指標排名
                for metric_name, data in rankings.items():
                    if metric_name != 'overall' and 'rankings' in data:
                        metric_df = pd.DataFrame(data['rankings'], index=[0]).T
                        metric_df.columns = [f'{metric_name}_value']
                        metric_df.to_excel(writer, sheet_name=f'{metric_name}_rankings')
                
                # 模型類型分組統計
                model_groups = {
                    'Classic ML': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                    'Quantum ML': ['VQE Classifier', 'QNN', 'QSVM'],
                    'Hybrid Quantum': ['QASA Hybrid', 'LSTM_QNN', 'QuantumRWKV'],
                    'Transformer': ['Transformer']
                }
                
                for group_name, models in model_groups.items():
                    available_models = [m for m in models if m in stats_df.index]
                    if available_models:
                        group_df = stats_df.loc[available_models]
                        group_df.to_excel(writer, sheet_name=f'{group_name}_Models')
            
            print(f"✅ Excel摘要報告已保存到: {self.output_dir}")
            
        except ImportError:
            print("⚠️  openpyxl未安裝，跳過Excel導出")
    
    def generate_readme(self, stats_df, rankings):
        """生成README文件"""
        print("📝 生成README文件...")
        
        readme_content = f"""# 實驗結果數據導出

## 概述
本目錄包含所有機器學習模型在金融數據上的實驗結果。

## 數據統計
- **總模型數**: {len(stats_df)}
- **總運行次數**: {stats_df['n_runs'].sum()}
- **實驗期間**: 2024年1月1日 - 2024年12月31日 (252個交易日)

## 文件說明

### 主要文件
- `experiment_summary.json`: 完整的實驗摘要統計
- `main_metrics_summary.csv`: 主要性能指標摘要
- `performance_rankings.json`: 各項指標的性能排名
- `performance_rankings.csv`: 性能排名表格
- `experiment_results_summary.xlsx`: Excel格式的完整摘要報告

### 時間序列數據
- `all_models_time_series.csv`: 所有模型的時間序列數據
- `[Model_Name]_time_series.csv`: 個別模型的時間序列數據

### 原始結果
- `all_raw_results.csv`: 所有原始實驗結果
- `[Model_Name]_raw_results.csv`: 個別模型的原始結果

## 性能排名 (基於Sharpe Ratio)

"""
        
        if 'overall' in rankings:
            for i, (model, sharpe) in enumerate(rankings['overall']['rankings'].items(), 1):
                readme_content += f"{i}. **{model}**: {sharpe:.3f}\n"
        
        readme_content += f"""
## 模型類型分組

### Classic ML
- Random Forest
- Gradient Boosting  
- Logistic Regression

### Quantum ML
- VQE Classifier
- QNN
- QSVM

### Hybrid Quantum
- QASA Hybrid
- LSTM_QNN
- QuantumRWKV

### Transformer
- Transformer

## 主要指標說明

- **Annual Return**: 年化收益率
- **Volatility**: 波動率
- **Sharpe Ratio**: 夏普比率 (風險調整後收益)
- **Max Drawdown**: 最大回撤
- **Calmar Ratio**: 卡爾瑪比率 (年化收益/最大回撤)
- **Win Rate**: 勝率
- **Profit Factor**: 盈利因子

## 數據格式

所有CSV文件使用UTF-8編碼，JSON文件使用UTF-8編碼並格式化輸出。

生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(self.output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ README文件已生成: {self.output_dir}/README.md")
    
    def export_all_results(self):
        """導出所有實驗結果"""
        print("🚀 開始導出所有實驗結果...")
        
        # 載入數據
        stats_df, time_series_data, raw_results = self.load_data()
        
        if stats_df is None:
            print("❌ 無法載入數據，導出終止")
            return
        
        print(f"📊 載入 {len(stats_df)} 個模型的數據")
        
        # 導出各種格式的數據
        summary_data = self.export_summary_statistics(stats_df)
        rankings = self.export_performance_rankings(stats_df)
        
        if time_series_data:
            self.export_time_series_data(time_series_data)
        
        if raw_results is not None:
            self.export_raw_results(raw_results)
        
        self.export_excel_summary(stats_df, rankings)
        self.generate_readme(stats_df, rankings)
        
        print(f"✅ 所有實驗結果已導出完成！")
        print(f"📁 輸出目錄: {self.output_dir}")
        print(f"📋 生成的文件:")
        for file_path in self.output_dir.glob('*'):
            print(f"   - {file_path.name}")

def main():
    """主函數"""
    exporter = ExperimentResultsExporter()
    exporter.export_all_results()

if __name__ == "__main__":
    main()
