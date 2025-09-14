#!/usr/bin/env python3
"""
公平比較分析
基於統一AMM Baseline labels的模型比較分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set English font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class FairComparisonAnalyzer:
    """公平比較分析器"""
    
    def __init__(self, output_dir="reports/fair_comparison_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results from unified training
        self.results = {
            'Random Forest': {'accuracy': 0.9948, 'type': 'Classical', 'category': 'Tree-based'},
            'Gradient Boosting': {'accuracy': 0.9948, 'type': 'Classical', 'category': 'Tree-based'},
            'Logistic Regression': {'accuracy': 0.6373, 'type': 'Classical', 'category': 'Linear'},
            'VQE Classifier': {'accuracy': 0.3731, 'type': 'Quantum', 'category': 'Pure Quantum'},
            'QNN': {'accuracy': 0.3731, 'type': 'Quantum', 'category': 'Pure Quantum'},
            'QASA Hybrid': {'accuracy': 0.6425, 'type': 'Quantum', 'category': 'Hybrid Quantum'}
        }
    
    def create_comprehensive_comparison(self):
        """創建綜合比較圖表"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._create_accuracy_comparison(ax1)
        
        # 2. Model Type Distribution (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._create_model_type_distribution(ax2)
        
        # 3. Category Performance (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._create_category_performance(ax3)
        
        # 4. Classical vs Quantum (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._create_classical_vs_quantum(ax4)
        
        # 5. Performance Ranking (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._create_performance_ranking(ax5)
        
        # 6. QASA Analysis (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._create_qasa_analysis(ax6)
        
        plt.suptitle('Fair Model Comparison Analysis\n(Unified AMM Baseline Labels)', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.savefig(self.output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_accuracy_comparison(self, ax):
        """創建準確率比較圖"""
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title('Model Accuracy Comparison\n(Unified AMM Baseline Labels)', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    def _create_model_type_distribution(self, ax):
        """創建模型類型分布圖"""
        type_counts = {}
        for model, data in self.results.items():
            model_type = data['type']
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        
        colors = ['#4ECDC4', '#FF6B6B']
        wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(), 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Model Type Distribution', fontweight='bold', fontsize=14)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
    
    def _create_category_performance(self, ax):
        """創建類別性能圖"""
        categories = {}
        for model, data in self.results.items():
            category = data['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(data['accuracy'])
        
        # Calculate average accuracy for each category
        category_means = {cat: np.mean(accs) for cat, accs in categories.items()}
        
        bars = ax.bar(category_means.keys(), category_means.values(), 
                     color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black')
        ax.set_title('Performance by Model Category', fontweight='bold', fontsize=14)
        ax.set_ylabel('Average Accuracy')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc in zip(bars, category_means.values()):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _create_classical_vs_quantum(self, ax):
        """創建經典vs量子比較圖"""
        classical_accs = [data['accuracy'] for model, data in self.results.items() 
                         if data['type'] == 'Classical']
        quantum_accs = [data['accuracy'] for model, data in self.results.items() 
                       if data['type'] == 'Quantum']
        
        # Create box plot
        data_for_box = [classical_accs, quantum_accs]
        labels = ['Classical ML', 'Quantum ML']
        colors = ['#4ECDC4', '#FF6B6B']
        
        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Classical vs Quantum ML Performance', fontweight='bold', fontsize=14)
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        classical_mean = np.mean(classical_accs)
        quantum_mean = np.mean(quantum_accs)
        ax.scatter([1], [classical_mean], color='darkblue', s=100, marker='D', label=f'Mean: {classical_mean:.3f}')
        ax.scatter([2], [quantum_mean], color='darkred', s=100, marker='D', label=f'Mean: {quantum_mean:.3f}')
        ax.legend()
    
    def _create_performance_ranking(self, ax):
        """創建性能排名圖"""
        # Sort models by accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        models = [item[0] for item in sorted_models]
        accuracies = [item[1]['accuracy'] for item in sorted_models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D']
        
        bars = ax.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Model Performance Ranking', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accuracy')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{acc:.3f}', ha='left', va='center', fontweight='bold')
            # Add ranking number
            ax.text(-0.05, bar.get_y() + bar.get_height()/2, f'#{i+1}', 
                   ha='right', va='center', fontweight='bold', fontsize=12)
    
    def _create_qasa_analysis(self, ax):
        """創建QASA分析圖"""
        # QASA performance analysis
        qasa_accuracy = self.results['QASA Hybrid']['accuracy']
        classical_avg = np.mean([data['accuracy'] for model, data in self.results.items() 
                               if data['type'] == 'Classical'])
        quantum_avg = np.mean([data['accuracy'] for model, data in self.results.items() 
                             if data['type'] == 'Quantum'])
        
        categories = ['QASA Hybrid', 'Classical ML\n(Average)', 'Quantum ML\n(Average)']
        values = [qasa_accuracy, classical_avg, quantum_avg]
        colors = ['#F18F01', '#4ECDC4', '#FF6B6B']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('QASA Hybrid Performance Analysis', fontweight='bold', fontsize=14)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add performance comparison text
        if qasa_accuracy > quantum_avg:
            ax.text(0.5, 0.8, f'QASA outperforms\nQuantum ML by\n{qasa_accuracy - quantum_avg:.3f}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                   fontweight='bold')
        else:
            ax.text(0.5, 0.8, f'QASA underperforms\nQuantum ML by\n{quantum_avg - qasa_accuracy:.3f}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                   fontweight='bold')
    
    def create_detailed_analysis_chart(self):
        """創建詳細分析圖表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model Performance Heatmap
        self._create_performance_heatmap(ax1)
        
        # 2. Accuracy Distribution
        self._create_accuracy_distribution(ax2)
        
        # 3. Model Complexity vs Performance
        self._create_complexity_analysis(ax3)
        
        # 4. Fair Comparison Summary
        self._create_fair_comparison_summary(ax4)
        
        plt.suptitle('Detailed Fair Comparison Analysis\n(Unified AMM Baseline Labels)', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_performance_heatmap(self, ax):
        """創建性能熱力圖"""
        # Create performance matrix
        models = list(self.results.keys())
        metrics = ['Accuracy', 'Type', 'Category']
        
        # Create data matrix
        matrix_data = []
        for model in models:
            row = [
                self.results[model]['accuracy'],
                1 if self.results[model]['type'] == 'Classical' else 0,
                1 if self.results[model]['category'] == 'Tree-based' else 
                (2 if self.results[model]['category'] == 'Linear' else 3)
            ]
            matrix_data.append(row)
        
        df_matrix = pd.DataFrame(matrix_data, index=models, columns=metrics)
        
        # Create heatmap
        sns.heatmap(df_matrix.T, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0.5, ax=ax, cbar_kws={'label': 'Normalized Score'})
        ax.set_title('Model Performance Matrix', fontweight='bold')
        ax.set_xlabel('Models')
        ax.set_ylabel('Metrics')
    
    def _create_accuracy_distribution(self, ax):
        """創建準確率分布圖"""
        accuracies = [data['accuracy'] for data in self.results.values()]
        
        ax.hist(accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(accuracies):.3f}')
        ax.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(accuracies):.3f}')
        
        ax.set_title('Accuracy Distribution', fontweight='bold')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_complexity_analysis(self, ax):
        """創建複雜度分析圖"""
        # Define complexity scores (arbitrary scale)
        complexity_scores = {
            'Random Forest': 3,
            'Gradient Boosting': 4,
            'Logistic Regression': 1,
            'VQE Classifier': 5,
            'QNN': 5,
            'QSVM': 5,
            'QASA Hybrid': 6,
            'QuantumRWKV': 9
        }
        
        models = list(self.results.keys())
        complexities = [complexity_scores[model] for model in models]
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        scatter = ax.scatter(complexities, accuracies, s=200, alpha=0.7, 
                           c=accuracies, cmap='viridis', edgecolors='black')
        
        ax.set_title('Model Complexity vs Performance', fontweight='bold')
        ax.set_xlabel('Complexity Score')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(model, (complexities[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Accuracy')
    
    def _create_fair_comparison_summary(self, ax):
        """創建公平比較摘要"""
        # Calculate statistics
        classical_accs = [data['accuracy'] for model, data in self.results.items() 
                         if data['type'] == 'Classical']
        quantum_accs = [data['accuracy'] for model, data in self.results.items() 
                       if data['type'] == 'Quantum']
        
        stats_text = f"""
        FAIR COMPARISON SUMMARY
        
        Classical ML Models: {len(classical_accs)}
        • Average Accuracy: {np.mean(classical_accs):.3f}
        • Best: {max(classical_accs):.3f}
        • Worst: {min(classical_accs):.3f}
        
        Quantum ML Models: {len(quantum_accs)}
        • Average Accuracy: {np.mean(quantum_accs):.3f}
        • Best: {max(quantum_accs):.3f}
        • Worst: {min(quantum_accs):.3f}
        
        QASA Hybrid Performance:
        • Accuracy: {self.results['QASA Hybrid']['accuracy']:.3f}
        • vs Classical Avg: {self.results['QASA Hybrid']['accuracy'] - np.mean(classical_accs):+.3f}
        • vs Quantum Avg: {self.results['QASA Hybrid']['accuracy'] - np.mean(quantum_accs):+.3f}
        
        Key Finding:
        {'Classical ML' if np.mean(classical_accs) > np.mean(quantum_accs) else 'Quantum ML'} 
        performs better overall
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Fair Comparison Summary', fontweight='bold')
        ax.axis('off')
    
    def generate_comprehensive_report(self):
        """生成綜合報告"""
        report_path = self.output_dir / "fair_comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Fair Model Comparison Analysis Report\n\n")
            f.write("## 🎯 Analysis Objective\n\n")
            f.write("This analysis compares all models using **unified AMM Baseline labels** for fair evaluation:\n\n")
            f.write("- **Label Standard**: `y = 1 if |price/MA_20 - 1| > 0.02 else 0`\n")
            f.write("- **Problem Type**: Binary classification\n")
            f.write("- **Evaluation Metric**: Accuracy\n")
            f.write("- **Fair Comparison**: All models solve identical problem\n\n")
            
            f.write("## 📊 Model Performance Results\n\n")
            f.write("| Rank | Model | Type | Category | Accuracy |\n")
            f.write("|------|-------|------|----------|----------|\n")
            
            # Sort by accuracy
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (model, data) in enumerate(sorted_models, 1):
                f.write(f"| {i} | {model} | {data['type']} | {data['category']} | {data['accuracy']:.4f} |\n")
            
            f.write("\n## 🔍 Key Findings\n\n")
            
            # Calculate statistics
            classical_accs = [data['accuracy'] for model, data in self.results.items() 
                             if data['type'] == 'Classical']
            quantum_accs = [data['accuracy'] for model, data in self.results.items() 
                           if data['type'] == 'Quantum']
            
            f.write(f"### 1. Overall Performance\n")
            f.write(f"- **Best Model**: {sorted_models[0][0]} ({sorted_models[0][1]['accuracy']:.4f})\n")
            f.write(f"- **Worst Model**: {sorted_models[-1][0]} ({sorted_models[-1][1]['accuracy']:.4f})\n")
            f.write(f"- **Performance Range**: {sorted_models[0][1]['accuracy'] - sorted_models[-1][1]['accuracy']:.4f}\n\n")
            
            f.write(f"### 2. Classical vs Quantum ML\n")
            f.write(f"- **Classical ML Average**: {np.mean(classical_accs):.4f}\n")
            f.write(f"- **Quantum ML Average**: {np.mean(quantum_accs):.4f}\n")
            f.write(f"- **Performance Difference**: {np.mean(classical_accs) - np.mean(quantum_accs):+.4f}\n")
            f.write(f"- **Winner**: {'Classical ML' if np.mean(classical_accs) > np.mean(quantum_accs) else 'Quantum ML'}\n\n")
            
            f.write(f"### 3. QASA Hybrid Analysis\n")
            qasa_acc = self.results['QASA Hybrid']['accuracy']
            f.write(f"- **QASA Accuracy**: {qasa_acc:.4f}\n")
            f.write(f"- **vs Classical Average**: {qasa_acc - np.mean(classical_accs):+.4f}\n")
            f.write(f"- **vs Quantum Average**: {qasa_acc - np.mean(quantum_accs):+.4f}\n")
            f.write(f"- **Ranking**: #{[i for i, (m, _) in enumerate(sorted_models, 1) if m == 'QASA Hybrid'][0]}\n\n")
            
            f.write("## 📈 Generated Charts\n\n")
            f.write("1. **comprehensive_comparison.png** - Complete comparison analysis\n")
            f.write("2. **detailed_analysis.png** - Detailed performance analysis\n")
            f.write("3. **fair_comparison_report.md** - This comprehensive report\n\n")
            
            f.write("## ✅ Conclusions\n\n")
            f.write("### Fair Comparison Achieved\n")
            f.write("By using unified AMM Baseline labels, we achieved a fair comparison between all models.\n")
            f.write("All models now solve the same binary classification problem with identical evaluation criteria.\n\n")
            
            f.write("### Key Insights\n")
            f.write("1. **Classical ML dominates**: Tree-based models (Random Forest, Gradient Boosting) achieve highest accuracy\n")
            f.write("2. **Quantum ML struggles**: Both pure quantum and hybrid models show lower performance\n")
            f.write("3. **QASA shows promise**: Hybrid approach outperforms pure quantum models\n")
            f.write("4. **Model complexity matters**: More complex models don't necessarily perform better\n\n")
            
            f.write("### Recommendations\n")
            f.write("1. **For production**: Use Random Forest or Gradient Boosting for highest accuracy\n")
            f.write("2. **For research**: Continue developing QASA hybrid approaches\n")
            f.write("3. **For simplicity**: Consider Logistic Regression for baseline performance\n")
            f.write("4. **For quantum**: Focus on improving quantum feature engineering\n")
    
    def run_analysis(self):
        """運行完整分析"""
        logger.info("🔍 Starting Fair Comparison Analysis...")
        
        # Create comprehensive comparison
        logger.info("📊 Creating comprehensive comparison charts...")
        self.create_comprehensive_comparison()
        
        # Create detailed analysis
        logger.info("📈 Creating detailed analysis charts...")
        self.create_detailed_analysis_chart()
        
        # Generate report
        logger.info("📝 Generating comprehensive report...")
        self.generate_comprehensive_report()
        
        logger.info(f"✅ Fair comparison analysis completed! Results saved to: {self.output_dir}")

def main():
    """主函數"""
    analyzer = FairComparisonAnalyzer()
    analyzer.run_analysis()
    
    print("\n📊 Fair Comparison Analysis Summary:")
    print("=" * 50)
    
    # Show top 3 models
    sorted_models = sorted(analyzer.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    print("\n🏆 Top 3 Models:")
    for i, (model, data) in enumerate(sorted_models[:3], 1):
        print(f"  {i}. {model}: {data['accuracy']:.4f} ({data['type']})")
    
    # Show QASA performance
    qasa_acc = analyzer.results['QASA Hybrid']['accuracy']
    classical_avg = np.mean([data['accuracy'] for model, data in analyzer.results.items() 
                           if data['type'] == 'Classical'])
    quantum_avg = np.mean([data['accuracy'] for model, data in analyzer.results.items() 
                         if data['type'] == 'Quantum'])
    
    print(f"\n🔬 QASA Hybrid Analysis:")
    print(f"  Accuracy: {qasa_acc:.4f}")
    print(f"  vs Classical Avg: {qasa_acc - classical_avg:+.4f}")
    print(f"  vs Quantum Avg: {qasa_acc - quantum_avg:+.4f}")

if __name__ == "__main__":
    main()
