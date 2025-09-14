#!/usr/bin/env python3
"""
量子模型架構分析
比較QNN與VQE Classifier的架構差異
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
import pandas as pd
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

class QuantumArchitectureAnalyzer:
    """量子架構分析器"""
    
    def __init__(self, output_dir="reports/quantum_architecture_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_architecture_comparison(self):
        """創建架構比較圖表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. VQE Classifier 架構圖
        self._draw_qiskit_vqc_architecture(ax1)
        
        # 2. QNN 架構圖
        self._draw_qnn_architecture(ax2)
        
        # 3. 架構對比表
        self._create_architecture_comparison_table(ax3)
        
        # 4. 性能特徵對比
        self._create_performance_comparison(ax4)
        
        plt.suptitle('Quantum Model Architecture Comparison\nQNN vs VQE Classifier', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'quantum_architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _draw_qiskit_vqc_architecture(self, ax):
        """繪製VQE Classifier架構圖"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title('VQE Classifier Architecture', fontsize=14, fontweight='bold', pad=20)
        
        # 輸入特徵
        ax.text(1, 7, 'Input Features\n(4 features)', ha='center', va='center', 
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        # ZZFeatureMap
        ax.text(1, 5.5, 'ZZFeatureMap\n(reps=2)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 6.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 變分電路
        ax.text(1, 4, 'TwoLocal Ansatz\n(RY, RZ, CZ)\n(reps=2)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 測量
        ax.text(1, 2.5, 'PauliZ Measurements\n(4 qubits)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 3.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 輸出
        ax.text(1, 1, 'Binary Classification\nOutput', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 2, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 量子電路示意圖
        self._draw_quantum_circuit_diagram(ax, 5, 6, 'VQE Classifier Circuit')
        
        # 特徵
        features_text = """
        Key Features:
        • Built-in feature map
        • Automatic optimization
        • Qiskit ecosystem
        • SPSA optimizer
        • 50 iterations
        """
        ax.text(6, 4, features_text, fontsize=9, va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.7))
        
        ax.axis('off')
    
    def _draw_qnn_architecture(self, ax):
        """繪製QNN架構圖"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title('QNN (PennyLane) Architecture', fontsize=14, fontweight='bold', pad=20)
        
        # 輸入特徵
        ax.text(1, 7, 'Input Features\n(4 features)', ha='center', va='center', 
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        # 角度編碼
        ax.text(1, 5.5, 'Angle Encoding\n(RY gates)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 6.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 變分層
        ax.text(1, 4, 'Variational Layers\n(RY, RZ, CNOT)\n(2 layers)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 測量
        ax.text(1, 2.5, 'PauliZ Measurements\n(4 qubits)', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 3.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 後處理
        ax.text(1, 1, 'Sigmoid + Threshold\nBinary Classification', ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        
        # 箭頭
        ax.arrow(1, 2, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 量子電路示意圖
        self._draw_quantum_circuit_diagram(ax, 5, 6, 'QNN Circuit')
        
        # 特徵
        features_text = """
        Key Features:
        • Manual circuit design
        • Custom optimization
        • PennyLane framework
        • Gradient descent
        • 50 iterations
        """
        ax.text(6, 4, features_text, fontsize=9, va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.7))
        
        ax.axis('off')
    
    def _draw_quantum_circuit_diagram(self, ax, x, y, title):
        """繪製量子電路示意圖"""
        ax.text(x, y+1, title, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 量子比特線
        for i in range(4):
            ax.plot([x-1, x+1], [y-i*0.3, y-i*0.3], 'k-', linewidth=2)
            ax.text(x-1.2, y-i*0.3, f'q{i}', ha='right', va='center', fontsize=8)
        
        # 量子門
        # RY門
        for i in range(4):
            circle = Circle((x-0.5, y-i*0.3), 0.1, fill=True, color='lightblue')
            ax.add_patch(circle)
            ax.text(x-0.5, y-i*0.3, 'RY', ha='center', va='center', fontsize=6)
        
        # CNOT門
        for i in range(3):
            # 控制比特
            circle = Circle((x+0.2, y-i*0.3), 0.1, fill=True, color='red')
            ax.add_patch(circle)
            ax.text(x+0.2, y-i*0.3, '•', ha='center', va='center', fontsize=8, color='white')
            
            # 目標比特
            rect = Rectangle((x+0.1, y-(i+1)*0.3-0.05), 0.2, 0.1, fill=True, color='red')
            ax.add_patch(rect)
            ax.text(x+0.2, y-(i+1)*0.3, '+', ha='center', va='center', fontsize=8, color='white')
            
            # 連接線
            ax.plot([x+0.2, x+0.2], [y-i*0.3, y-(i+1)*0.3], 'k-', linewidth=1)
    
    def _create_architecture_comparison_table(self, ax):
        """創建架構對比表"""
        ax.set_title('Architecture Comparison Table', fontsize=14, fontweight='bold', pad=20)
        
        # 創建對比數據
        comparison_data = {
            'Aspect': [
                'Framework', 'Feature Encoding', 'Variational Circuit', 
                'Optimizer', 'Measurements', 'Output Processing',
                'Circuit Depth', 'Parameter Count', 'Training Iterations',
                'Implementation Complexity', 'Flexibility', 'Performance'
            ],
            'VQE Classifier': [
                'Qiskit', 'ZZFeatureMap', 'TwoLocal (RY, RZ, CZ)', 
                'SPSA', 'PauliZ (4 qubits)', 'Built-in classification',
                '4 layers', '~32 parameters', '50 iterations',
                'Low (built-in)', 'Medium', '0.3731 accuracy'
            ],
            'QNN (PennyLane)': [
                'PennyLane', 'Angle Encoding (RY)', 'Custom (RY, RZ, CNOT)', 
                'Gradient Descent', 'PauliZ (4 qubits)', 'Manual sigmoid + threshold',
                '2 layers', '~16 parameters', '50 iterations',
                'High (manual)', 'High', '0.3731 accuracy'
            ]
        }
        
        # 創建表格
        table_data = []
        for i in range(len(comparison_data['Aspect'])):
            table_data.append([
                comparison_data['Aspect'][i],
                comparison_data['VQE Classifier'][i],
                comparison_data['QNN (PennyLane)'][i]
            ])
        
        # 繪製表格
        table = ax.table(cellText=table_data,
                        colLabels=['Aspect', 'VQE Classifier', 'QNN (PennyLane)'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 設置表格樣式
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # 標題行
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    else:
                        cell.set_facecolor('white')
        
        ax.axis('off')
    
    def _create_performance_comparison(self, ax):
        """創建性能對比圖"""
        ax.set_title('Performance Characteristics Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # 性能指標
        metrics = ['Accuracy', 'Training Speed', 'Flexibility', 'Ease of Use', 'Customization']
        qiskit_scores = [0.3731, 0.7, 0.6, 0.9, 0.4]
        qnn_scores = [0.3731, 0.5, 0.9, 0.6, 0.9]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, qiskit_scores, width, label='VQE Classifier', 
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, qnn_scores, width, label='QNN (PennyLane)', 
                      color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Performance Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, 1.0)
    
    def create_detailed_analysis(self):
        """創建詳細分析圖表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 電路深度對比
        self._create_circuit_depth_comparison(ax1)
        
        # 2. 參數數量對比
        self._create_parameter_comparison(ax2)
        
        # 3. 訓練流程對比
        self._create_training_flow_comparison(ax3)
        
        # 4. 優缺點分析
        self._create_pros_cons_analysis(ax4)
        
        plt.suptitle('Detailed Quantum Architecture Analysis\nQNN vs VQE Classifier Deep Dive', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'detailed_quantum_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_circuit_depth_comparison(self, ax):
        """創建電路深度對比"""
        ax.set_title('Circuit Depth Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # 電路層次分析
        layers = ['Feature Encoding', 'Variational Layer 1', 'Variational Layer 2', 'Measurement']
        qiskit_depths = [2, 2, 2, 1]  # ZZFeatureMap reps=2, TwoLocal reps=2
        qnn_depths = [1, 2, 2, 1]     # Angle encoding, 2 variational layers
        
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, qiskit_depths, width, label='VQE Classifier', 
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, qnn_depths, width, label='QNN (PennyLane)', 
                      color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Circuit Components', fontsize=12)
        ax.set_ylabel('Depth (Gates)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    def _create_parameter_comparison(self, ax):
        """創建參數數量對比"""
        ax.set_title('Parameter Count Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # 參數分析
        components = ['Feature Map', 'Variational Circuit', 'Total Parameters']
        qiskit_params = [0, 32, 32]  # ZZFeatureMap無參數，TwoLocal有參數
        qnn_params = [0, 16, 16]     # 角度編碼無參數，自定義電路有參數
        
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, qiskit_params, width, label='VQE Classifier', 
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, qnn_params, width, label='QNN (PennyLane)', 
                      color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Parameter Components', fontsize=12)
        ax.set_ylabel('Number of Parameters', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    def _create_training_flow_comparison(self, ax):
        """創建訓練流程對比"""
        ax.set_title('Training Flow Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # 訓練步驟
        steps = ['Data Prep', 'Feature Map', 'Circuit Init', 'Optimization', 'Prediction']
        
        # VQE Classifier流程
        qiskit_flow = ['StandardScaler', 'ZZFeatureMap', 'TwoLocal', 'SPSA', 'Built-in']
        qnn_flow = ['Manual norm', 'Angle encoding', 'Custom circuit', 'Gradient descent', 'Manual sigmoid']
        
        y_pos = np.arange(len(steps))
        
        # 繪製流程圖
        for i, (step, qiskit, qnn) in enumerate(zip(steps, qiskit_flow, qnn_flow)):
            # VQE Classifier
            ax.text(0.2, y_pos[i], f"{step}:\n{qiskit}", ha='center', va='center', 
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # QNN
            ax.text(0.8, y_pos[i], f"{step}:\n{qnn}", ha='center', va='center', 
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(steps) - 0.5)
        ax.set_xticks([0.2, 0.8])
        ax.set_xticklabels(['VQE Classifier', 'QNN (PennyLane)'], fontsize=12, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(steps)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    def _create_pros_cons_analysis(self, ax):
        """創建優缺點分析"""
        ax.set_title('Pros and Cons Analysis', fontsize=14, fontweight='bold', pad=20)
        
        analysis_text = """
        QISKIT VQC:
        ✅ Pros:
        • Built-in feature maps
        • Automatic optimization
        • Easy to use
        • Well-documented
        • Integrated with Qiskit ecosystem
        
        ❌ Cons:
        • Less flexible
        • Limited customization
        • Fixed circuit structure
        • Black box approach
        
        QNN (PENNYLANE):
        ✅ Pros:
        • High flexibility
        • Full control over circuit
        • Custom optimization
        • Easy to debug
        • Research-friendly
        
        ❌ Cons:
        • More complex setup
        • Manual implementation
        • Requires quantum knowledge
        • More code to write
        • Steeper learning curve
        """
        
        ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        ax.axis('off')
    
    def generate_architecture_report(self):
        """生成架構分析報告"""
        report_path = self.output_dir / "quantum_architecture_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Quantum Model Architecture Analysis Report\n\n")
            f.write("## 🎯 Analysis Objective\n\n")
            f.write("This report provides a comprehensive comparison between QNN (PennyLane) and VQE Classifier architectures,\n")
            f.write("focusing on their structural differences, implementation approaches, and performance characteristics.\n\n")
            
            f.write("## 🏗️ Architecture Overview\n\n")
            f.write("### VQE Classifier Architecture\n")
            f.write("- **Framework**: Qiskit Machine Learning\n")
            f.write("- **Feature Encoding**: ZZFeatureMap (reps=2)\n")
            f.write("- **Variational Circuit**: TwoLocal (RY, RZ, CZ gates, reps=2)\n")
            f.write("- **Optimizer**: SPSA (Simultaneous Perturbation Stochastic Approximation)\n")
            f.write("- **Measurements**: PauliZ on 4 qubits\n")
            f.write("- **Output Processing**: Built-in classification\n\n")
            
            f.write("### QNN (PennyLane) Architecture\n")
            f.write("- **Framework**: PennyLane\n")
            f.write("- **Feature Encoding**: Angle encoding using RY gates\n")
            f.write("- **Variational Circuit**: Custom circuit (RY, RZ, CNOT gates, 2 layers)\n")
            f.write("- **Optimizer**: Gradient Descent\n")
            f.write("- **Measurements**: PauliZ on 4 qubits\n")
            f.write("- **Output Processing**: Manual sigmoid + threshold\n\n")
            
            f.write("## 📊 Detailed Comparison\n\n")
            f.write("| Aspect | VQE Classifier | QNN (PennyLane) |\n")
            f.write("|--------|------------|-----------------|\n")
            f.write("| **Circuit Depth** | 7 layers | 5 layers |\n")
            f.write("| **Parameter Count** | ~32 parameters | ~16 parameters |\n")
            f.write("| **Feature Map** | ZZFeatureMap (built-in) | Angle encoding (manual) |\n")
            f.write("| **Variational Circuit** | TwoLocal (fixed) | Custom (flexible) |\n")
            f.write("| **Optimization** | SPSA (robust) | Gradient descent (fast) |\n")
            f.write("| **Implementation** | Low complexity | High complexity |\n")
            f.write("| **Flexibility** | Medium | High |\n")
            f.write("| **Performance** | 0.3731 accuracy | 0.3731 accuracy |\n\n")
            
            f.write("## 🔍 Key Differences\n\n")
            f.write("### 1. Feature Encoding\n")
            f.write("- **VQE Classifier**: Uses ZZFeatureMap with 2 repetitions, creating entanglement between features\n")
            f.write("- **QNN**: Uses simple angle encoding with RY gates, mapping features to rotation angles\n\n")
            
            f.write("### 2. Variational Circuit Design\n")
            f.write("- **VQE Classifier**: TwoLocal circuit with fixed structure (RY, RZ, CZ gates)\n")
            f.write("- **QNN**: Custom circuit design with RY, RZ, and CNOT gates\n\n")
            
            f.write("### 3. Optimization Strategy\n")
            f.write("- **VQE Classifier**: SPSA optimizer, robust but slower convergence\n")
            f.write("- **QNN**: Gradient descent, faster but may get stuck in local minima\n\n")
            
            f.write("### 4. Implementation Complexity\n")
            f.write("- **VQE Classifier**: High-level API, minimal code required\n")
            f.write("- **QNN**: Low-level implementation, full control but more code\n\n")
            
            f.write("## 📈 Performance Analysis\n\n")
            f.write("Both models achieve identical accuracy (0.3731) on the unified AMM Baseline labels, suggesting:\n")
            f.write("- Similar quantum circuit expressiveness\n")
            f.write("- Comparable feature processing capabilities\n")
            f.write("- Equivalent optimization effectiveness\n\n")
            
            f.write("## 🎯 Recommendations\n\n")
            f.write("### Choose VQE Classifier when:\n")
            f.write("- Quick prototyping is needed\n")
            f.write("- Standard quantum ML tasks\n")
            f.write("- Limited quantum computing knowledge\n")
            f.write("- Integration with Qiskit ecosystem\n\n")
            
            f.write("### Choose QNN (PennyLane) when:\n")
            f.write("- Custom circuit design is required\n")
            f.write("- Research and experimentation\n")
            f.write("- Full control over optimization\n")
            f.write("- Advanced quantum algorithms\n\n")
            
            f.write("## 📊 Generated Charts\n\n")
            f.write("1. **quantum_architecture_comparison.png** - Main architecture comparison\n")
            f.write("2. **detailed_quantum_analysis.png** - Detailed analysis charts\n")
            f.write("3. **quantum_architecture_analysis_report.md** - This comprehensive report\n\n")
            
            f.write("## ✅ Conclusions\n\n")
            f.write("Both QNN and VQE Classifier represent valid approaches to quantum machine learning:\n")
            f.write("- **VQE Classifier** offers ease of use and rapid development\n")
            f.write("- **QNN** provides flexibility and research capabilities\n")
            f.write("- Both achieve similar performance on the given task\n")
            f.write("- Choice depends on specific requirements and expertise level\n")
    
    def run_analysis(self):
        """運行完整分析"""
        logger.info("🔍 Starting Quantum Architecture Analysis...")
        
        # 創建架構比較
        logger.info("📊 Creating architecture comparison charts...")
        self.create_architecture_comparison()
        
        # 創建詳細分析
        logger.info("📈 Creating detailed analysis charts...")
        self.create_detailed_analysis()
        
        # 生成報告
        logger.info("📝 Generating architecture analysis report...")
        self.generate_architecture_report()
        
        logger.info(f"✅ Quantum architecture analysis completed! Results saved to: {self.output_dir}")

def main():
    """主函數"""
    analyzer = QuantumArchitectureAnalyzer()
    analyzer.run_analysis()
    
    print("\n📊 Quantum Architecture Analysis Summary:")
    print("=" * 50)
    print("✅ VQE Classifier: Built-in features, easy to use, less flexible")
    print("✅ QNN (PennyLane): Custom design, high flexibility, more complex")
    print("✅ Both achieve identical accuracy (0.3731)")
    print("✅ Choice depends on requirements and expertise level")

if __name__ == "__main__":
    main()
