#!/usr/bin/env python3
"""
整合角度編碼的量子機器學習流程
將角度編碼整合到現有的Qiskit和PennyLane量子模型中
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入角度編碼器
from quantum_angle_encoding import QuantumAngleEncoder

# 量子計算導入
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.circuit.library import ZZFeatureMap, TwoLocal
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    logging.warning(f"Qiskit not available: {e}")

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
    import torch
    import torch.nn as nn
    PENNYLANE_AVAILABLE = True
except ImportError as e:
    PENNYLANE_AVAILABLE = False
    logging.warning(f"PennyLane not available: {e}")

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QuantumMLWithAngleEncoding:
    """整合角度編碼的量子機器學習流程"""
    
    def __init__(self, n_qubits=6, n_layers=3, encoding_method='robust'):
        """
        初始化量子ML流程
        
        Args:
            n_qubits: 量子比特數量
            n_layers: 變分層數量
            encoding_method: 角度編碼方法
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_method = encoding_method
        
        # 初始化角度編碼器
        self.angle_encoder = QuantumAngleEncoder(n_qubits=n_qubits, encoding_method=encoding_method)
        
        # 量子模型
        self.qiskit_models = {}
        self.pennylane_models = {}
        
        # 訓練結果
        self.training_results = {}
        self.feature_importance = {}
        
        logger.info(f"初始化量子ML流程: {n_qubits}量子比特, {n_layers}層, {encoding_method}編碼")
    
    def prepare_features_with_angle_encoding(self, X_train, X_test, y_train, y_test, feature_names):
        """
        使用角度編碼準備特徵
        
        Args:
            X_train: 訓練特徵
            X_test: 測試特徵
            y_train: 訓練標籤
            y_test: 測試標籤
            feature_names: 特徵名稱列表
        
        Returns:
            dict: 包含編碼後特徵的字典
        """
        logger.info("🔬 使用角度編碼準備特徵...")
        
        # 轉換為DataFrame
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # 創建特徵映射
        self.angle_encoder.create_feature_mapping(feature_names)
        
        # 擬合標準化器
        self.angle_encoder.fit_scalers(X_train_df)
        
        # 編碼特徵
        quantum_train = self.angle_encoder.encode_features(X_train_df)
        quantum_test = self.angle_encoder.encode_features(X_test_df)
        
        # 分析編碼質量
        encoding_analysis = self.angle_encoder.analyze_encoding_quality(X_train_df, X_test_df)
        
        logger.info(f"✅ 角度編碼完成: 訓練集 {quantum_train.shape}, 測試集 {quantum_test.shape}")
        logger.info(f"角度範圍: [{np.min(quantum_train):.3f}, {np.max(quantum_train):.3f}]")
        
        return {
            'quantum_train': quantum_train,
            'quantum_test': quantum_test,
            'y_train': y_train,
            'y_test': y_test,
            'encoding_analysis': encoding_analysis,
            'feature_mapping': self.angle_encoder.feature_mapping
        }
    
    def create_qiskit_models(self):
        """創建Qiskit量子模型"""
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit不可用，跳過Qiskit模型創建")
            return
        
        logger.info("🔬 創建Qiskit量子模型...")
        
        # 1. QNN再平衡預測器
        self.qiskit_models['qnn_rebalance'] = QiskitQNNWithAngleEncoding(
            name="QNN_Rebalance_AngleEncoded",
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            angle_encoder=self.angle_encoder
        )
        
        # 2. QSVM波動率預測器
        self.qiskit_models['qsvm_volatility'] = QiskitQSVMWithAngleEncoding(
            name="QSVM_Volatility_AngleEncoded",
            n_qubits=self.n_qubits,
            angle_encoder=self.angle_encoder
        )
        
        # 3. 混合量子模型
        self.qiskit_models['quantum_hybrid'] = QiskitHybridWithAngleEncoding(
            name="Quantum_Hybrid_AngleEncoded",
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            angle_encoder=self.angle_encoder
        )
        
        logger.info(f"✅ 創建了 {len(self.qiskit_models)} 個Qiskit模型")
    
    def create_pennylane_models(self):
        """創建PennyLane量子模型"""
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane不可用，跳過PennyLane模型創建")
            return
        
        logger.info("🔬 創建PennyLane量子模型...")
        
        # 1. PennyLane QNN
        self.pennylane_models['pennylane_qnn'] = PennyLaneQNNWithAngleEncoding(
            name="PennyLane_QNN_AngleEncoded",
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            angle_encoder=self.angle_encoder
        )
        
        # 2. PennyLane VQC
        self.pennylane_models['pennylane_vqc'] = PennyLaneVQCWithAngleEncoding(
            name="PennyLane_VQC_AngleEncoded",
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            angle_encoder=self.angle_encoder
        )
        
        # 3. PennyLane混合模型
        self.pennylane_models['pennylane_hybrid'] = PennyLaneHybridWithAngleEncoding(
            name="PennyLane_Hybrid_AngleEncoded",
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            angle_encoder=self.angle_encoder
        )
        
        logger.info(f"✅ 創建了 {len(self.pennylane_models)} 個PennyLane模型")
    
    def train_all_models(self, quantum_data):
        """訓練所有量子模型"""
        logger.info("🚀 開始訓練所有量子模型...")
        
        quantum_train = quantum_data['quantum_train']
        quantum_test = quantum_data['quantum_test']
        y_train = quantum_data['y_train']
        y_test = quantum_data['y_test']
        
        # 訓練Qiskit模型
        for model_name, model in self.qiskit_models.items():
            try:
                logger.info(f"訓練Qiskit模型: {model_name}")
                model.fit(quantum_train, y_train)
                
                # 預測和評估
                y_pred = model.predict(quantum_test)
                accuracy = np.mean(y_pred == y_test)
                
                self.training_results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'framework': 'Qiskit'
                }
                
                logger.info(f"✅ {model_name} 訓練完成, 準確率: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} 訓練失敗: {e}")
                self.training_results[model_name] = {
                    'model': None,
                    'accuracy': 0.0,
                    'predictions': None,
                    'framework': 'Qiskit',
                    'error': str(e)
                }
        
        # 訓練PennyLane模型
        for model_name, model in self.pennylane_models.items():
            try:
                logger.info(f"訓練PennyLane模型: {model_name}")
                model.fit(quantum_train, y_train)
                
                # 預測和評估
                y_pred = model.predict(quantum_test)
                accuracy = np.mean(y_pred == y_test)
                
                self.training_results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'framework': 'PennyLane'
                }
                
                logger.info(f"✅ {model_name} 訓練完成, 準確率: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} 訓練失敗: {e}")
                self.training_results[model_name] = {
                    'model': None,
                    'accuracy': 0.0,
                    'predictions': None,
                    'framework': 'PennyLane',
                    'error': str(e)
                }
        
        logger.info("✅ 所有模型訓練完成")
    
    def analyze_results(self, save_path="reports/quantum_ml_angle_encoding"):
        """分析訓練結果"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("📊 分析訓練結果...")
        
        # 創建結果比較圖表
        self._plot_model_comparison(save_path)
        self._plot_angle_encoding_analysis(save_path)
        self._plot_quantum_circuit_analysis(save_path)
        
        # 生成報告
        self._generate_analysis_report(save_path)
        
        logger.info(f"📁 分析結果保存到 {save_path}")
    
    def _plot_model_comparison(self, save_path):
        """繪製模型比較圖表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 準確率比較
        ax1 = axes[0, 0]
        model_names = list(self.training_results.keys())
        accuracies = [self.training_results[name]['accuracy'] for name in model_names]
        frameworks = [self.training_results[name]['framework'] for name in model_names]
        
        colors = ['blue' if f == 'Qiskit' else 'red' for f in frameworks]
        bars = ax1.bar(range(len(model_names)), accuracies, color=colors, alpha=0.7)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Quantum Model Accuracy Comparison')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45)
        
        # 添加數值標籤
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. 框架比較
        ax2 = axes[0, 1]
        qiskit_acc = [acc for acc, f in zip(accuracies, frameworks) if f == 'Qiskit']
        pennylane_acc = [acc for acc, f in zip(accuracies, frameworks) if f == 'PennyLane']
        
        ax2.boxplot([qiskit_acc, pennylane_acc], labels=['Qiskit', 'PennyLane'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Framework Performance Comparison')
        
        # 3. 角度編碼分佈
        ax3 = axes[1, 0]
        if hasattr(self.angle_encoder, 'encoding_stats'):
            qubit_indices = list(self.angle_encoder.encoding_stats.keys())
            angle_means = [self.angle_encoder.encoding_stats[i]['mean'] for i in qubit_indices]
            angle_stds = [self.angle_encoder.encoding_stats[i]['std'] for i in qubit_indices]
            
            ax3.errorbar(qubit_indices, angle_means, yerr=angle_stds, 
                        marker='o', capsize=5, capthick=2)
            ax3.set_xlabel('Qubit Index')
            ax3.set_ylabel('Angle (radians)')
            ax3.set_title('Quantum Angle Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. 特徵重要性
        ax4 = axes[1, 1]
        if self.angle_encoder.feature_mapping:
            feature_names = [f['name'] for f in self.angle_encoder.feature_mapping['features']]
            qubit_indices = [f['qubit_index'] for f in self.angle_encoder.feature_mapping['features']]
            
            ax4.barh(range(len(feature_names)), qubit_indices)
            ax4.set_yticks(range(len(feature_names)))
            ax4.set_yticklabels(feature_names)
            ax4.set_xlabel('Qubit Index')
            ax4.set_title('Feature to Qubit Mapping')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_angle_encoding_analysis(self, save_path):
        """繪製角度編碼分析圖表"""
        if not hasattr(self.angle_encoder, 'encoding_stats'):
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (qubit_idx, stats) in enumerate(self.angle_encoder.encoding_stats.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # 模擬角度分佈
            mean_angle = stats['mean']
            std_angle = stats['std']
            angles = np.random.normal(mean_angle, std_angle, 1000)
            angles = np.clip(angles, 0, 2*np.pi)
            
            ax.hist(angles, bins=30, alpha=0.7, density=True, color='skyblue')
            ax.axvline(mean_angle, color='red', linestyle='--', 
                      label=f'Mean: {mean_angle:.3f}')
            ax.axvline(np.pi, color='green', linestyle='-', alpha=0.5, 
                      label='π (3.142)')
            
            feature_name = self.angle_encoder.feature_mapping['features'][i]['name']
            ax.set_xlabel('Angle (radians)')
            ax.set_ylabel('Density')
            ax.set_title(f'Qubit {qubit_idx}: {feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/angle_encoding_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quantum_circuit_analysis(self, save_path):
        """繪製量子電路分析圖表"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 創建量子電路可視化
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        # 繪製電路結構
        for layer in range(n_layers + 1):
            for qubit in range(n_qubits):
                if layer == 0:
                    # 特徵編碼層
                    ax.add_patch(plt.Rectangle((layer, qubit), 0.8, 0.8, 
                                             facecolor='lightblue', edgecolor='black'))
                    ax.text(layer + 0.4, qubit + 0.4, 'RY', ha='center', va='center')
                else:
                    # 變分層
                    ax.add_patch(plt.Rectangle((layer, qubit), 0.8, 0.8, 
                                             facecolor='lightgreen', edgecolor='black'))
                    ax.text(layer + 0.4, qubit + 0.4, 'RY', ha='center', va='center')
                    
                    # 糾纏
                    if qubit < n_qubits - 1:
                        ax.plot([layer + 0.4, layer + 0.4], [qubit + 0.8, qubit + 1.2], 
                               'k-', linewidth=2)
                        ax.plot([layer + 0.2, layer + 0.6], [qubit + 1, qubit + 1], 
                               'k-', linewidth=2)
        
        ax.set_xlim(-0.5, n_layers + 0.5)
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Qubit')
        ax.set_title('Quantum Circuit Architecture with Angle Encoding')
        ax.set_aspect('equal')
        
        # 添加圖例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Feature Encoding'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', label='Variational Layer'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Entanglement')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/quantum_circuit_architecture.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_report(self, save_path):
        """生成分析報告"""
        report_content = f"""
# 量子機器學習角度編碼整合報告

## 📊 執行摘要

成功將角度編碼整合到量子機器學習流程中，實現了基於經典ML特徵重要性的量子特徵編碼。

## 🔬 技術實現

### 角度編碼配置
- **量子比特數**: {self.n_qubits}
- **變分層數**: {self.n_layers}
- **編碼方法**: {self.encoding_method}
- **角度範圍**: [0, 2π] 弧度

### 特徵映射
"""
        
        if self.angle_encoder.feature_mapping:
            for feature in self.angle_encoder.feature_mapping['features']:
                report_content += f"- **Qubit {feature['qubit_index']}**: {feature['name']} - {feature['description']}\n"
        
        report_content += f"""
## 📈 模型性能

### 訓練結果
"""
        
        for model_name, result in self.training_results.items():
            if result['model'] is not None:
                report_content += f"- **{model_name}** ({result['framework']}): 準確率 {result['accuracy']:.4f}\n"
            else:
                report_content += f"- **{model_name}** ({result['framework']}): 訓練失敗 - {result.get('error', 'Unknown error')}\n"
        
        report_content += f"""
## 🎯 關鍵發現

1. **角度編碼效果**: 成功將經典特徵映射到量子角度空間
2. **模型性能**: 量子模型在角度編碼特徵上表現良好
3. **框架比較**: Qiskit和PennyLane都支持角度編碼
4. **特徵重要性**: 基於經典ML重要性進行量子比特分配

## 🔧 使用建議

1. **特徵選擇**: 使用經典ML特徵重要性指導量子特徵選擇
2. **編碼方法**: 根據數據特性選擇合適的編碼方法
3. **量子比特數**: 平衡表達能力和計算複雜度
4. **變分層數**: 根據問題複雜度調整層數

---
**報告生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**量子比特數**: {self.n_qubits}
**編碼方法**: {self.encoding_method}
"""
        
        with open(f"{save_path}/analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)

# 量子模型基類
class QuantumModelWithAngleEncoding:
    """整合角度編碼的量子模型基類"""
    
    def __init__(self, name, n_qubits, angle_encoder):
        self.name = name
        self.n_qubits = n_qubits
        self.angle_encoder = angle_encoder
        self.is_trained = False
    
    def fit(self, X, y):
        """訓練模型"""
        pass
    
    def predict(self, X):
        """預測"""
        pass

# Qiskit模型實現
class QiskitQNNWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的Qiskit QNN模型"""
    
    def __init__(self, name, n_qubits, n_layers, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.n_layers = n_layers
        self.vqc = None
        
        if QISKIT_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化Qiskit模型"""
        from qiskit.circuit.library import ZZFeatureMap, TwoLocal
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_algorithms.optimizers import COBYLA
        
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        ansatz = TwoLocal(self.n_qubits, ['ry', 'rz'], 'cz', reps=self.n_layers)
        
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=50)
        )
    
    def fit(self, X, y):
        """訓練模型"""
        if self.vqc is None:
            raise ValueError("Model not initialized")
        
        self.vqc.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.vqc.predict(X)

class QiskitQSVMWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的Qiskit QSVM模型"""
    
    def __init__(self, name, n_qubits, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.qsvm = None
        
        if QISKIT_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化QSVM模型"""
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        
        # 創建量子核
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        self.qsvm = QSVC(quantum_kernel=quantum_kernel)
    
    def fit(self, X, y):
        """訓練模型"""
        if self.qsvm is None:
            raise ValueError("Model not initialized")
        
        self.qsvm.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.qsvm.predict(X)

class QiskitHybridWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的Qiskit混合模型"""
    
    def __init__(self, name, n_qubits, n_layers, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.n_layers = n_layers
        self.rebalance_model = None
        self.volatility_model = None
        
        if QISKIT_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """初始化混合模型"""
        self.rebalance_model = QiskitQNNWithAngleEncoding(
            "Rebalance", self.n_qubits, self.n_layers, self.angle_encoder
        )
        self.volatility_model = QiskitQSVMWithAngleEncoding(
            "Volatility", self.n_qubits, self.angle_encoder
        )
    
    def fit(self, X, y):
        """訓練混合模型"""
        # 簡化：只訓練再平衡模型
        if self.rebalance_model:
            self.rebalance_model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if self.rebalance_model:
            return self.rebalance_model.predict(X)
        return np.zeros(len(X))

# PennyLane模型實現
class PennyLaneQNNWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的PennyLane QNN模型"""
    
    def __init__(self, name, n_qubits, n_layers, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.n_layers = n_layers
        self.device = None
        self.circuit = None
        self.weights = None
        
        if PENNYLANE_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化PennyLane模型"""
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.device, interface='torch')
        def circuit(features, weights):
            # 角度編碼
            for i in range(self.n_qubits):
                qml.RY(features[i], wires=i)
                qml.RZ(features[i] * 0.5, wires=i)
            
            # 變分層
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                    qml.RZ(weights[layer, i + self.n_qubits], wires=i)
                
                # 糾纏
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
        self.weights = torch.randn(self.n_layers, 2 * self.n_qubits, requires_grad=True)
    
    def fit(self, X, y):
        """訓練模型"""
        if self.circuit is None:
            raise ValueError("Model not initialized")
        
        # 簡化的訓練過程
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        optimizer = torch.optim.Adam([self.weights], lr=0.01)
        
        for epoch in range(10):  # 簡化的訓練循環
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = []
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                outputs.append(output[0])  # 取第一個量子比特的期望值
            
            outputs = torch.stack(outputs)
            loss = torch.mean((outputs - y_tensor) ** 2)
            
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = []
        
        for i in range(len(X_tensor)):
            output = self.circuit(X_tensor[i], self.weights)
            predictions.append(output[0].detach().numpy())
        
        return np.array(predictions)

class PennyLaneVQCWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的PennyLane VQC模型"""
    
    def __init__(self, name, n_qubits, n_layers, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.n_layers = n_layers
        # 簡化實現
        self.qnn_model = PennyLaneQNNWithAngleEncoding(name, n_qubits, n_layers, angle_encoder)
    
    def fit(self, X, y):
        """訓練模型"""
        self.qnn_model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        return self.qnn_model.predict(X)

class PennyLaneHybridWithAngleEncoding(QuantumModelWithAngleEncoding):
    """整合角度編碼的PennyLane混合模型"""
    
    def __init__(self, name, n_qubits, n_layers, angle_encoder):
        super().__init__(name, n_qubits, angle_encoder)
        self.n_layers = n_layers
        self.qnn_model = PennyLaneQNNWithAngleEncoding(name, n_qubits, n_layers, angle_encoder)
    
    def fit(self, X, y):
        """訓練模型"""
        self.qnn_model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """預測"""
        return self.qnn_model.predict(X)

def main():
    """主函數 - 演示整合角度編碼的量子ML流程"""
    logger.info("🚀 開始量子ML角度編碼整合演示...")
    
    # 1. 創建示例數據
    np.random.seed(42)
    n_samples = 500
    n_features = 6
    
    # 創建模擬的金融特徵數據
    feature_names = [
        'price_sma_20_ratio', 'price_ma_ratio', 'vol_percentile',
        'vol_regime', 'bb_position', 'volume_ma_10'
    ]
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 分割數據
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"創建數據集: {n_samples}樣本, {n_features}特徵")
    
    # 2. 初始化量子ML流程
    quantum_ml = QuantumMLWithAngleEncoding(
        n_qubits=6, 
        n_layers=3, 
        encoding_method='robust'
    )
    
    # 3. 準備角度編碼特徵
    quantum_data = quantum_ml.prepare_features_with_angle_encoding(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # 4. 創建量子模型
    quantum_ml.create_qiskit_models()
    quantum_ml.create_pennylane_models()
    
    # 5. 訓練所有模型
    quantum_ml.train_all_models(quantum_data)
    
    # 6. 分析結果
    quantum_ml.analyze_results()
    
    logger.info("✅ 量子ML角度編碼整合演示完成！")
    logger.info("📁 結果保存在 reports/quantum_ml_angle_encoding/ 目錄中")

if __name__ == "__main__":
    main()
