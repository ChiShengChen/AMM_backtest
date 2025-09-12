#!/usr/bin/env python3
"""
角度編碼實現：將經典特徵映射到 [0, 2π] 範圍
基於經典ML特徵重要性分析結果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QuantumAngleEncoder:
    """量子角度編碼器"""
    
    def __init__(self, n_qubits=6, encoding_method='standard'):
        """
        初始化角度編碼器
        
        Args:
            n_qubits: 量子比特數量
            encoding_method: 編碼方法 ('standard', 'robust', 'minmax', 'custom')
        """
        self.n_qubits = n_qubits
        self.encoding_method = encoding_method
        self.scalers = {}
        self.feature_mapping = {}
        self.encoding_stats = {}
        
    def create_feature_mapping(self, top_features):
        """創建特徵映射配置"""
        # 基於經典ML重要性分析的特徵映射
        self.feature_mapping = {
            'n_qubits': self.n_qubits,
            'features': [
                {
                    'name': 'price_momentum',
                    'classical_features': ['price_sma_20_ratio', 'price_ma_ratio'],
                    'qubit_index': 0,
                    'weight': [0.6, 0.4],  # 特徵權重
                    'description': 'Price momentum and trend'
                },
                {
                    'name': 'price_ma_ratio',
                    'classical_features': ['price_sma_20_ratio'],
                    'qubit_index': 1,
                    'weight': [1.0],
                    'description': 'Price to moving average ratio'
                },
                {
                    'name': 'volatility_level',
                    'classical_features': ['vol_percentile'],
                    'qubit_index': 2,
                    'weight': [1.0],
                    'description': 'Current volatility level'
                },
                {
                    'name': 'volatility_regime',
                    'classical_features': ['vol_regime'],
                    'qubit_index': 3,
                    'weight': [1.0],
                    'description': 'Volatility regime state'
                },
                {
                    'name': 'technical_signal',
                    'classical_features': ['bb_position'],
                    'qubit_index': 4,
                    'weight': [1.0],
                    'description': 'Technical indicator signal'
                },
                {
                    'name': 'volume_signal',
                    'classical_features': ['volume_ma_10'],
                    'qubit_index': 5,
                    'weight': [1.0],
                    'description': 'Volume signal'
                }
            ]
        }
        
        logger.info(f"Created feature mapping for {self.n_qubits} qubits")
        return self.feature_mapping
    
    def fit_scalers(self, X_train):
        """擬合標準化器"""
        logger.info(f"Fitting scalers using {self.encoding_method} method...")
        
        for feature_config in self.feature_mapping['features']:
            qubit_idx = feature_config['qubit_index']
            classical_features = feature_config['classical_features']
            
            # 選擇存在的特徵
            available_features = [f for f in classical_features if f in X_train.columns]
            
            if not available_features:
                logger.warning(f"No features available for qubit {qubit_idx}")
                continue
            
            # 提取特徵數據
            feature_data = X_train[available_features].values
            
            # 根據權重組合特徵
            if len(available_features) > 1:
                weights = feature_config['weight'][:len(available_features)]
                weights = np.array(weights) / np.sum(weights)  # 歸一化權重
                combined_feature = np.dot(feature_data, weights)
            else:
                combined_feature = feature_data.flatten()
            
            # 擬合標準化器
            if self.encoding_method == 'standard':
                scaler = StandardScaler()
            elif self.encoding_method == 'robust':
                scaler = RobustScaler()
            elif self.encoding_method == 'minmax':
                scaler = MinMaxScaler(feature_range=(-3, 3))
            else:  # custom
                scaler = self._create_custom_scaler(combined_feature)
            
            scaler.fit(combined_feature.reshape(-1, 1))
            self.scalers[qubit_idx] = {
                'scaler': scaler,
                'features': available_features,
                'weights': feature_config['weight'][:len(available_features)]
            }
            
            # 記錄統計信息
            self.encoding_stats[qubit_idx] = {
                'mean': np.mean(combined_feature),
                'std': np.std(combined_feature),
                'min': np.min(combined_feature),
                'max': np.max(combined_feature),
                'q25': np.percentile(combined_feature, 25),
                'q75': np.percentile(combined_feature, 75)
            }
        
        logger.info(f"Fitted scalers for {len(self.scalers)} qubits")
    
    def _create_custom_scaler(self, data):
        """創建自定義標準化器"""
        class CustomScaler:
            def __init__(self, data):
                self.data_min = np.min(data)
                self.data_max = np.max(data)
                self.data_mean = np.mean(data)
                self.data_std = np.std(data)
                
            def fit(self, X):
                pass
                
            def transform(self, X):
                # 使用robust標準化
                X_robust = (X - self.data_mean) / (self.data_std + 1e-8)
                # 限制到 [-3, 3] 範圍
                X_clipped = np.clip(X_robust, -3, 3)
                return X_clipped
        
        return CustomScaler(data)
    
    def encode_features(self, X):
        """將經典特徵編碼為量子角度"""
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        n_samples = len(X)
        quantum_features = np.zeros((n_samples, self.n_qubits))
        
        for qubit_idx, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            features = scaler_info['features']
            weights = scaler_info['weights']
            
            # 提取特徵數據
            feature_data = X[features].values
            
            # 根據權重組合特徵
            if len(features) > 1:
                weights = np.array(weights) / np.sum(weights)
                combined_feature = np.dot(feature_data, weights)
            else:
                combined_feature = feature_data.flatten()
            
            # 標準化
            normalized_feature = scaler.transform(combined_feature.reshape(-1, 1)).flatten()
            
            # 映射到 [0, 2π]
            if self.encoding_method == 'minmax':
                # 已經在 [-3, 3] 範圍，直接映射
                angle_feature = (normalized_feature + 3) / 6 * 2 * np.pi
            else:
                # 其他方法需要先限制到 [-3, 3]
                clipped_feature = np.clip(normalized_feature, -3, 3)
                angle_feature = (clipped_feature + 3) / 6 * 2 * np.pi
            
            quantum_features[:, qubit_idx] = angle_feature
        
        return quantum_features
    
    def decode_features(self, quantum_features):
        """將量子角度解碼回經典特徵範圍"""
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        n_samples = len(quantum_features)
        decoded_features = np.zeros((n_samples, self.n_qubits))
        
        for qubit_idx, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            
            # 從 [0, 2π] 映射回標準化範圍
            if self.encoding_method == 'minmax':
                normalized_feature = (quantum_features[:, qubit_idx] / (2 * np.pi)) * 6 - 3
            else:
                normalized_feature = (quantum_features[:, qubit_idx] / (2 * np.pi)) * 6 - 3
            
            # 反標準化
            try:
                decoded_feature = scaler.inverse_transform(normalized_feature.reshape(-1, 1)).flatten()
            except:
                # 如果沒有inverse_transform，使用統計信息
                stats = self.encoding_stats[qubit_idx]
                decoded_feature = normalized_feature * stats['std'] + stats['mean']
            
            decoded_features[:, qubit_idx] = decoded_feature
        
        return decoded_features
    
    def analyze_encoding_quality(self, X_train, X_test=None):
        """分析編碼質量"""
        logger.info("Analyzing encoding quality...")
        
        # 編碼訓練數據
        quantum_train = self.encode_features(X_train)
        
        # 解碼並比較
        decoded_train = self.decode_features(quantum_train)
        
        analysis_results = {
            'encoding_stats': self.encoding_stats,
            'quantum_angles': {
                'mean': np.mean(quantum_train, axis=0),
                'std': np.std(quantum_train, axis=0),
                'min': np.min(quantum_train, axis=0),
                'max': np.max(quantum_train, axis=0)
            },
            'reconstruction_error': {}
        }
        
        # 計算重建誤差
        for qubit_idx in range(self.n_qubits):
            if qubit_idx in self.scalers:
                original_features = self.scalers[qubit_idx]['features']
                if len(original_features) == 1:
                    original_data = X_train[original_features[0]].values
                    reconstructed_data = decoded_train[:, qubit_idx]
                    
                    mse = np.mean((original_data - reconstructed_data) ** 2)
                    mae = np.mean(np.abs(original_data - reconstructed_data))
                    
                    analysis_results['reconstruction_error'][qubit_idx] = {
                        'mse': mse,
                        'mae': mae,
                        'correlation': np.corrcoef(original_data, reconstructed_data)[0, 1]
                    }
        
        if X_test is not None:
            quantum_test = self.encode_features(X_test)
            analysis_results['test_quantum_angles'] = {
                'mean': np.mean(quantum_test, axis=0),
                'std': np.std(quantum_test, axis=0),
                'min': np.min(quantum_test, axis=0),
                'max': np.max(quantum_test, axis=0)
            }
        
        return analysis_results
    
    def plot_encoding_analysis(self, X_train, X_test=None, save_path="reports/quantum_angle_encoding"):
        """繪製編碼分析圖表"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 編碼數據
        quantum_train = self.encode_features(X_train)
        if X_test is not None:
            quantum_test = self.encode_features(X_test)
        
        # 創建圖表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for qubit_idx in range(self.n_qubits):
            ax = axes[qubit_idx]
            
            if qubit_idx in self.scalers:
                feature_name = self.feature_mapping['features'][qubit_idx]['name']
                
                # 繪製角度分布
                ax.hist(quantum_train[:, qubit_idx], bins=30, alpha=0.7, 
                       label='Train', density=True, color='blue')
                
                if X_test is not None:
                    ax.hist(quantum_test[:, qubit_idx], bins=30, alpha=0.7, 
                           label='Test', density=True, color='red')
                
                ax.set_xlabel('Angle (radians)')
                ax.set_ylabel('Density')
                ax.set_title(f'Qubit {qubit_idx}: {feature_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 添加統計信息
                stats = self.encoding_stats[qubit_idx]
                ax.text(0.02, 0.98, f'Mean: {np.mean(quantum_train[:, qubit_idx]):.3f}\n'
                                    f'Std: {np.std(quantum_train[:, qubit_idx]):.3f}\n'
                                    f'Range: [{np.min(quantum_train[:, qubit_idx]):.3f}, {np.max(quantum_train[:, qubit_idx]):.3f}]',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/angle_encoding_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 繪製角度相關性矩陣
        plt.figure(figsize=(10, 8))
        correlation_matrix = np.corrcoef(quantum_train.T)
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[f'Qubit {i}' for i in range(self.n_qubits)],
                   yticklabels=[f'Qubit {i}' for i in range(self.n_qubits)])
        plt.title('Quantum Angle Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{save_path}/angle_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 繪製特徵重要性與角度範圍的關係
        plt.figure(figsize=(12, 8))
        
        qubit_indices = list(self.scalers.keys())
        angle_ranges = [np.max(quantum_train[:, i]) - np.min(quantum_train[:, i]) for i in qubit_indices]
        angle_means = [np.mean(quantum_train[:, i]) for i in qubit_indices]
        
        plt.scatter(qubit_indices, angle_ranges, s=100, alpha=0.7, label='Angle Range')
        plt.scatter(qubit_indices, angle_means, s=100, alpha=0.7, label='Angle Mean')
        
        for i, qubit_idx in enumerate(qubit_indices):
            feature_name = self.feature_mapping['features'][qubit_idx]['name']
            plt.annotate(f'{feature_name}\nQ{qubit_idx}', 
                        (qubit_idx, angle_ranges[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Qubit Index')
        plt.ylabel('Angle Value')
        plt.title('Feature Importance vs Quantum Angle Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance_vs_angles.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Encoding analysis plots saved to {save_path}")

def create_sample_data():
    """創建示例數據"""
    np.random.seed(42)
    n_samples = 1000
    
    # 創建模擬的金融特徵數據
    data = {
        'price_sma_20_ratio': np.random.normal(1.0, 0.1, n_samples),
        'price_ma_ratio': np.random.normal(1.0, 0.08, n_samples),
        'price_ema_20_ratio': np.random.normal(1.0, 0.09, n_samples),
        'price_ema_10_ratio': np.random.normal(1.0, 0.07, n_samples),
        'price_ema_50_ratio': np.random.normal(1.0, 0.12, n_samples),
        'price_sma_50_ratio': np.random.normal(1.0, 0.11, n_samples),
        'price_sma_10_ratio': np.random.normal(1.0, 0.06, n_samples),
        'oc_ratio': np.random.normal(1.0, 0.05, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'price_ema_5_ratio': np.random.normal(1.0, 0.05, n_samples),
        'vol_percentile': np.random.uniform(0, 1, n_samples),
        'vol_regime': np.random.randint(0, 2, n_samples),
        'volume_ma_10': np.random.normal(1000, 200, n_samples),
        'rsi': np.random.uniform(0, 1, n_samples),
        'volume_ma_20': np.random.normal(1000, 180, n_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """主函數"""
    logger.info("🚀 Starting quantum angle encoding implementation...")
    
    # 創建示例數據
    data = create_sample_data()
    
    # 分割數據
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    
    logger.info(f"Created dataset with {len(data)} samples and {len(data.columns)} features")
    
    # 測試不同的編碼方法
    encoding_methods = ['standard', 'robust', 'minmax', 'custom']
    
    for method in encoding_methods:
        logger.info(f"\n🔬 Testing {method} encoding method...")
        
        # 創建編碼器
        encoder = QuantumAngleEncoder(n_qubits=6, encoding_method=method)
        
        # 創建特徵映射
        top_features = ['price_sma_20_ratio', 'price_ma_ratio', 'price_ema_20_ratio', 
                       'price_ema_10_ratio', 'price_ema_50_ratio', 'price_sma_50_ratio']
        encoder.create_feature_mapping(top_features)
        
        # 擬合標準化器
        encoder.fit_scalers(X_train)
        
        # 編碼特徵
        quantum_train = encoder.encode_features(X_train)
        quantum_test = encoder.encode_features(X_test)
        
        # 分析編碼質量
        analysis = encoder.analyze_encoding_quality(X_train, X_test)
        
        # 繪製分析圖表
        encoder.plot_encoding_analysis(X_train, X_test, f"reports/quantum_angle_encoding/{method}")
        
        # 打印結果
        logger.info(f"✅ {method} encoding completed")
        logger.info(f"   Quantum angles range: [{np.min(quantum_train):.3f}, {np.max(quantum_train):.3f}]")
        logger.info(f"   Mean angle: {np.mean(quantum_train):.3f}")
        logger.info(f"   Std angle: {np.std(quantum_train):.3f}")
        
        # 保存結果
        results_df = pd.DataFrame(quantum_train, columns=[f'Qubit_{i}' for i in range(6)])
        results_df.to_csv(f"reports/quantum_angle_encoding/{method}/quantum_features.csv", index=False)
        
        # 保存分析結果
        import json
        with open(f"reports/quantum_angle_encoding/{method}/encoding_analysis.json", 'w') as f:
            # 轉換numpy類型為Python類型
            analysis_serializable = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    analysis_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            analysis_serializable[key][k] = {}
                            for k2, v2 in v.items():
                                if isinstance(v2, np.ndarray):
                                    analysis_serializable[key][k][k2] = v2.tolist()
                                elif isinstance(v2, (np.integer, np.floating)):
                                    analysis_serializable[key][k][k2] = float(v2)
                                else:
                                    analysis_serializable[key][k][k2] = v2
                        elif isinstance(v, np.ndarray):
                            analysis_serializable[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            analysis_serializable[key][k] = float(v)
                        else:
                            analysis_serializable[key][k] = v
                elif isinstance(value, np.ndarray):
                    analysis_serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    analysis_serializable[key] = float(value)
                else:
                    analysis_serializable[key] = value
            json.dump(analysis_serializable, f, indent=2)
    
    # 創建比較報告
    create_comparison_report(encoding_methods)
    
    logger.info("✅ Quantum angle encoding implementation completed!")
    logger.info("📁 Results saved to reports/quantum_angle_encoding/")

def create_comparison_report(methods):
    """創建編碼方法比較報告"""
    report_content = f"""
# 量子角度編碼方法比較報告

## 編碼方法比較

### 測試的方法
{', '.join(methods)}

### 編碼原理

#### 1. Standard Encoding
- 使用StandardScaler進行標準化
- 將特徵值限制到[-3, 3]範圍
- 映射到[0, 2π]角度範圍

#### 2. Robust Encoding  
- 使用RobustScaler進行標準化
- 對異常值更魯棒
- 映射到[0, 2π]角度範圍

#### 3. MinMax Encoding
- 使用MinMaxScaler直接映射到[-3, 3]
- 保持原始分佈形狀
- 映射到[0, 2π]角度範圍

#### 4. Custom Encoding
- 自定義標準化方法
- 基於數據統計特性
- 映射到[0, 2π]角度範圍

### 量子特徵映射

| Qubit | 特徵名稱 | 經典特徵 | 描述 |
|-------|----------|----------|------|
| 0 | price_momentum | price_sma_20_ratio, price_ma_ratio | 價格動量和趨勢 |
| 1 | price_ma_ratio | price_sma_20_ratio | 價格與移動平均比率 |
| 2 | volatility_level | vol_percentile | 波動性水平 |
| 3 | volatility_regime | vol_regime | 波動性狀態 |
| 4 | technical_signal | bb_position | 技術指標信號 |
| 5 | volume_signal | volume_ma_10 | 成交量信號 |

### 實現特點

1. **角度編碼**: 將經典特徵映射到[0, 2π]範圍
2. **特徵組合**: 支持多個特徵的加權組合
3. **標準化**: 多種標準化方法可選
4. **可逆性**: 支持從量子角度解碼回經典特徵
5. **質量分析**: 提供編碼質量評估

### 使用建議

- **Standard**: 適用於正態分佈的特徵
- **Robust**: 適用於有異常值的特徵  
- **MinMax**: 適用於需要保持分佈形狀的特徵
- **Custom**: 適用於特殊需求的特徵

---
**報告生成時間**: 2025-01-27
**編碼方法**: 角度編碼 (Angle Encoding)
**量子比特數**: 6
"""
    
    with open("reports/quantum_angle_encoding/encoding_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    main()
