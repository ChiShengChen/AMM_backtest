"""
整合角度編碼的量子機器學習模型
基於經典ML特徵重要性的量子特徵編碼
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import joblib
from datetime import datetime
import sys
import os

# 添加項目根目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# 導入角度編碼器
try:
    from quantum_angle_encoding import QuantumAngleEncoder
    ANGLE_ENCODING_AVAILABLE = True
except ImportError as e:
    ANGLE_ENCODING_AVAILABLE = False
    logging.warning(f"Angle encoding not available: {e}")

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.circuit.library import ZZFeatureMap, TwoLocal
    QUANTUM_AVAILABLE = True
except ImportError as e:
    QUANTUM_AVAILABLE = False
    logging.warning(f"Quantum computing libraries not available: {e}. Install qiskit and pennylane.")

logger = logging.getLogger(__name__)

class QuantumModelWithAngleEncodingBase(ABC):
    """整合角度編碼的量子模型基類"""
    
    def __init__(self, name: str, n_qubits: int = 6, n_layers: int = 3, 
                 encoding_method: str = 'robust', feature_names: List[str] = None):
        self.name = name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_method = encoding_method
        self.feature_names = feature_names or []
        self.is_trained = False
        self.training_history = []
        
        # 初始化角度編碼器
        if ANGLE_ENCODING_AVAILABLE:
            self.angle_encoder = QuantumAngleEncoder(
                n_qubits=n_qubits, 
                encoding_method=encoding_method
            )
            self._setup_feature_mapping()
        else:
            self.angle_encoder = None
            logger.warning("Angle encoding not available, using standard features")
    
    def _setup_feature_mapping(self):
        """設置特徵映射"""
        if self.angle_encoder and self.feature_names:
            self.angle_encoder.create_feature_mapping(self.feature_names)
            logger.info(f"設置特徵映射: {len(self.feature_names)}個特徵映射到{self.n_qubits}個量子比特")
    
    def prepare_features(self, X: np.ndarray, fit_encoder: bool = True) -> np.ndarray:
        """
        使用角度編碼準備特徵
        
        Args:
            X: 輸入特徵
            fit_encoder: 是否擬合編碼器
        
        Returns:
            編碼後的量子特徵
        """
        if self.angle_encoder is None:
            # 如果沒有角度編碼器，返回原始特徵
            return X
        
        # 轉換為DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        else:
            X_df = X
        
        # 擬合編碼器（如果需要）
        if fit_encoder:
            self.angle_encoder.fit_scalers(X_df)
        
        # 編碼特徵
        quantum_features = self.angle_encoder.encode_features(X_df)
        
        logger.info(f"特徵編碼完成: {X.shape} -> {quantum_features.shape}")
        return quantum_features
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練量子模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'name': self.name,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoding_method': self.encoding_method,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'angle_encoder': self.angle_encoder,
            'model_params': self._get_model_params()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved quantum model with angle encoding to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加載模型"""
        model_data = joblib.load(filepath)
        
        self.name = model_data['name']
        self.n_qubits = model_data['n_qubits']
        self.n_layers = model_data['n_layers']
        self.encoding_method = model_data.get('encoding_method', 'robust')
        self.feature_names = model_data.get('feature_names', [])
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.angle_encoder = model_data.get('angle_encoder', None)
        
        self._set_model_params(model_data['model_params'])
        logger.info(f"Loaded quantum model with angle encoding from {filepath}")
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        pass
    
    @abstractmethod
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        pass

class QNNRebalancePredictorWithAngleEncoding(QuantumModelWithAngleEncodingBase):
    """整合角度編碼的QNN再平衡預測器"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3, 
                 encoding_method: str = 'robust', feature_names: List[str] = None):
        super().__init__("QNN_Rebalance_Predictor_AngleEncoded", n_qubits, n_layers, 
                        encoding_method, feature_names)
        
        self.vqc = None
        self.sampler = None
        self.optimizer = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available. Please install qiskit and qiskit-machine-learning.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化VQC模型"""
        # 創建特徵映射
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        
        # 創建變分形式
        ansatz = TwoLocal(self.n_qubits, ['ry', 'rz'], 'cz', reps=self.n_layers)
        
        # 創建VQC
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=100),
            sampler=Sampler()
        )
        
        logger.info(f"初始化QNN: {self.n_qubits}量子比特, {self.n_layers}層")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練QNN模型"""
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        logger.info(f"訓練QNN: {len(X)}樣本, {X.shape[1]}特徵")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=True)
        
        # 訓練模型
        self.vqc.fit(X_encoded, y)
        self.is_trained = True
        
        logger.info("QNN訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        return self.vqc.predict(X_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        return self.vqc.predict_proba(X_encoded)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'vqc_params': self.vqc.get_params() if self.vqc else None,
            'sampler': self.sampler,
            'optimizer': self.optimizer
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        if params.get('vqc_params'):
            self.vqc.set_params(**params['vqc_params'])
        self.sampler = params.get('sampler')
        self.optimizer = params.get('optimizer')

class QSVMVolatilityPredictorWithAngleEncoding(QuantumModelWithAngleEncodingBase):
    """整合角度編碼的QSVM波動率預測器"""
    
    def __init__(self, n_qubits: int = 6, encoding_method: str = 'robust', 
                 feature_names: List[str] = None):
        super().__init__("QSVM_Volatility_Predictor_AngleEncoded", n_qubits, 0, 
                        encoding_method, feature_names)
        
        self.qsvm = None
        self.quantum_kernel = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available. Please install qiskit and qiskit-machine-learning.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化QSVM模型"""
        # 創建量子核
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        self.quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # 創建QSVM
        self.qsvm = QSVC(quantum_kernel=self.quantum_kernel)
        
        logger.info(f"初始化QSVM: {self.n_qubits}量子比特")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練QSVM模型"""
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        logger.info(f"訓練QSVM: {len(X)}樣本, {X.shape[1]}特徵")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=True)
        
        # 訓練模型
        self.qsvm.fit(X_encoded, y)
        self.is_trained = True
        
        logger.info("QSVM訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        return self.qsvm.predict(X_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        # QSVM沒有predict_proba，返回預測結果
        predictions = self.predict(X_encoded)
        return np.column_stack([1 - predictions, predictions])
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'qsvm_params': self.qsvm.get_params() if self.qsvm else None,
            'quantum_kernel': self.quantum_kernel
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        if params.get('qsvm_params'):
            self.qsvm.set_params(**params['qsvm_params'])
        self.quantum_kernel = params.get('quantum_kernel')

class QuantumHybridStrategyWithAngleEncoding(QuantumModelWithAngleEncodingBase):
    """整合角度編碼的量子混合策略"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3, 
                 encoding_method: str = 'robust', feature_names: List[str] = None):
        super().__init__("Quantum_Hybrid_Strategy_AngleEncoded", n_qubits, n_layers, 
                        encoding_method, feature_names)
        
        self.rebalance_model = None
        self.volatility_model = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available. Please install qiskit and qiskit-machine-learning.")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化混合模型"""
        self.rebalance_model = QNNRebalancePredictorWithAngleEncoding(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            encoding_method=self.encoding_method,
            feature_names=self.feature_names
        )
        
        self.volatility_model = QSVMVolatilityPredictorWithAngleEncoding(
            n_qubits=self.n_qubits,
            encoding_method=self.encoding_method,
            feature_names=self.feature_names
        )
        
        logger.info("初始化量子混合策略")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練混合模型"""
        logger.info(f"訓練量子混合策略: {len(X)}樣本")
        
        # 訓練再平衡模型
        self.rebalance_model.fit(X, y)
        
        # 訓練波動率模型（使用相同的標籤，實際應用中可能需要不同的標籤）
        self.volatility_model.fit(X, y)
        
        self.is_trained = True
        logger.info("量子混合策略訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用再平衡模型進行預測
        return self.rebalance_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用再平衡模型進行概率預測
        return self.rebalance_model.predict_proba(X)
    
    def predict_volatility(self, X: np.ndarray) -> np.ndarray:
        """預測波動率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.volatility_model.predict(X)
    
    def predict_volatility_proba(self, X: np.ndarray) -> np.ndarray:
        """預測波動率概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.volatility_model.predict_proba(X)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'rebalance_model': self.rebalance_model,
            'volatility_model': self.volatility_model
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        self.rebalance_model = params.get('rebalance_model')
        self.volatility_model = params.get('volatility_model')

def create_quantum_models_with_angle_encoding(feature_names: List[str], 
                                            n_qubits: int = 6, 
                                            n_layers: int = 3,
                                            encoding_method: str = 'robust') -> Dict[str, QuantumModelWithAngleEncodingBase]:
    """
    創建整合角度編碼的量子模型
    
    Args:
        feature_names: 特徵名稱列表
        n_qubits: 量子比特數
        n_layers: 變分層數
        encoding_method: 角度編碼方法
    
    Returns:
        量子模型字典
    """
    models = {}
    
    if not QUANTUM_AVAILABLE:
        logger.warning("量子計算庫不可用，無法創建量子模型")
        return models
    
    if not ANGLE_ENCODING_AVAILABLE:
        logger.warning("角度編碼不可用，使用標準特徵")
        return models
    
    try:
        # QNN再平衡預測器
        models['qnn_rebalance'] = QNNRebalancePredictorWithAngleEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        # QSVM波動率預測器
        models['qsvm_volatility'] = QSVMVolatilityPredictorWithAngleEncoding(
            n_qubits=n_qubits,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        # 量子混合策略
        models['quantum_hybrid'] = QuantumHybridStrategyWithAngleEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        logger.info(f"創建了 {len(models)} 個整合角度編碼的量子模型")
        
    except Exception as e:
        logger.error(f"創建量子模型時發生錯誤: {e}")
    
    return models

def main():
    """測試函數"""
    logger.info("測試整合角度編碼的量子模型...")
    
    # 創建示例特徵名稱
    feature_names = [
        'price_sma_20_ratio', 'price_ma_ratio', 'vol_percentile',
        'vol_regime', 'bb_position', 'volume_ma_10'
    ]
    
    # 創建量子模型
    models = create_quantum_models_with_angle_encoding(
        feature_names=feature_names,
        n_qubits=6,
        n_layers=3,
        encoding_method='robust'
    )
    
    if models:
        logger.info(f"成功創建 {len(models)} 個量子模型")
        for name, model in models.items():
            logger.info(f"- {name}: {model.name}")
    else:
        logger.warning("未能創建任何量子模型")

if __name__ == "__main__":
    main()
