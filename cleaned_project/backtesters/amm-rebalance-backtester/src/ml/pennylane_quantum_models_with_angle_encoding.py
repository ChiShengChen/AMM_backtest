"""
整合角度編碼的PennyLane量子機器學習模型
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

# PennyLane imports
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
    import torch
    import torch.nn as nn
    PENNYLANE_AVAILABLE = True
except ImportError as e:
    PENNYLANE_AVAILABLE = False
    logging.warning(f"PennyLane not available: {e}. Install pennylane.")

logger = logging.getLogger(__name__)

class PennyLaneQuantumModelWithAngleEncodingBase(ABC):
    """整合角度編碼的PennyLane量子模型基類"""
    
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
        
        # PennyLane設備和電路
        self.device = None
        self.circuit = None
        self.weights = None
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available. Please install pennylane.")
        
        self._initialize_device()
    
    def _setup_feature_mapping(self):
        """設置特徵映射"""
        if self.angle_encoder and self.feature_names:
            self.angle_encoder.create_feature_mapping(self.feature_names)
            logger.info(f"設置特徵映射: {len(self.feature_names)}個特徵映射到{self.n_qubits}個量子比特")
    
    def _initialize_device(self):
        """初始化量子設備"""
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        logger.info(f"初始化PennyLane設備: {self.n_qubits}量子比特")
    
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
    
    def create_quantum_circuit(self):
        """創建量子電路"""
        @qml.qnode(self.device, interface='torch')
        def circuit(features, weights):
            # 角度編碼層
            for i in range(self.n_qubits):
                qml.RY(features[i], wires=i)
                qml.RZ(features[i] * 0.5, wires=i)
            
            # 變分層
            for layer in range(self.n_layers):
                # 旋轉門
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                    qml.RZ(weights[layer, i + self.n_qubits], wires=i)
                
                # 糾纏層 - 基於特徵重要性設計
                # 價格和波動性特徵糾纏
                if self.n_qubits >= 4:
                    qml.CNOT(wires=[0, 2])  # price_momentum -> volatility_level
                    qml.CNOT(wires=[1, 3])  # price_ma_ratio -> volatility_regime
                
                # 技術指標和成交量糾纏
                if self.n_qubits >= 6:
                    qml.CNOT(wires=[4, 5])  # technical_signal -> volume_signal
                
                # 全局糾纏
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # 測量期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
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
            'weights': self.weights.detach().numpy() if self.weights is not None else None,
            'model_params': self._get_model_params()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved PennyLane quantum model with angle encoding to {filepath}")
    
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
        
        if model_data.get('weights') is not None:
            self.weights = torch.tensor(model_data['weights'], requires_grad=True)
        
        self._set_model_params(model_data['model_params'])
        logger.info(f"Loaded PennyLane quantum model with angle encoding from {filepath}")
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        pass
    
    @abstractmethod
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        pass

class PennyLaneQNNWithAngleEncoding(PennyLaneQuantumModelWithAngleEncodingBase):
    """整合角度編碼的QNN模型"""
    
    def __init__(self, name: str = "PennyLane_QNN_AngleEncoded", n_qubits: int = 6, 
                 n_layers: int = 3, encoding_method: str = 'robust', 
                 feature_names: List[str] = None):
        super().__init__(name, n_qubits, n_layers, encoding_method, feature_names)
        
        # 初始化權重
        self.weights = torch.randn(self.n_layers, 2 * self.n_qubits, requires_grad=True)
        
        # 創建量子電路
        self.circuit = self.create_quantum_circuit()
        
        logger.info(f"初始化QNN: {self.n_qubits}量子比特, {self.n_layers}層")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練QNN模型"""
        logger.info(f"訓練QNN: {len(X)}樣本, {X.shape[1]}特徵")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=True)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # 優化器
        optimizer = torch.optim.Adam([self.weights], lr=0.01)
        
        # 訓練循環
        n_epochs = 50
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = []
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                outputs.append(output[0])  # 取第一個量子比特的期望值
            
            outputs = torch.stack(outputs)
            
            # 計算損失
            loss = torch.mean((outputs - y_tensor) ** 2)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 記錄訓練歷史
            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item()
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        logger.info("QNN訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                predictions.append(output[0].item())
        
        # 二分類：將連續值轉換為0或1
        predictions = np.array(predictions)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        
        probabilities = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                prob = torch.sigmoid(output[0]).item()
                probabilities.append([1 - prob, prob])
        
        return np.array(probabilities)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'weights': self.weights.detach().numpy() if self.weights is not None else None,
            'circuit': str(self.circuit)
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        if params.get('weights') is not None:
            self.weights = torch.tensor(params['weights'], requires_grad=True)

class PennyLaneVQCWithAngleEncoding(PennyLaneQuantumModelWithAngleEncodingBase):
    """整合角度編碼的PennyLane VQC模型"""
    
    def __init__(self, name: str = "PennyLane_VQC_AngleEncoded", n_qubits: int = 6, 
                 n_layers: int = 3, encoding_method: str = 'robust', 
                 feature_names: List[str] = None):
        super().__init__(name, n_qubits, n_layers, encoding_method, feature_names)
        
        # 初始化權重
        self.weights = torch.randn(self.n_layers, 2 * self.n_qubits, requires_grad=True)
        
        # 創建量子電路
        self.circuit = self.create_quantum_circuit()
        
        logger.info(f"初始化PennyLane VQC: {self.n_qubits}量子比特, {self.n_layers}層")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練PennyLane VQC模型"""
        logger.info(f"訓練PennyLane VQC: {len(X)}樣本, {X.shape[1]}特徵")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=True)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # 優化器
        optimizer = torch.optim.Adam([self.weights], lr=0.01)
        
        # 訓練循環
        n_epochs = 50
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = []
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                # 使用所有量子比特的期望值
                outputs.append(torch.sum(output))
            
            outputs = torch.stack(outputs)
            
            # 計算損失
            loss = torch.mean((outputs - y_tensor) ** 2)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 記錄訓練歷史
            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item()
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        logger.info("PennyLane VQC訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                predictions.append(torch.sum(output).item())
        
        # 二分類：將連續值轉換為0或1
        predictions = np.array(predictions)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用角度編碼準備特徵
        X_encoded = self.prepare_features(X, fit_encoder=False)
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        
        probabilities = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                output = self.circuit(X_tensor[i], self.weights)
                prob = torch.sigmoid(torch.sum(output)).item()
                probabilities.append([1 - prob, prob])
        
        return np.array(probabilities)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'weights': self.weights.detach().numpy() if self.weights is not None else None,
            'circuit': str(self.circuit)
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        if params.get('weights') is not None:
            self.weights = torch.tensor(params['weights'], requires_grad=True)

class PennyLaneHybridWithAngleEncoding(PennyLaneQuantumModelWithAngleEncodingBase):
    """整合角度編碼的PennyLane混合模型"""
    
    def __init__(self, name: str = "PennyLane_Hybrid_AngleEncoded", n_qubits: int = 6, 
                 n_layers: int = 3, encoding_method: str = 'robust', 
                 feature_names: List[str] = None):
        super().__init__(name, n_qubits, n_layers, encoding_method, feature_names)
        
        # 初始化子模型
        self.qnn_model = PennyLaneQNNWithAngleEncoding(
            name="QNN_Component",
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        self.vqc_model = PennyLaneVQCWithAngleEncoding(
            name="VQC_Component",
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        logger.info("初始化PennyLane混合模型")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練混合模型"""
        logger.info(f"訓練PennyLane混合模型: {len(X)}樣本")
        
        # 訓練QNN組件
        self.qnn_model.fit(X, y)
        
        # 訓練VQC組件
        self.vqc_model.fit(X, y)
        
        self.is_trained = True
        logger.info("PennyLane混合模型訓練完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用QNN進行預測
        qnn_pred = self.qnn_model.predict(X)
        
        # 使用VQC進行預測
        vqc_pred = self.vqc_model.predict(X)
        
        # 簡單平均
        predictions = (qnn_pred + vqc_pred) / 2
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 使用QNN進行概率預測
        qnn_proba = self.qnn_model.predict_proba(X)
        
        # 使用VQC進行概率預測
        vqc_proba = self.vqc_model.predict_proba(X)
        
        # 簡單平均
        return (qnn_proba + vqc_proba) / 2
    
    def _get_model_params(self) -> Dict[str, Any]:
        """獲取模型參數"""
        return {
            'qnn_model': self.qnn_model,
            'vqc_model': self.vqc_model
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """設置模型參數"""
        self.qnn_model = params.get('qnn_model')
        self.vqc_model = params.get('vqc_model')

def create_pennylane_models_with_angle_encoding(feature_names: List[str], 
                                              n_qubits: int = 6, 
                                              n_layers: int = 3,
                                              encoding_method: str = 'robust') -> Dict[str, PennyLaneQuantumModelWithAngleEncodingBase]:
    """
    創建整合角度編碼的PennyLane量子模型
    
    Args:
        feature_names: 特徵名稱列表
        n_qubits: 量子比特數
        n_layers: 變分層數
        encoding_method: 角度編碼方法
    
    Returns:
        PennyLane量子模型字典
    """
    models = {}
    
    if not PENNYLANE_AVAILABLE:
        logger.warning("PennyLane不可用，無法創建量子模型")
        return models
    
    if not ANGLE_ENCODING_AVAILABLE:
        logger.warning("角度編碼不可用，使用標準特徵")
        return models
    
    try:
        # QNN
        models['pennylane_qnn'] = PennyLaneQNNWithAngleEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        # PennyLane VQC
        models['pennylane_vqc'] = PennyLaneVQCWithAngleEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        # PennyLane混合模型
        models['pennylane_hybrid'] = PennyLaneHybridWithAngleEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_method=encoding_method,
            feature_names=feature_names
        )
        
        logger.info(f"創建了 {len(models)} 個整合角度編碼的PennyLane量子模型")
        
    except Exception as e:
        logger.error(f"創建PennyLane量子模型時發生錯誤: {e}")
    
    return models

def main():
    """測試函數"""
    logger.info("測試整合角度編碼的PennyLane量子模型...")
    
    # 創建示例特徵名稱
    feature_names = [
        'price_sma_20_ratio', 'price_ma_ratio', 'vol_percentile',
        'vol_regime', 'bb_position', 'volume_ma_10'
    ]
    
    # 創建PennyLane量子模型
    models = create_pennylane_models_with_angle_encoding(
        feature_names=feature_names,
        n_qubits=6,
        n_layers=3,
        encoding_method='robust'
    )
    
    if models:
        logger.info(f"成功創建 {len(models)} 個PennyLane量子模型")
        for name, model in models.items():
            logger.info(f"- {name}: {model.name}")
    else:
        logger.warning("未能創建任何PennyLane量子模型")

if __name__ == "__main__":
    main()
