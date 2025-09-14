"""
Quantum Machine Learning models for Steer Intent strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import joblib
from datetime import datetime

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    import pennylane as qml
    from pennylane import numpy as pnp
    QUANTUM_AVAILABLE = True
except ImportError as e:
    QUANTUM_AVAILABLE = False
    logging.warning(f"Quantum computing libraries not available: {e}. Install qiskit and pennylane.")

logger = logging.getLogger(__name__)

class SteerQuantumModelBase(ABC):
    """Base class for Steer Intent quantum machine learning models."""
    
    def __init__(self, name: str, n_qubits: int = 4, n_layers: int = 2):
        self.name = name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.is_trained = False
        self.training_history = []
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the quantum model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'name': self.name,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'model_params': self._get_model_params()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved quantum model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.name = model_data['name']
        self.n_qubits = model_data['n_qubits']
        self.n_layers = model_data['n_layers']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        
        self._set_model_params(model_data['model_params'])
        logger.info(f"Loaded quantum model from {filepath}")
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        pass
    
    @abstractmethod
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        pass

class SteerQNNIntentPredictor(SteerQuantumModelBase):
    """Quantum Neural Network for Steer Intent prediction using Qiskit."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        super().__init__("Steer_QNN_Intent_Predictor", n_qubits, n_layers)
        self.feature_dim = feature_dim
        self.vqc = None
        self.sampler = None
        self.optimizer = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available. Please install qiskit and qiskit-machine-learning.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the VQC model for intent prediction."""
        # Create feature map optimized for financial data
        from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
        feature_map = PauliFeatureMap(feature_dimension=self.feature_dim, reps=2, paulis=['Z', 'X'])
        
        # Create variational form
        from qiskit.circuit.library import TwoLocal, EfficientSU2
        ansatz = EfficientSU2(self.feature_dim, reps=self.n_layers)
        
        # Create VQC
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=200),
            sampler=Sampler()
        )
        
        logger.info(f"Initialized Steer QNN with {self.n_qubits} qubits, {self.n_layers} layers")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the QNN model for intent prediction."""
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        logger.info(f"Training Steer QNN on {len(X)} samples with {X.shape[1]} features")
        
        # Ensure features are normalized
        X_normalized = self._normalize_features(X)
        
        # Train the model
        self.vqc.fit(X_normalized, y)
        self.is_trained = True
        
        logger.info("Steer QNN training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        return self.vqc.predict(X_normalized)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        return self.vqc.predict_proba(X_normalized)
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range for quantum circuits."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        
        return (X - X_min) / X_range
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'feature_dim': self.feature_dim,
            'vqc_params': self.vqc.parameters if self.vqc else None
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)

class SteerQSVMPricePredictor(SteerQuantumModelBase):
    """Quantum Support Vector Machine for Steer Intent price prediction using Qiskit."""
    
    def __init__(self, n_qubits: int = 4, feature_dim: int = 8):
        super().__init__("Steer_QSVM_Price_Predictor", n_qubits, 0)  # SVM doesn't use layers
        self.feature_dim = feature_dim
        self.qsvc = None
        self.quantum_kernel = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available. Please install qiskit and qiskit-machine-learning.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the QSVM model for price prediction."""
        # Create feature map optimized for price data
        from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
        feature_map = ZZFeatureMap(feature_dimension=self.feature_dim, reps=3)
        
        # Create quantum kernel
        self.quantum_kernel = QuantumKernel(feature_map=feature_map)
        
        # Create QSVC
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)
        
        logger.info(f"Initialized Steer QSVM with {self.n_qubits} qubits")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the QSVM model for price prediction."""
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        logger.info(f"Training Steer QSVM on {len(X)} samples with {X.shape[1]} features")
        
        # Ensure features are normalized
        X_normalized = self._normalize_features(X)
        
        # Train the model
        self.qsvc.fit(X_normalized, y)
        self.is_trained = True
        
        logger.info("Steer QSVM training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        return self.qsvc.predict(X_normalized)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (QSVM doesn't provide probabilities directly)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.predict(X)
        # Convert predictions to probabilities (simplified approach)
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions] = 1.0
        return proba
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range for quantum circuits."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        
        return (X - X_min) / X_range
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'feature_dim': self.feature_dim,
            'support_vectors': self.qsvc.support_vectors_ if hasattr(self.qsvc, 'support_vectors_') else None
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)

class SteerPennyLaneQNNPredictor(SteerQuantumModelBase):
    """PennyLane-based Quantum Neural Network for Steer Intent advanced quantum machine learning."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        super().__init__("Steer_PennyLane_QNN_Predictor", n_qubits, n_layers)
        self.feature_dim = feature_dim
        self.device = None
        self.circuit = None
        self.weights = None
        self.optimizer = None
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("PennyLane not available. Please install pennylane.")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the QNN model for Steer Intent."""
        # Create quantum device
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        
        # Initialize weights
        self.weights = pnp.random.normal(0, 0.1, (self.n_layers, self.n_qubits, 3))
        
        # Create optimizer
        self.optimizer = qml.AdamOptimizer(stepsize=0.01)
        
        logger.info(f"Initialized Steer QNN with {self.n_qubits} qubits, {self.n_layers} layers")
    
    def _quantum_circuit(self, features, weights):
        """Define the quantum circuit optimized for Steer Intent."""
        # Encode features with rotation gates
        for i in range(min(len(features), self.n_qubits)):
            qml.RY(features[i], wires=i)
            qml.RZ(features[i] * 0.5, wires=i)  # Additional encoding
        
        # Variational layers with entangling gates
        for layer in range(self.n_layers):
            # Single qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            # Entangling gates (ring topology)
            for qubit in range(self.n_qubits):
                next_qubit = (qubit + 1) % self.n_qubits
                qml.CNOT(wires=[qubit, next_qubit])
        
        # Measure expectation value
        return qml.expval(qml.PauliZ(0))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the QNN model for Steer Intent."""
        if not QUANTUM_AVAILABLE:
            raise ImportError("PennyLane not available")
        
        logger.info(f"Training Steer QNN on {len(X)} samples with {X.shape[1]} features")
        
        # Create QNode
        self.circuit = qml.QNode(self._quantum_circuit, self.device)
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Training loop with early stopping
        n_epochs = 100
        best_cost = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            cost = 0
            for i in range(len(X_normalized)):
                prediction = self.circuit(X_normalized[i], self.weights)
                cost += (prediction - y[i]) ** 2
            
            cost /= len(X_normalized)
            self.weights = self.optimizer.step(lambda w: cost, self.weights)
            
            # Early stopping
            if cost < best_cost:
                best_cost = cost
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        self.is_trained = True
        logger.info("Steer QNN training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        predictions = []
        
        for i in range(len(X_normalized)):
            pred = self.circuit(X_normalized[i], self.weights)
            predictions.append(1 if pred > 0 else 0)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        probabilities = []
        
        for i in range(len(X_normalized)):
            pred = self.circuit(X_normalized[i], self.weights)
            # Convert to probability with sigmoid-like function
            prob_1 = 1 / (1 + np.exp(-pred))  # Sigmoid
            prob_0 = 1 - prob_1
            probabilities.append([prob_0, prob_1])
        
        return np.array(probabilities)
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [0, π] range for quantum circuits."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        
        # Normalize to [0, 1] then scale to [0, π]
        X_normalized = (X - X_min) / X_range
        return X_normalized * np.pi
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'feature_dim': self.feature_dim,
            'weights': self.weights.tolist() if self.weights is not None else None
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)
        weights_data = params.get('weights')
        if weights_data is not None:
            self.weights = pnp.array(weights_data)

def create_steer_quantum_model(model_type: str, **kwargs) -> SteerQuantumModelBase:
    """Factory function to create Steer Intent quantum models."""
    if not QUANTUM_AVAILABLE:
        raise ImportError("Quantum computing libraries not available")
    
    if model_type == "qnn_intent":
        return SteerQNNIntentPredictor(**kwargs)
    elif model_type == "qsvm_price":
        return SteerQSVMPricePredictor(**kwargs)
    elif model_type == "pennylane_qnn":
        return SteerPennyLaneQNNPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown quantum model type: {model_type}")

def save_steer_quantum_model(model: SteerQuantumModelBase, filepath: str) -> None:
    """Save a Steer Intent quantum model to file."""
    model.save_model(filepath)

def load_steer_quantum_model(filepath: str) -> SteerQuantumModelBase:
    """Load a Steer Intent quantum model from file."""
    model_data = joblib.load(filepath)
    model_type = model_data['name'].lower()
    
    if 'qnn_intent' in model_type:
        model = SteerQNNIntentPredictor()
    elif 'qsvm_price' in model_type:
        model = SteerQSVMPricePredictor()
    elif 'pennylane_qnn' in model_type:
        model = SteerPennyLaneQNNPredictor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_model(filepath)
    return model
