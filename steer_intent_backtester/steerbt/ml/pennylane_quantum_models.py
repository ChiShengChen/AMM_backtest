"""
Pure PennyLane Quantum Machine Learning models for Steer Intent strategies.
No Qiskit dependency - only PennyLane.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import joblib
from datetime import datetime
import torch
import torch.nn as nn

# PennyLane imports
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
    PENNYLANE_AVAILABLE = True
except ImportError as e:
    PENNYLANE_AVAILABLE = False
    logging.warning(f"PennyLane not available: {e}. Install pennylane.")

logger = logging.getLogger(__name__)

class SteerPennyLaneQuantumModelBase(ABC):
    """Base class for Steer Intent PennyLane quantum machine learning models."""
    
    def __init__(self, name: str, n_qubits: int = 4, n_layers: int = 2):
        self.name = name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.is_trained = False
        self.training_history = []
        self.device = None
        self.circuit = None
        self.weights = None
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available. Please install pennylane.")
        
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the quantum device."""
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        logger.info(f"Initialized PennyLane device with {self.n_qubits} qubits")
    
    @abstractmethod
    def _create_circuit(self, inputs, weights):
        """Create the quantum circuit."""
        pass
    
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
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [0, π] range for quantum circuits."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        
        X_normalized = (X - X_min) / X_range * np.pi
        return X_normalized
    
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
            'weights': self.weights,
            'model_params': self._get_model_params()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved Steer PennyLane quantum model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.name = model_data['name']
        self.n_qubits = model_data['n_qubits']
        self.n_layers = model_data['n_layers']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.weights = model_data['weights']
        
        self._set_model_params(model_data['model_params'])
        logger.info(f"Loaded Steer PennyLane quantum model from {filepath}")
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        pass
    
    @abstractmethod
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        pass

class SteerPennyLaneQNNIntentPredictor(SteerPennyLaneQuantumModelBase):
    """PennyLane Quantum Neural Network for intent prediction in Steer Intent strategies."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        super().__init__("Steer_PennyLane_QNN_Intent_Predictor", n_qubits, n_layers)
        self.feature_dim = feature_dim
        self.optimizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PennyLane QNN model for intent prediction."""
        # Initialize weights
        self.weights = np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits))
        
        # Create the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            return self._create_circuit(inputs, weights)
        
        self.circuit = circuit
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(stepsize=0.01)
        
        logger.info(f"Initialized Steer PennyLane QNN Intent Predictor with {self.n_qubits} qubits, {self.n_layers} layers")
    
    def _create_circuit(self, inputs, weights):
        """Create the quantum circuit for intent prediction."""
        # Feature encoding for intent prediction
        for i in range(min(len(inputs), self.n_qubits)):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i] * 0.5, wires=i)
        
        # Variational layers optimized for intent prediction
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i] * 0.5, wires=i)
            
            # Entangling gates - ring topology
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Complete the ring
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # Measurement for intent classification
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _cost_function(self, weights, X, y):
        """Cost function for intent prediction."""
        predictions = []
        for x in X:
            pred = self.circuit(x, weights)
            # Convert to intent prediction (0: no rebalance, 1: rebalance)
            prediction = 1 if np.mean(pred) > 0 else 0
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y)
        return 1 - accuracy  # Minimize (1 - accuracy)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the PennyLane QNN model for intent prediction."""
        logger.info(f"Training Steer PennyLane QNN Intent Predictor on {len(X)} samples with {X.shape[1]} features")
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Ensure binary labels for intent
        y_binary = np.array([1 if label > 0.5 else 0 for label in y])
        
        # Training loop
        epochs = 50
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        for epoch in range(epochs):
            # Compute cost and gradients
            cost = self._cost_function(self.weights, X_normalized, y_binary)
            
            # Update weights
            self.weights = self.optimizer.step(
                lambda w: self._cost_function(w, X_normalized, y_binary),
                self.weights
            )
            
            # Calculate accuracy
            predictions = []
            for x in X_normalized:
                pred = self.circuit(x, self.weights)
                prediction = 1 if np.mean(pred) > 0 else 0
                predictions.append(prediction)
            
            accuracy = np.mean(np.array(predictions) == y_binary)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = self.weights.copy()
            
            self.training_history.append({
                'epoch': epoch,
                'cost': cost,
                'accuracy': accuracy
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Cost = {cost:.4f}, Accuracy = {accuracy:.4f}")
        
        # Use best weights
        self.weights = best_weights
        self.is_trained = True
        
        logger.info(f"Steer PennyLane QNN Intent training completed. Best accuracy: {best_accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make intent predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        predictions = []
        
        for x in X_normalized:
            pred = self.circuit(x, self.weights)
            prediction = 1 if np.mean(pred) > 0 else 0
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict intent probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        probabilities = []
        
        for x in X_normalized:
            pred = self.circuit(x, self.weights)
            # Convert expectation values to probabilities
            prob_rebalance = (np.mean(pred) + 1) / 2  # Map [-1, 1] to [0, 1]
            prob_no_rebalance = 1 - prob_rebalance
            probabilities.append([prob_no_rebalance, prob_rebalance])
        
        return np.array(probabilities)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'feature_dim': self.feature_dim,
            'weights': self.weights,
            'optimizer_stepsize': self.optimizer.stepsize if hasattr(self.optimizer, 'stepsize') else 0.01
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)
        self.weights = params.get('weights', np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits)))
        stepsize = params.get('optimizer_stepsize', 0.01)
        self.optimizer = AdamOptimizer(stepsize=stepsize)

class SteerPennyLaneQNNPricePredictor(SteerPennyLaneQuantumModelBase):
    """PennyLane Quantum Neural Network for price prediction in Steer Intent strategies."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        super().__init__("Steer_PennyLane_QNN_Price_Predictor", n_qubits, n_layers)
        self.feature_dim = feature_dim
        self.optimizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PennyLane QNN model for price prediction."""
        # Initialize weights
        self.weights = np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits))
        
        # Create the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            return self._create_circuit(inputs, weights)
        
        self.circuit = circuit
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(stepsize=0.01)
        
        logger.info(f"Initialized Steer PennyLane QNN Price Predictor with {self.n_qubits} qubits, {self.n_layers} layers")
    
    def _create_circuit(self, inputs, weights):
        """Create the quantum circuit for price prediction."""
        # Feature encoding for price prediction
        for i in range(min(len(inputs), self.n_qubits)):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i] * 0.5, wires=i)
            qml.RX(inputs[i] * 0.3, wires=i)
        
        # Variational layers optimized for price prediction
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i] * 0.4, wires=i)
                qml.RX(weights[layer, i] * 0.3, wires=i)
            
            # Complex entangling pattern for price dynamics
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.CZ(wires=[i, i + 1])
            
            # Additional entangling for better price modeling
            if self.n_qubits > 2:
                qml.CNOT(wires=[0, self.n_qubits - 1])
                qml.CZ(wires=[1, self.n_qubits - 1])
        
        # Measurement for price regression
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _cost_function(self, weights, X, y):
        """Cost function for price prediction (regression)."""
        predictions = []
        for x in X:
            pred = self.circuit(x, weights)
            # Convert to price prediction
            prediction = np.mean(pred) * 0.1 + 0.05  # Scale to reasonable price change range
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the PennyLane QNN model for price prediction."""
        logger.info(f"Training Steer PennyLane QNN Price Predictor on {len(X)} samples")
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Normalize targets to [0, 1] range
        y_min, y_max = np.min(y), np.max(y)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1
        y_normalized = (y - y_min) / y_range
        
        # Training loop
        epochs = 50
        best_mse = float('inf')
        best_weights = self.weights.copy()
        
        for epoch in range(epochs):
            # Compute cost and gradients
            cost = self._cost_function(self.weights, X_normalized, y_normalized)
            
            # Update weights
            self.weights = self.optimizer.step(
                lambda w: self._cost_function(w, X_normalized, y_normalized),
                self.weights
            )
            
            if cost < best_mse:
                best_mse = cost
                best_weights = self.weights.copy()
            
            self.training_history.append({
                'epoch': epoch,
                'cost': cost,
                'mse': cost
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: MSE = {cost:.6f}")
        
        # Use best weights
        self.weights = best_weights
        self.is_trained = True
        
        # Store normalization parameters
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_range
        
        logger.info(f"Steer PennyLane QNN Price training completed. Best MSE: {best_mse:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make price predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        predictions = []
        
        for x in X_normalized:
            pred = self.circuit(x, self.weights)
            # Convert to price prediction and denormalize
            prediction = np.mean(pred) * 0.1 + 0.05
            prediction = prediction * self.y_range + self.y_min
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict price probabilities (for compatibility)."""
        predictions = self.predict(X)
        # Convert to probability-like format
        probabilities = np.column_stack([1 - predictions, predictions])
        return probabilities
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'feature_dim': self.feature_dim,
            'weights': self.weights,
            'y_min': getattr(self, 'y_min', 0),
            'y_max': getattr(self, 'y_max', 1),
            'y_range': getattr(self, 'y_range', 1),
            'optimizer_stepsize': self.optimizer.stepsize if hasattr(self.optimizer, 'stepsize') else 0.01
        }
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)
        self.weights = params.get('weights', np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits)))
        self.y_min = params.get('y_min', 0)
        self.y_max = params.get('y_max', 1)
        self.y_range = params.get('y_range', 1)
        stepsize = params.get('optimizer_stepsize', 0.01)
        self.optimizer = AdamOptimizer(stepsize=stepsize)

class SteerPennyLaneQNNHybridPredictor(SteerPennyLaneQuantumModelBase):
    """PennyLane Quantum Neural Network for hybrid prediction (intent + price) in Steer Intent strategies."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8, task_type: str = 'classification'):
        super().__init__("Steer_PennyLane_QNN_Hybrid_Predictor", n_qubits, n_layers)
        self.feature_dim = feature_dim
        self.task_type = task_type  # 'classification' or 'regression'
        self.optimizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PennyLane QNN hybrid model."""
        # Initialize weights
        self.weights = np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits))
        
        # Create the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            return self._create_circuit(inputs, weights)
        
        self.circuit = circuit
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(stepsize=0.01)
        
        logger.info(f"Initialized Steer PennyLane QNN Hybrid Predictor ({self.task_type}) with {self.n_qubits} qubits, {self.n_layers} layers")
    
    def _create_circuit(self, inputs, weights):
        """Create the quantum circuit for hybrid prediction."""
        # Advanced feature encoding for hybrid prediction
        for i in range(min(len(inputs), self.n_qubits)):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i] * 0.5, wires=i)
            qml.RX(inputs[i] * 0.3, wires=i)
        
        # Variational layers with maximum complexity
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i] * 0.4, wires=i)
                qml.RX(weights[layer, i] * 0.3, wires=i)
            
            # Complex entangling pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.CZ(wires=[i, i + 1])
            
            # Additional entangling
            if self.n_qubits > 2:
                qml.CNOT(wires=[0, self.n_qubits - 1])
                qml.CZ(wires=[1, self.n_qubits - 1])
        
        # Multiple measurements for richer output
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _cost_function(self, weights, X, y):
        """Cost function for hybrid prediction."""
        predictions = []
        for x in X:
            pred = self.circuit(x, weights)
            
            if self.task_type == 'classification':
                # Intent classification
                prediction = 1 if np.mean(pred) > 0 else 0
            else:
                # Price regression
                prediction = np.mean(pred) * 0.1 + 0.05
            
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        
        if self.task_type == 'classification':
            accuracy = np.mean(predictions == y)
            return 1 - accuracy
        else:
            mse = np.mean((predictions - y) ** 2)
            return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the PennyLane QNN hybrid model."""
        logger.info(f"Training Steer PennyLane QNN Hybrid Predictor ({self.task_type}) on {len(X)} samples")
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        if self.task_type == 'classification':
            # Ensure binary labels for intent
            y_processed = np.array([1 if label > 0.5 else 0 for label in y])
        else:
            # Normalize targets for price regression
            y_min, y_max = np.min(y), np.max(y)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1
            y_processed = (y - y_min) / y_range
            self.y_min = y_min
            self.y_max = y_max
            self.y_range = y_range
        
        # Training loop
        epochs = 50
        best_metric = float('inf')
        best_weights = self.weights.copy()
        
        for epoch in range(epochs):
            # Compute cost and gradients
            cost = self._cost_function(self.weights, X_normalized, y_processed)
            
            # Update weights
            self.weights = self.optimizer.step(
                lambda w: self._cost_function(w, X_normalized, y_processed),
                self.weights
            )
            
            if cost < best_metric:
                best_metric = cost
                best_weights = self.weights.copy()
            
            self.training_history.append({
                'epoch': epoch,
                'cost': cost
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        # Use best weights
        self.weights = best_weights
        self.is_trained = True
        
        logger.info(f"Steer PennyLane QNN Hybrid training completed. Best cost: {best_metric:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        predictions = []
        
        for x in X_normalized:
            pred = self.circuit(x, self.weights)
            
            if self.task_type == 'classification':
                prediction = 1 if np.mean(pred) > 0 else 0
            else:
                # Price regression - denormalize
                prediction = np.mean(pred) * 0.1 + 0.05
                prediction = prediction * self.y_range + self.y_min
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X)
        probabilities = []
        
        for x in X_normalized:
            pred = self.circuit(x, self.weights)
            
            if self.task_type == 'classification':
                prob_1 = (np.mean(pred) + 1) / 2
                prob_0 = 1 - prob_1
                probabilities.append([prob_0, prob_1])
            else:
                # For price regression, return prediction as probability
                prediction = np.mean(pred) * 0.1 + 0.05
                prediction = prediction * self.y_range + self.y_min
                probabilities.append([1 - prediction, prediction])
        
        return np.array(probabilities)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        params = {
            'feature_dim': self.feature_dim,
            'task_type': self.task_type,
            'weights': self.weights,
            'optimizer_stepsize': self.optimizer.stepsize if hasattr(self.optimizer, 'stepsize') else 0.01
        }
        
        if self.task_type == 'regression':
            params.update({
                'y_min': getattr(self, 'y_min', 0),
                'y_max': getattr(self, 'y_max', 1),
                'y_range': getattr(self, 'y_range', 1)
            })
        
        return params
    
    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.feature_dim = params.get('feature_dim', 8)
        self.task_type = params.get('task_type', 'classification')
        self.weights = params.get('weights', np.random.uniform(0, 2*np.pi, (self.n_layers, self.n_qubits)))
        
        if self.task_type == 'regression':
            self.y_min = params.get('y_min', 0)
            self.y_max = params.get('y_max', 1)
            self.y_range = params.get('y_range', 1)
        
        stepsize = params.get('optimizer_stepsize', 0.01)
        self.optimizer = AdamOptimizer(stepsize=stepsize)
