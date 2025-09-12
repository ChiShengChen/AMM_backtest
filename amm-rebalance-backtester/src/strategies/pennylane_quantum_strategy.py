"""
Pure PennyLane Quantum-based strategies for AMM rebalancing.
No Qiskit dependency - only PennyLane.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from ..ml.pennylane_quantum_models import (
    PennyLaneQNNRebalancePredictor,
    PennyLaneQNNVolatilityPredictor,
    PennyLaneQNNHybridPredictor
)

logger = logging.getLogger(__name__)

class PennyLaneQuantumBasedStrategy:
    """Pure PennyLane Quantum-based rebalancing strategy."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 rebalance_threshold: float = 0.1,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum-based strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            rebalance_threshold: Threshold for rebalancing decisions
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rebalance_threshold = rebalance_threshold
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.rebalance_predictor = PennyLaneQNNRebalancePredictor(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        self.is_trained = False
        self.last_rebalance_time = 0
        self.position_history = []
        
        logger.info(f"Initialized PennyLane Quantum-based strategy with {n_qubits} qubits, {n_layers} layers")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for quantum model."""
        features = []
        
        # Price-based features
        if 'price' in data.columns:
            price = data['price'].values
            features.extend([
                np.diff(price, prepend=price[0]),  # Price change
                np.roll(price, 1) / price - 1,     # Price ratio
            ])
        
        # Volume-based features
        if 'volume' in data.columns:
            volume = data['volume'].values
            features.extend([
                np.diff(volume, prepend=volume[0]),  # Volume change
                volume / np.mean(volume),            # Volume ratio
            ])
        
        # Technical indicators
        if len(data) > 20:
            # Moving averages
            ma_5 = data['price'].rolling(5).mean().fillna(data['price'])
            ma_20 = data['price'].rolling(20).mean().fillna(data['price'])
            features.extend([
                (data['price'] - ma_5) / ma_5,      # Price vs MA5
                (data['price'] - ma_20) / ma_20,    # Price vs MA20
            ])
        
        # Volatility features
        if len(data) > 10:
            returns = data['price'].pct_change().fillna(0)
            volatility = returns.rolling(10).std().fillna(0)
            features.append(volatility.values)
        
        # Convert to numpy array and handle missing values
        features_array = np.array(features).T
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure we have enough features
        if features_array.shape[1] < 8:
            # Pad with zeros if needed
            padding = np.zeros((features_array.shape[0], 8 - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        elif features_array.shape[1] > 8:
            # Truncate if too many features
            features_array = features_array[:, :8]
        
        return features_array
    
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the quantum models."""
        logger.info("Training PennyLane quantum models...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Create rebalance labels (simplified)
        price = training_data['price'].values
        price_changes = np.diff(price, prepend=price[0])
        rebalance_labels = (np.abs(price_changes) > self.rebalance_threshold).astype(int)
        
        # Train rebalance predictor
        self.rebalance_predictor.fit(X, rebalance_labels)
        
        self.is_trained = True
        logger.info("PennyLane quantum models training completed")
    
    def should_rebalance(self, current_data: pd.DataFrame, current_time: int) -> bool:
        """Determine if rebalancing should occur."""
        if not self.is_trained:
            return False
        
        # Check minimum interval
        if current_time - self.last_rebalance_time < self.min_rebalance_interval:
            return False
        
        # Prepare features for current data
        X = self.prepare_features(current_data)
        
        # Get quantum prediction
        prediction = self.rebalance_predictor.predict(X[-1:])[0]
        probability = self.rebalance_predictor.predict_proba(X[-1:])[0]
        
        # Decision based on quantum prediction
        should_rebalance = prediction == 1 and probability[1] > 0.6
        
        if should_rebalance:
            self.last_rebalance_time = current_time
        
        return should_rebalance
    
    def calculate_position_width(self, current_data: pd.DataFrame) -> float:
        """Calculate optimal position width using quantum model."""
        if not self.is_trained:
            return 0.1  # Default width
        
        # Prepare features
        X = self.prepare_features(current_data)
        
        # Get quantum prediction probability
        probability = self.rebalance_predictor.predict_proba(X[-1:])[0]
        
        # Map probability to position width
        confidence = probability[1]  # Probability of rebalance
        position_width = min(confidence * self.max_position_width, self.max_position_width)
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Based',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'rebalance_threshold': self.rebalance_threshold,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Rebalance_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)'
        }

class PennyLaneQuantumVolatilityStrategy:
    """Pure PennyLane Quantum-based volatility strategy."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 volatility_threshold: float = 0.05,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum volatility strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            volatility_threshold: Threshold for volatility-based rebalancing
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.volatility_threshold = volatility_threshold
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.volatility_predictor = PennyLaneQNNVolatilityPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        self.is_trained = False
        self.last_rebalance_time = 0
        
        logger.info(f"Initialized PennyLane Quantum Volatility strategy with {n_qubits} qubits, {n_layers} layers")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for volatility prediction."""
        features = []
        
        # Price-based features
        if 'price' in data.columns:
            price = data['price'].values
            returns = np.diff(price, prepend=price[0]) / price
            features.extend([
                returns,                              # Returns
                np.roll(returns, 1),                 # Lagged returns
                np.roll(returns, 2),                 # Double lagged returns
            ])
        
        # Volume-based features
        if 'volume' in data.columns:
            volume = data['volume'].values
            volume_returns = np.diff(volume, prepend=volume[0]) / volume
            features.extend([
                volume_returns,                       # Volume returns
                volume / np.mean(volume),            # Volume ratio
            ])
        
        # Technical indicators
        if len(data) > 20:
            # RSI-like indicator
            price = data['price']
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.fillna(50).values)
        
        # Volatility features
        if len(data) > 10:
            returns = data['price'].pct_change().fillna(0)
            volatility = returns.rolling(10).std().fillna(0)
            features.extend([
                volatility.values,                    # Current volatility
                volatility.shift(1).fillna(0).values, # Lagged volatility
            ])
        
        # Convert to numpy array and handle missing values
        features_array = np.array(features).T
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure we have enough features
        if features_array.shape[1] < 8:
            # Pad with zeros if needed
            padding = np.zeros((features_array.shape[0], 8 - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        elif features_array.shape[1] > 8:
            # Truncate if too many features
            features_array = features_array[:, :8]
        
        return features_array
    
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the quantum volatility model."""
        logger.info("Training PennyLane quantum volatility model...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Create volatility targets
        price = training_data['price'].values
        returns = np.diff(price, prepend=price[0]) / price
        volatility_targets = np.abs(returns)  # Target volatility
        
        # Train volatility predictor
        self.volatility_predictor.fit(X, volatility_targets)
        
        self.is_trained = True
        logger.info("PennyLane quantum volatility model training completed")
    
    def should_rebalance(self, current_data: pd.DataFrame, current_time: int) -> bool:
        """Determine if rebalancing should occur based on volatility."""
        if not self.is_trained:
            return False
        
        # Check minimum interval
        if current_time - self.last_rebalance_time < self.min_rebalance_interval:
            return False
        
        # Prepare features for current data
        X = self.prepare_features(current_data)
        
        # Get quantum volatility prediction
        predicted_volatility = self.volatility_predictor.predict(X[-1:])[0]
        
        # Decision based on predicted volatility
        should_rebalance = predicted_volatility > self.volatility_threshold
        
        if should_rebalance:
            self.last_rebalance_time = current_time
        
        return should_rebalance
    
    def calculate_position_width(self, current_data: pd.DataFrame) -> float:
        """Calculate optimal position width based on predicted volatility."""
        if not self.is_trained:
            return 0.1  # Default width
        
        # Prepare features
        X = self.prepare_features(current_data)
        
        # Get quantum volatility prediction
        predicted_volatility = self.volatility_predictor.predict(X[-1:])[0]
        
        # Map volatility to position width (inverse relationship)
        volatility_factor = max(0.1, 1 - predicted_volatility)
        position_width = volatility_factor * self.max_position_width
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Volatility',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'volatility_threshold': self.volatility_threshold,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Volatility_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)'
        }

class PennyLaneQuantumHybridStrategy:
    """Pure PennyLane Quantum-based hybrid strategy combining rebalance and volatility prediction."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 rebalance_threshold: float = 0.1,
                 volatility_threshold: float = 0.05,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum hybrid strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            rebalance_threshold: Threshold for rebalancing decisions
            volatility_threshold: Threshold for volatility-based rebalancing
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rebalance_threshold = rebalance_threshold
        self.volatility_threshold = volatility_threshold
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.rebalance_predictor = PennyLaneQNNHybridPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers,
            task_type='classification'
        )
        
        self.volatility_predictor = PennyLaneQNNHybridPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers,
            task_type='regression'
        )
        
        self.is_trained = False
        self.last_rebalance_time = 0
        
        logger.info(f"Initialized PennyLane Quantum Hybrid strategy with {n_qubits} qubits, {n_layers} layers")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for hybrid prediction."""
        features = []
        
        # Price-based features
        if 'price' in data.columns:
            price = data['price'].values
            returns = np.diff(price, prepend=price[0]) / price
            features.extend([
                returns,                              # Returns
                np.roll(returns, 1),                 # Lagged returns
                np.roll(returns, 2),                 # Double lagged returns
                np.roll(price, 1) / price - 1,       # Price ratio
            ])
        
        # Volume-based features
        if 'volume' in data.columns:
            volume = data['volume'].values
            volume_returns = np.diff(volume, prepend=volume[0]) / volume
            features.extend([
                volume_returns,                       # Volume returns
                volume / np.mean(volume),            # Volume ratio
            ])
        
        # Technical indicators
        if len(data) > 20:
            # Moving averages
            ma_5 = data['price'].rolling(5).mean().fillna(data['price'])
            ma_20 = data['price'].rolling(20).mean().fillna(data['price'])
            features.extend([
                (data['price'] - ma_5) / ma_5,      # Price vs MA5
                (data['price'] - ma_20) / ma_20,    # Price vs MA20
            ])
        
        # Convert to numpy array and handle missing values
        features_array = np.array(features).T
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure we have enough features
        if features_array.shape[1] < 8:
            # Pad with zeros if needed
            padding = np.zeros((features_array.shape[0], 8 - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        elif features_array.shape[1] > 8:
            # Truncate if too many features
            features_array = features_array[:, :8]
        
        return features_array
    
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the quantum hybrid models."""
        logger.info("Training PennyLane quantum hybrid models...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Create rebalance labels
        price = training_data['price'].values
        price_changes = np.diff(price, prepend=price[0])
        rebalance_labels = (np.abs(price_changes) > self.rebalance_threshold).astype(int)
        
        # Create volatility targets
        returns = price_changes / price
        volatility_targets = np.abs(returns)
        
        # Train both models
        self.rebalance_predictor.fit(X, rebalance_labels)
        self.volatility_predictor.fit(X, volatility_targets)
        
        self.is_trained = True
        logger.info("PennyLane quantum hybrid models training completed")
    
    def should_rebalance(self, current_data: pd.DataFrame, current_time: int) -> bool:
        """Determine if rebalancing should occur using hybrid approach."""
        if not self.is_trained:
            return False
        
        # Check minimum interval
        if current_time - self.last_rebalance_time < self.min_rebalance_interval:
            return False
        
        # Prepare features for current data
        X = self.prepare_features(current_data)
        
        # Get quantum predictions
        rebalance_prediction = self.rebalance_predictor.predict(X[-1:])[0]
        rebalance_probability = self.rebalance_predictor.predict_proba(X[-1:])[0]
        predicted_volatility = self.volatility_predictor.predict(X[-1:])[0]
        
        # Hybrid decision logic
        rebalance_signal = rebalance_prediction == 1 and rebalance_probability[1] > 0.6
        volatility_signal = predicted_volatility > self.volatility_threshold
        
        # Combine signals
        should_rebalance = rebalance_signal or volatility_signal
        
        if should_rebalance:
            self.last_rebalance_time = current_time
        
        return should_rebalance
    
    def calculate_position_width(self, current_data: pd.DataFrame) -> float:
        """Calculate optimal position width using hybrid approach."""
        if not self.is_trained:
            return 0.1  # Default width
        
        # Prepare features
        X = self.prepare_features(current_data)
        
        # Get quantum predictions
        rebalance_probability = self.rebalance_predictor.predict_proba(X[-1:])[0]
        predicted_volatility = self.volatility_predictor.predict(X[-1:])[0]
        
        # Combine predictions for position width
        confidence = rebalance_probability[1]  # Probability of rebalance
        volatility_factor = max(0.1, 1 - predicted_volatility)
        
        # Weighted combination
        position_width = (confidence * 0.6 + volatility_factor * 0.4) * self.max_position_width
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Hybrid',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'rebalance_threshold': self.rebalance_threshold,
            'volatility_threshold': self.volatility_threshold,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Hybrid_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)_+_PennyLane_QNN_Volatility_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)'
        }
