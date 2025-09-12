"""
Pure PennyLane Quantum-based strategies for Steer Intent.
No Qiskit dependency - only PennyLane.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from ..ml.pennylane_quantum_models import (
    SteerPennyLaneQNNIntentPredictor,
    SteerPennyLaneQNNPricePredictor,
    SteerPennyLaneQNNHybridPredictor
)

logger = logging.getLogger(__name__)

class PennyLaneQuantumBollingerStrategy:
    """Pure PennyLane Quantum-based Bollinger Bands strategy."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum Bollinger Bands strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.intent_predictor = SteerPennyLaneQNNIntentPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        self.is_trained = False
        self.last_rebalance_time = 0
        
        logger.info(f"Initialized PennyLane Quantum Bollinger strategy with {n_qubits} qubits, {n_layers} layers")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for intent prediction."""
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
        
        # Bollinger Bands features
        if len(data) >= self.bb_period:
            price = data['price']
            bb_middle = price.rolling(self.bb_period).mean()
            bb_std = price.rolling(self.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.bb_std)
            bb_lower = bb_middle - (bb_std * self.bb_std)
            
            features.extend([
                (price - bb_middle) / bb_middle,     # Price vs BB middle
                (price - bb_upper) / bb_upper,       # Price vs BB upper
                (price - bb_lower) / bb_lower,       # Price vs BB lower
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
        if len(data) > 10:
            # RSI-like indicator
            price = data['price']
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.fillna(50).values)
        
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
        """Train the quantum intent model."""
        logger.info("Training PennyLane quantum Bollinger intent model...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Create intent labels based on Bollinger Bands
        price = training_data['price']
        bb_middle = price.rolling(self.bb_period).mean()
        bb_std = price.rolling(self.bb_period).std()
        bb_upper = bb_middle + (bb_std * self.bb_std)
        bb_lower = bb_middle - (bb_std * self.bb_std)
        
        # Intent: 1 if price touches BB bands, 0 otherwise
        intent_labels = ((price >= bb_upper) | (price <= bb_lower)).astype(int)
        intent_labels = intent_labels.fillna(0).values
        
        # Train intent predictor
        self.intent_predictor.fit(X, intent_labels)
        
        self.is_trained = True
        logger.info("PennyLane quantum Bollinger intent model training completed")
    
    def should_rebalance(self, current_data: pd.DataFrame, current_time: int) -> bool:
        """Determine if rebalancing should occur using quantum Bollinger Bands."""
        if not self.is_trained:
            return False
        
        # Check minimum interval
        if current_time - self.last_rebalance_time < self.min_rebalance_interval:
            return False
        
        # Prepare features for current data
        X = self.prepare_features(current_data)
        
        # Get quantum intent prediction
        intent_prediction = self.intent_predictor.predict(X[-1:])[0]
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        
        # Decision based on quantum intent prediction
        should_rebalance = intent_prediction == 1 and intent_probability[1] > 0.6
        
        if should_rebalance:
            self.last_rebalance_time = current_time
        
        return should_rebalance
    
    def calculate_position_width(self, current_data: pd.DataFrame) -> float:
        """Calculate optimal position width using quantum Bollinger Bands."""
        if not self.is_trained:
            return 0.1  # Default width
        
        # Prepare features
        X = self.prepare_features(current_data)
        
        # Get quantum intent prediction probability
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        
        # Map probability to position width
        confidence = intent_probability[1]  # Probability of intent
        position_width = min(confidence * self.max_position_width, self.max_position_width)
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Bollinger',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Intent_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)_+_Bollinger_Bands'
        }

class PennyLaneQuantumKeltnerStrategy:
    """Pure PennyLane Quantum-based Keltner Channels strategy."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 kc_period: int = 20,
                 kc_multiplier: float = 2.0,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum Keltner Channels strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            kc_period: Keltner Channels period
            kc_multiplier: Keltner Channels multiplier
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kc_period = kc_period
        self.kc_multiplier = kc_multiplier
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.intent_predictor = SteerPennyLaneQNNIntentPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        self.is_trained = False
        self.last_rebalance_time = 0
        
        logger.info(f"Initialized PennyLane Quantum Keltner strategy with {n_qubits} qubits, {n_layers} layers")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for intent prediction."""
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
        
        # Keltner Channels features
        if len(data) >= self.kc_period:
            price = data['price']
            kc_middle = price.rolling(self.kc_period).mean()
            atr = price.rolling(self.kc_period).apply(lambda x: np.mean(np.abs(np.diff(x))))
            kc_upper = kc_middle + (atr * self.kc_multiplier)
            kc_lower = kc_middle - (atr * self.kc_multiplier)
            
            features.extend([
                (price - kc_middle) / kc_middle,     # Price vs KC middle
                (price - kc_upper) / kc_upper,       # Price vs KC upper
                (price - kc_lower) / kc_lower,       # Price vs KC lower
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
        if len(data) > 10:
            # MACD-like indicator
            price = data['price']
            ema_12 = price.ewm(span=12).mean()
            ema_26 = price.ewm(span=26).mean()
            macd = ema_12 - ema_26
            features.append(macd.fillna(0).values)
        
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
        """Train the quantum intent model."""
        logger.info("Training PennyLane quantum Keltner intent model...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Create intent labels based on Keltner Channels
        price = training_data['price']
        kc_middle = price.rolling(self.kc_period).mean()
        atr = price.rolling(self.kc_period).apply(lambda x: np.mean(np.abs(np.diff(x))))
        kc_upper = kc_middle + (atr * self.kc_multiplier)
        kc_lower = kc_middle - (atr * self.kc_multiplier)
        
        # Intent: 1 if price touches KC channels, 0 otherwise
        intent_labels = ((price >= kc_upper) | (price <= kc_lower)).astype(int)
        intent_labels = intent_labels.fillna(0).values
        
        # Train intent predictor
        self.intent_predictor.fit(X, intent_labels)
        
        self.is_trained = True
        logger.info("PennyLane quantum Keltner intent model training completed")
    
    def should_rebalance(self, current_data: pd.DataFrame, current_time: int) -> bool:
        """Determine if rebalancing should occur using quantum Keltner Channels."""
        if not self.is_trained:
            return False
        
        # Check minimum interval
        if current_time - self.last_rebalance_time < self.min_rebalance_interval:
            return False
        
        # Prepare features for current data
        X = self.prepare_features(current_data)
        
        # Get quantum intent prediction
        intent_prediction = self.intent_predictor.predict(X[-1:])[0]
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        
        # Decision based on quantum intent prediction
        should_rebalance = intent_prediction == 1 and intent_probability[1] > 0.6
        
        if should_rebalance:
            self.last_rebalance_time = current_time
        
        return should_rebalance
    
    def calculate_position_width(self, current_data: pd.DataFrame) -> float:
        """Calculate optimal position width using quantum Keltner Channels."""
        if not self.is_trained:
            return 0.1  # Default width
        
        # Prepare features
        X = self.prepare_features(current_data)
        
        # Get quantum intent prediction probability
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        
        # Map probability to position width
        confidence = intent_probability[1]  # Probability of intent
        position_width = min(confidence * self.max_position_width, self.max_position_width)
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Keltner',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'kc_period': self.kc_period,
            'kc_multiplier': self.kc_multiplier,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Intent_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)_+_Keltner_Channels'
        }

class PennyLaneQuantumHybridStrategy:
    """Pure PennyLane Quantum-based hybrid strategy combining intent and price prediction."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 intent_threshold: float = 0.6,
                 price_threshold: float = 0.05,
                 min_rebalance_interval: int = 1,
                 max_position_width: float = 0.2):
        """
        Initialize the PennyLane quantum hybrid strategy.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            intent_threshold: Threshold for intent-based rebalancing
            price_threshold: Threshold for price-based rebalancing
            min_rebalance_interval: Minimum intervals between rebalances
            max_position_width: Maximum position width
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.intent_threshold = intent_threshold
        self.price_threshold = price_threshold
        self.min_rebalance_interval = min_rebalance_interval
        self.max_position_width = max_position_width
        
        # Initialize quantum models
        self.intent_predictor = SteerPennyLaneQNNHybridPredictor(
            n_qubits=n_qubits,
            n_layers=n_layers,
            task_type='classification'
        )
        
        self.price_predictor = SteerPennyLaneQNNHybridPredictor(
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
        
        # Create intent labels (simplified)
        price = training_data['price'].values
        price_changes = np.diff(price, prepend=price[0])
        intent_labels = (np.abs(price_changes) > self.price_threshold).astype(int)
        
        # Create price targets
        price_targets = price_changes / price  # Price change rates
        
        # Train both models
        self.intent_predictor.fit(X, intent_labels)
        self.price_predictor.fit(X, price_targets)
        
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
        intent_prediction = self.intent_predictor.predict(X[-1:])[0]
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        predicted_price_change = self.price_predictor.predict(X[-1:])[0]
        
        # Hybrid decision logic
        intent_signal = intent_prediction == 1 and intent_probability[1] > self.intent_threshold
        price_signal = abs(predicted_price_change) > self.price_threshold
        
        # Combine signals
        should_rebalance = intent_signal or price_signal
        
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
        intent_probability = self.intent_predictor.predict_proba(X[-1:])[0]
        predicted_price_change = self.price_predictor.predict(X[-1:])[0]
        
        # Combine predictions for position width
        intent_confidence = intent_probability[1]  # Probability of intent
        price_confidence = min(abs(predicted_price_change) / self.price_threshold, 1.0)
        
        # Weighted combination
        position_width = (intent_confidence * 0.6 + price_confidence * 0.4) * self.max_position_width
        
        return max(position_width, 0.05)  # Minimum width
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': 'PennyLane_Quantum_Hybrid',
            'type': 'Quantum',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'intent_threshold': self.intent_threshold,
            'price_threshold': self.price_threshold,
            'min_rebalance_interval': self.min_rebalance_interval,
            'max_position_width': self.max_position_width,
            'is_trained': self.is_trained,
            'model_components': f'PennyLane_QNN_Intent_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)_+_PennyLane_QNN_Price_Predictor_({self.n_qubits}_qubits_{self.n_layers}_layers)_+_Hybrid_Strategy'
        }
