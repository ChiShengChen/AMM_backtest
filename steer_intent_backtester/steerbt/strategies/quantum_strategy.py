"""
Quantum machine learning-based Steer Intent strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

from .base import BaseStrategy
from ..ml import (
    SteerFeatureEngineer, 
    QUANTUM_AVAILABLE,
    SteerQNNIntentPredictor,
    SteerQSVMPricePredictor,
    SteerPennyLaneQNNPredictor,
    create_steer_quantum_model
)

logger = logging.getLogger(__name__)

class QuantumBollingerStrategy(BaseStrategy):
    """Quantum Neural Network-enhanced Bollinger Bands strategy for Steer Intent."""
    
    def __init__(self, 
                 quantum_model_type: str = "qnn_intent",
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 feature_dim: int = 8,
                 n: int = 20,
                 k: float = 2.0,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.quantum_model_type = quantum_model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.n = n
        self.k = k
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum model
        self.quantum_model = create_steer_quantum_model(
            model_type=quantum_model_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = SteerFeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed using quantum intent prediction."""
        price_data = kwargs.get('price_data', pd.DataFrame())
        
        if len(price_data) < self.n + 5:
            return False
        
        # Check cooldown period
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            return False
        
        try:
            # Create features
            features_df = self.feature_engineer.create_features(price_data)
            if len(features_df) == 0:
                return False
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return False
            
            # Prepare features for quantum model
            X = latest_features[feature_cols].values
            
            # Ensure we have the right number of features
            if X.shape[1] > self.feature_dim:
                X = X[:, :self.feature_dim]
            elif X.shape[1] < self.feature_dim:
                padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Make quantum intent prediction
            if self.quantum_model.is_trained:
                prediction = self.quantum_model.predict(X)[0]
                should_rebalance = prediction > 0.5
            else:
                # Fallback to traditional Bollinger Bands
                sma = price_data['close'].rolling(self.n).mean().iloc[-1]
                std = price_data['close'].rolling(self.n).std().iloc[-1]
                upper_band = sma + (self.k * std)
                lower_band = sma - (self.k * std)
                should_rebalance = current_price > upper_band or current_price < lower_band
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'quantum_prediction': prediction if self.quantum_model.is_trained else 0,
                'should_rebalance': should_rebalance,
                'quantum_model_type': self.quantum_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum Bollinger rebalancing: model={self.quantum_model_type}, "
                           f"prediction={prediction if self.quantum_model.is_trained else 'N/A'}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum intent prediction: {e}")
            return False
    
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using quantum-enhanced Bollinger Bands."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate Bollinger Bands
        sma = price_data['close'].rolling(self.n).mean().iloc[-1]
        std = price_data['close'].rolling(self.n).std().iloc[-1]
        upper_band = sma + (self.k * std)
        lower_band = sma - (self.k * std)
        
        # Adjust bands based on quantum prediction
        if should_rebalance and self.quantum_model.is_trained:
            try:
                # Get quantum prediction for band adjustment
                features_df = self.feature_engineer.create_features(price_data)
                if len(features_df) > 0:
                    latest_features = features_df.iloc[-1:].copy()
                    feature_cols = [col for col in latest_features.columns 
                                  if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    if len(feature_cols) > 0:
                        X = latest_features[feature_cols].values
                        
                        # Ensure correct feature dimension
                        if X.shape[1] > self.feature_dim:
                            X = X[:, :self.feature_dim]
                        elif X.shape[1] < self.feature_dim:
                            padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                            X = np.hstack([X, padding])
                        
                        # Get quantum prediction
                        prediction = self.quantum_model.predict_proba(X)[0]
                        confidence = max(prediction)
                        
                        # Adjust bands based on quantum confidence
                        adjustment_factor = 1 + (confidence - 0.5) * 0.5  # Range: 0.75 to 1.25
                        upper_band *= adjustment_factor
                        lower_band *= adjustment_factor
                        
            except Exception as e:
                logger.error(f"Error in quantum band adjustment: {e}")
        
        # Calculate position ranges
        ranges = [(lower_band, upper_band)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_prediction = np.mean([d['quantum_prediction'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Enhanced Bollinger Bands',
            'quantum_model_type': self.quantum_model_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_dim': self.feature_dim,
            'n': self.n,
            'k': self.k,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'rebalance_count': self.rebalance_count,
            'avg_quantum_prediction': avg_prediction,
            'decision_history_count': len(self.decision_history),
            'model_trained': self.quantum_model.is_trained
        }

class QuantumKeltnerStrategy(BaseStrategy):
    """Quantum SVM-enhanced Keltner Channels strategy for Steer Intent."""
    
    def __init__(self, 
                 quantum_model_type: str = "qsvm_price",
                 n_qubits: int = 4,
                 feature_dim: int = 8,
                 n: int = 20,
                 m: float = 2.0,
                 rebalance_cooldown_hours: int = 2):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.quantum_model_type = quantum_model_type
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.n = n
        self.m = m
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum model
        self.quantum_model = create_steer_quantum_model(
            model_type=quantum_model_type,
            n_qubits=n_qubits,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = SteerFeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def _calculate_atr(self, price_data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed using quantum price prediction."""
        price_data = kwargs.get('price_data', pd.DataFrame())
        
        if len(price_data) < self.n + 5:
            return False
        
        # Check cooldown period
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            return False
        
        try:
            # Create features
            features_df = self.feature_engineer.create_features(price_data)
            if len(features_df) == 0:
                return False
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return False
            
            # Prepare features for quantum model
            X = latest_features[feature_cols].values
            
            # Ensure we have the right number of features
            if X.shape[1] > self.feature_dim:
                X = X[:, :self.feature_dim]
            elif X.shape[1] < self.feature_dim:
                padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Make quantum price prediction
            if self.quantum_model.is_trained:
                prediction = self.quantum_model.predict(X)[0]
                should_rebalance = prediction > 0.5
            else:
                # Fallback to traditional Keltner Channels
                ema = price_data['close'].ewm(span=self.n).mean().iloc[-1]
                atr = self._calculate_atr(price_data, self.n).iloc[-1]
                upper_channel = ema + (self.m * atr)
                lower_channel = ema - (self.m * atr)
                should_rebalance = current_price > upper_channel or current_price < lower_channel
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'quantum_prediction': prediction if self.quantum_model.is_trained else 0,
                'should_rebalance': should_rebalance,
                'quantum_model_type': self.quantum_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum Keltner rebalancing: model={self.quantum_model_type}, "
                           f"prediction={prediction if self.quantum_model.is_trained else 'N/A'}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum price prediction: {e}")
            return False
    
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using quantum-enhanced Keltner Channels."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate Keltner Channels
        ema = price_data['close'].ewm(span=self.n).mean().iloc[-1]
        atr = self._calculate_atr(price_data, self.n).iloc[-1]
        upper_channel = ema + (self.m * atr)
        lower_channel = ema - (self.m * atr)
        
        # Adjust channels based on quantum prediction
        if should_rebalance and self.quantum_model.is_trained:
            try:
                # Get quantum prediction for channel adjustment
                features_df = self.feature_engineer.create_features(price_data)
                if len(features_df) > 0:
                    latest_features = features_df.iloc[-1:].copy()
                    feature_cols = [col for col in latest_features.columns 
                                  if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    if len(feature_cols) > 0:
                        X = latest_features[feature_cols].values
                        
                        # Ensure correct feature dimension
                        if X.shape[1] > self.feature_dim:
                            X = X[:, :self.feature_dim]
                        elif X.shape[1] < self.feature_dim:
                            padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                            X = np.hstack([X, padding])
                        
                        # Get quantum prediction
                        prediction = self.quantum_model.predict_proba(X)[0]
                        confidence = max(prediction)
                        
                        # Adjust channels based on quantum confidence
                        adjustment_factor = 1 + (confidence - 0.5) * 0.3  # Range: 0.85 to 1.15
                        upper_channel *= adjustment_factor
                        lower_channel *= adjustment_factor
                        
            except Exception as e:
                logger.error(f"Error in quantum channel adjustment: {e}")
        
        # Calculate position ranges
        ranges = [(lower_channel, upper_channel)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_prediction = np.mean([d['quantum_prediction'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Enhanced Keltner Channels',
            'quantum_model_type': self.quantum_model_type,
            'n_qubits': self.n_qubits,
            'feature_dim': self.feature_dim,
            'n': self.n,
            'm': self.m,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'rebalance_count': self.rebalance_count,
            'avg_quantum_prediction': avg_prediction,
            'decision_history_count': len(self.decision_history),
            'model_trained': self.quantum_model.is_trained
        }

class QuantumHybridStrategy(BaseStrategy):
    """Hybrid quantum strategy combining multiple quantum models for Steer Intent."""
    
    def __init__(self, 
                 intent_model_type: str = "qnn_intent",
                 price_model_type: str = "qsvm_price",
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 feature_dim: int = 8,
                 intent_weight: float = 0.6,
                 price_weight: float = 0.4,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.intent_model_type = intent_model_type
        self.price_model_type = price_model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.intent_weight = intent_weight
        self.price_weight = price_weight
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum models
        self.intent_model = create_steer_quantum_model(
            model_type=intent_model_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_dim=feature_dim
        )
        
        self.price_model = create_steer_quantum_model(
            model_type=price_model_type,
            n_qubits=n_qubits,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = SteerFeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed using hybrid quantum predictions."""
        price_data = kwargs.get('price_data', pd.DataFrame())
        
        if len(price_data) < 20:
            return False
        
        # Check cooldown period
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            return False
        
        try:
            # Create features
            features_df = self.feature_engineer.create_features(price_data)
            if len(features_df) == 0:
                return False
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return False
            
            # Prepare features for quantum models
            X = latest_features[feature_cols].values
            
            # Ensure we have the right number of features
            if X.shape[1] > self.feature_dim:
                X = X[:, :self.feature_dim]
            elif X.shape[1] < self.feature_dim:
                padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Get quantum predictions
            intent_score = 0
            price_score = 0
            
            if self.intent_model.is_trained:
                intent_pred = self.intent_model.predict_proba(X)[0]
                intent_score = max(intent_pred)
            
            if self.price_model.is_trained:
                price_pred = self.price_model.predict(X)[0]
                price_score = price_pred
            
            # Combine predictions
            combined_score = (self.intent_weight * intent_score + 
                            self.price_weight * price_score)
            
            should_rebalance = combined_score > 0.5
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'intent_score': intent_score,
                'price_score': price_score,
                'combined_score': combined_score,
                'should_rebalance': should_rebalance,
                'intent_model_type': self.intent_model_type,
                'price_model_type': self.price_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum hybrid rebalancing: intent={intent_score:.3f}, "
                           f"price={price_score:.3f}, combined={combined_score:.3f}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum hybrid prediction: {e}")
            return False
    
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using hybrid quantum predictions."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate dynamic range based on hybrid quantum predictions
        if should_rebalance and (self.intent_model.is_trained or self.price_model.is_trained):
            try:
                # Get quantum predictions for range adjustment
                features_df = self.feature_engineer.create_features(price_data)
                if len(features_df) > 0:
                    latest_features = features_df.iloc[-1:].copy()
                    feature_cols = [col for col in latest_features.columns 
                                  if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    if len(feature_cols) > 0:
                        X = latest_features[feature_cols].values
                        
                        # Ensure correct feature dimension
                        if X.shape[1] > self.feature_dim:
                            X = X[:, :self.feature_dim]
                        elif X.shape[1] < self.feature_dim:
                            padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                            X = np.hstack([X, padding])
                        
                        # Get quantum predictions
                        intent_confidence = 0
                        price_level = 0
                        
                        if self.intent_model.is_trained:
                            intent_pred = self.intent_model.predict_proba(X)[0]
                            intent_confidence = max(intent_pred)
                        
                        if self.price_model.is_trained:
                            price_pred = self.price_model.predict(X)[0]
                            price_level = price_pred
                        
                        # Calculate dynamic range based on hybrid predictions
                        range_adjustment = (self.intent_weight * intent_confidence + 
                                          self.price_weight * price_level)
                        
                        # Create adaptive range
                        base_range = current_price * 0.1  # 10% base range
                        dynamic_range = base_range * (1 + range_adjustment)
                        
                        lower_price = current_price - dynamic_range
                        upper_price = current_price + dynamic_range
                        
                        ranges = [(lower_price, upper_price)]
                        liquidities = [portfolio_value * 0.5]
                        
                        return ranges, liquidities
                        
            except Exception as e:
                logger.error(f"Error in quantum hybrid range calculation: {e}")
        
        # Default range calculation
        base_range = current_price * 0.1
        lower_price = current_price - base_range
        upper_price = current_price + base_range
        
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_intent_score = np.mean([d['intent_score'] for d in self.decision_history]) if self.decision_history else 0
        avg_price_score = np.mean([d['price_score'] for d in self.decision_history]) if self.decision_history else 0
        avg_combined_score = np.mean([d['combined_score'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Hybrid',
            'intent_model_type': self.intent_model_type,
            'price_model_type': self.price_model_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_dim': self.feature_dim,
            'intent_weight': self.intent_weight,
            'price_weight': self.price_weight,
            'rebalance_count': self.rebalance_count,
            'avg_intent_score': avg_intent_score,
            'avg_price_score': avg_price_score,
            'avg_combined_score': avg_combined_score,
            'decision_history_count': len(self.decision_history),
            'intent_model_trained': self.intent_model.is_trained,
            'price_model_trained': self.price_model.is_trained
        }
