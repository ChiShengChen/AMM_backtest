"""
Quantum machine learning-based AMM rebalancing strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

from .base import BaseStrategy
from ..ml import (
    FeatureEngineer, 
    QUANTUM_AVAILABLE,
    QNNRebalancePredictor,
    QSVMVolatilityPredictor,
    PennyLaneQNNPredictor,
    create_quantum_model
)

logger = logging.getLogger(__name__)

class QuantumBasedStrategy(BaseStrategy):
    """Quantum Neural Network-based rebalancing strategy for AMM."""
    
    def __init__(self, 
                 quantum_model_type: str = "qnn_rebalance",
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 feature_dim: int = 8,
                 rebalance_threshold: float = 0.3,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.quantum_model_type = quantum_model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum model
        self.quantum_model = create_quantum_model(
            model_type=quantum_model_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        self.current_width = 0.1
        
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed using quantum ML prediction."""
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
            
            # Prepare features for quantum model
            X = latest_features[feature_cols].values
            
            # Ensure we have the right number of features
            if X.shape[1] > self.feature_dim:
                X = X[:, :self.feature_dim]
            elif X.shape[1] < self.feature_dim:
                # Pad with zeros
                padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Make quantum prediction
            if self.quantum_model.is_trained:
                prediction = self.quantum_model.predict(X)[0]
                should_rebalance = prediction > self.rebalance_threshold
            else:
                # Fallback to simple threshold
                price_deviation = abs(current_price - price_data['close'].mean()) / price_data['close'].mean()
                should_rebalance = price_deviation > 0.02
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'prediction': prediction if self.quantum_model.is_trained else 0,
                'should_rebalance': should_rebalance,
                'quantum_model_type': self.quantum_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum rebalancing triggered: model={self.quantum_model_type}, "
                           f"prediction={prediction if self.quantum_model.is_trained else 'N/A'}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum prediction: {e}")
            return False
    
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using quantum ML predictions."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate dynamic width based on quantum prediction
        if should_rebalance and self.quantum_model.is_trained:
            try:
                # Get quantum prediction for width adjustment
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
                        
                        # Get quantum prediction for width
                        prediction = self.quantum_model.predict_proba(X)[0]
                        confidence = max(prediction)
                        
                        # Adjust width based on quantum confidence
                        self.current_width = 0.05 + (confidence * 0.15)  # Range: 0.05 to 0.20
                        
            except Exception as e:
                logger.error(f"Error in quantum width calculation: {e}")
                self.current_width = 0.1  # Default width
        
        # Calculate position ranges
        width_pct = self.current_width
        lower_price = current_price * (1 - width_pct)
        upper_price = current_price * (1 + width_pct)
        
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_prediction = np.mean([d['prediction'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Based',
            'quantum_model_type': self.quantum_model_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_dim': self.feature_dim,
            'rebalance_threshold': self.rebalance_threshold,
            'current_width': self.current_width,
            'rebalance_count': self.rebalance_count,
            'avg_quantum_prediction': avg_prediction,
            'decision_history_count': len(self.decision_history),
            'model_trained': self.quantum_model.is_trained
        }

class QuantumVolatilityStrategy(BaseStrategy):
    """Quantum SVM-based volatility prediction strategy for AMM."""
    
    def __init__(self, 
                 quantum_model_type: str = "qsvm_volatility",
                 n_qubits: int = 4,
                 feature_dim: int = 8,
                 base_width: float = 0.1,
                 volatility_sensitivity: float = 2.0,
                 rebalance_cooldown_hours: int = 2):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.quantum_model_type = quantum_model_type
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.base_width = base_width
        self.volatility_sensitivity = volatility_sensitivity
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum model
        self.quantum_model = create_quantum_model(
            model_type=quantum_model_type,
            n_qubits=n_qubits,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        self.current_width = base_width
        
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed based on quantum volatility prediction."""
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
            
            # Prepare features for quantum model
            X = latest_features[feature_cols].values
            
            # Ensure we have the right number of features
            if X.shape[1] > self.feature_dim:
                X = X[:, :self.feature_dim]
            elif X.shape[1] < self.feature_dim:
                padding = np.zeros((X.shape[0], self.feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Make quantum volatility prediction
            if self.quantum_model.is_trained:
                prediction = self.quantum_model.predict(X)[0]
                # Convert prediction to volatility level (0 = low, 1 = high)
                volatility_level = prediction
            else:
                # Fallback to traditional volatility calculation
                returns = price_data['close'].pct_change().dropna()
                volatility_level = returns.rolling(20).std().iloc[-1] * self.volatility_sensitivity
            
            # Determine if rebalancing is needed based on volatility
            should_rebalance = volatility_level > 0.5
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'volatility_prediction': volatility_level,
                'should_rebalance': should_rebalance,
                'quantum_model_type': self.quantum_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum volatility rebalancing: model={self.quantum_model_type}, "
                           f"vol_pred={volatility_level:.4f}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum volatility prediction: {e}")
            return False
    
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using quantum volatility predictions."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate dynamic width based on quantum volatility prediction
        if should_rebalance and self.quantum_model.is_trained:
            try:
                # Get quantum volatility prediction
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
                        
                        # Get quantum volatility prediction
                        volatility_pred = self.quantum_model.predict(X)[0]
                        
                        # Adjust width based on quantum volatility prediction
                        self.current_width = self.base_width * (1 + volatility_pred * self.volatility_sensitivity)
                        self.current_width = min(0.5, max(0.05, self.current_width))  # Clamp to reasonable range
                        
            except Exception as e:
                logger.error(f"Error in quantum volatility width calculation: {e}")
                self.current_width = self.base_width
        
        # Calculate position ranges
        width_pct = self.current_width
        lower_price = current_price * (1 - width_pct)
        upper_price = current_price * (1 + width_pct)
        
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_volatility_pred = np.mean([d['volatility_prediction'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Volatility',
            'quantum_model_type': self.quantum_model_type,
            'n_qubits': self.n_qubits,
            'feature_dim': self.feature_dim,
            'base_width': self.base_width,
            'volatility_sensitivity': self.volatility_sensitivity,
            'current_width': self.current_width,
            'rebalance_count': self.rebalance_count,
            'avg_volatility_prediction': avg_volatility_pred,
            'decision_history_count': len(self.decision_history),
            'model_trained': self.quantum_model.is_trained
        }

class QuantumHybridStrategy(BaseStrategy):
    """Hybrid quantum strategy combining multiple quantum models."""
    
    def __init__(self, 
                 rebalance_model_type: str = "qnn_rebalance",
                 volatility_model_type: str = "qsvm_volatility",
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 feature_dim: int = 8,
                 rebalance_weight: float = 0.6,
                 volatility_weight: float = 0.4,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum computing libraries not available. Please install qiskit and pennylane.")
        
        self.rebalance_model_type = rebalance_model_type
        self.volatility_model_type = volatility_model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.rebalance_weight = rebalance_weight
        self.volatility_weight = volatility_weight
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        # Initialize quantum models
        self.rebalance_model = create_quantum_model(
            model_type=rebalance_model_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_dim=feature_dim
        )
        
        self.volatility_model = create_quantum_model(
            model_type=volatility_model_type,
            n_qubits=n_qubits,
            feature_dim=feature_dim
        )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Strategy state
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        self.current_width = 0.1
        
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
            rebalance_score = 0
            volatility_score = 0
            
            if self.rebalance_model.is_trained:
                rebalance_pred = self.rebalance_model.predict_proba(X)[0]
                rebalance_score = max(rebalance_pred)
            
            if self.volatility_model.is_trained:
                volatility_pred = self.volatility_model.predict(X)[0]
                volatility_score = volatility_pred
            
            # Combine predictions
            combined_score = (self.rebalance_weight * rebalance_score + 
                            self.volatility_weight * volatility_score)
            
            should_rebalance = combined_score > 0.5
            
            # Record decision
            decision_record = {
                'timestamp': current_time,
                'current_price': current_price,
                'rebalance_score': rebalance_score,
                'volatility_score': volatility_score,
                'combined_score': combined_score,
                'should_rebalance': should_rebalance,
                'rebalance_model_type': self.rebalance_model_type,
                'volatility_model_type': self.volatility_model_type
            }
            self.decision_history.append(decision_record)
            
            if should_rebalance:
                self.last_rebalance_time = current_time
                self.rebalance_count += 1
                
                logger.info(f"Quantum hybrid rebalancing: rebalance={rebalance_score:.3f}, "
                           f"volatility={volatility_score:.3f}, combined={combined_score:.3f}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error in quantum hybrid prediction: {e}")
            return False
    
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float, 
                        **kwargs) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Calculate position ranges using hybrid quantum predictions."""
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Determine if rebalancing is needed
        should_rebalance = self.should_rebalance(current_price, current_time, price_data=price_data)
        
        # Calculate dynamic width based on hybrid quantum predictions
        if should_rebalance and (self.rebalance_model.is_trained or self.volatility_model.is_trained):
            try:
                # Get quantum predictions for width adjustment
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
                        rebalance_confidence = 0
                        volatility_level = 0
                        
                        if self.rebalance_model.is_trained:
                            rebalance_pred = self.rebalance_model.predict_proba(X)[0]
                            rebalance_confidence = max(rebalance_pred)
                        
                        if self.volatility_model.is_trained:
                            volatility_pred = self.volatility_model.predict(X)[0]
                            volatility_level = volatility_pred
                        
                        # Adjust width based on hybrid predictions
                        width_adjustment = (self.rebalance_weight * rebalance_confidence + 
                                          self.volatility_weight * volatility_level)
                        self.current_width = 0.05 + (width_adjustment * 0.15)
                        self.current_width = min(0.5, max(0.05, self.current_width))
                        
            except Exception as e:
                logger.error(f"Error in quantum hybrid width calculation: {e}")
                self.current_width = 0.1
        
        # Calculate position ranges
        width_pct = self.current_width
        lower_price = current_price * (1 - width_pct)
        upper_price = current_price * (1 + width_pct)
        
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_rebalance_score = np.mean([d['rebalance_score'] for d in self.decision_history]) if self.decision_history else 0
        avg_volatility_score = np.mean([d['volatility_score'] for d in self.decision_history]) if self.decision_history else 0
        avg_combined_score = np.mean([d['combined_score'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'Quantum-Hybrid',
            'rebalance_model_type': self.rebalance_model_type,
            'volatility_model_type': self.volatility_model_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_dim': self.feature_dim,
            'rebalance_weight': self.rebalance_weight,
            'volatility_weight': self.volatility_weight,
            'current_width': self.current_width,
            'rebalance_count': self.rebalance_count,
            'avg_rebalance_score': avg_rebalance_score,
            'avg_volatility_score': avg_volatility_score,
            'avg_combined_score': avg_combined_score,
            'decision_history_count': len(self.decision_history),
            'rebalance_model_trained': self.rebalance_model.is_trained,
            'volatility_model_trained': self.volatility_model.is_trained
        }
