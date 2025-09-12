"""
Machine learning models for Steer Intent strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

logger = logging.getLogger(__name__)

class BaseSteerMLModel(ABC):
    """Base class for Steer Intent ML models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump({
            'model': self.model,
            'feature_importance_': self.feature_importance_,
            'model_name': self.model_name
        }, filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_importance_ = data['feature_importance_']
        self.model_name = data['model_name']
        self.is_trained = True
        logger.info(f"Loaded model from {filepath}")

class IntentPredictor(BaseSteerMLModel):
    """ML model for predicting optimal intent-based rebalancing decisions."""
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"intent_predictor_{model_type}")
        self.model_type = model_type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the intent prediction model."""
        logger.info(f"Training {self.model_name} on {len(X)} samples with {len(X.columns)} features")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        
        logger.info(f"Training completed for {self.model_name}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict intent-based rebalancing decisions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict intent probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regression models, convert to probabilities
            predictions = self.model.predict(X)
            return np.column_stack([1 - predictions, predictions])

class PricePredictor(BaseSteerMLModel):
    """ML model for predicting future price movements."""
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"price_predictor_{model_type}")
        self.model_type = model_type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the price prediction model."""
        logger.info(f"Training {self.model_name} on {len(X)} samples with {len(X.columns)} features")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        
        logger.info(f"Training completed for {self.model_name}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict future price movements."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)

class SteerMLStrategy:
    """Machine learning-based Steer Intent strategy."""
    
    def __init__(self, 
                 intent_model: IntentPredictor,
                 price_model: PricePredictor,
                 feature_engineer,
                 strategy_type: str = 'bollinger',
                 rebalance_threshold: float = 0.5,
                 price_threshold: float = 0.02,
                 min_rebalance_interval: int = 1):
        self.intent_model = intent_model
        self.price_model = price_model
        self.feature_engineer = feature_engineer
        self.strategy_type = strategy_type
        self.rebalance_threshold = rebalance_threshold
        self.price_threshold = price_threshold
        self.min_rebalance_interval = min_rebalance_interval
        
        self.last_rebalance_time = None
        self.current_position_width = 0.1  # Default 10% width
        self.rebalance_history = []
        self.prediction_history = []
        
    def should_rebalance(self, 
                        current_price: float, 
                        price_data: pd.DataFrame,
                        current_time: pd.Timestamp) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if rebalancing should occur using ML models.
        
        Args:
            current_price: Current asset price
            price_data: Historical price data for feature engineering
            current_time: Current timestamp
            
        Returns:
            Tuple of (should_rebalance, decision_info)
        """
        # Check minimum rebalance interval
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.min_rebalance_interval * 3600):
            return False, {'reason': 'min_interval_not_met'}
        
        try:
            # Create features for current state
            features_df = self.feature_engineer.create_features(price_data)
            
            if len(features_df) == 0:
                return False, {'reason': 'insufficient_data'}
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            
            # Remove non-feature columns
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return False, {'reason': 'no_features'}
            
            X = latest_features[feature_cols]
            
            # Transform features if scaler is fitted
            if self.feature_engineer.is_fitted:
                X = self.feature_engineer.transform_features(latest_features)[feature_cols]
            
            # Predict intent probability
            if self.intent_model.is_trained:
                intent_proba = self.intent_model.predict_proba(X)[0]
                intent_decision = intent_proba[1] > self.rebalance_threshold
            else:
                intent_decision = False
                intent_proba = [0.5, 0.5]
            
            # Predict future price movement
            if self.price_model.is_trained:
                predicted_return = self.price_model.predict(X)[0]
                price_direction = 1 if predicted_return > self.price_threshold else (-1 if predicted_return < -self.price_threshold else 0)
            else:
                predicted_return = 0
                price_direction = 0
            
            # Combine intent and price predictions
            combined_decision = intent_decision and (abs(price_direction) > 0)
            
            decision_info = {
                'intent_probability': intent_proba[1],
                'predicted_return': predicted_return,
                'price_direction': price_direction,
                'strategy_type': self.strategy_type,
                'feature_count': len(feature_cols),
                'combined_decision': combined_decision
            }
            
            # Record prediction
            self.prediction_history.append({
                'timestamp': current_time,
                'current_price': current_price,
                'decision_info': decision_info
            })
            
            return combined_decision, decision_info
            
        except Exception as e:
            logger.error(f"Error in Steer ML strategy decision: {e}")
            return False, {'reason': 'error', 'error': str(e)}
    
    def update_position_width(self, new_width: float) -> None:
        """Update the current position width."""
        self.current_position_width = new_width
    
    def record_rebalance(self, timestamp: pd.Timestamp, decision_info: Dict[str, Any]) -> None:
        """Record a rebalancing event."""
        self.last_rebalance_time = timestamp
        self.rebalance_history.append({
            'timestamp': timestamp,
            'decision_info': decision_info
        })
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        if not self.rebalance_history:
            return {'total_rebalances': 0}
        
        rebalances = pd.DataFrame(self.rebalance_history)
        
        return {
            'total_rebalances': len(self.rebalance_history),
            'avg_rebalance_interval': self._calculate_avg_interval(),
            'avg_intent_probability': np.mean([r['decision_info'].get('intent_probability', 0) 
                                            for r in self.rebalance_history]),
            'avg_predicted_return': np.mean([r['decision_info'].get('predicted_return', 0) 
                                           for r in self.rebalance_history]),
            'strategy_type': self.strategy_type
        }
    
    def _calculate_avg_interval(self) -> float:
        """Calculate average rebalancing interval in hours."""
        if len(self.rebalance_history) < 2:
            return 0
        
        intervals = []
        for i in range(1, len(self.rebalance_history)):
            interval = (self.rebalance_history[i]['timestamp'] - 
                       self.rebalance_history[i-1]['timestamp']).total_seconds() / 3600
            intervals.append(interval)
        
        return np.mean(intervals) if intervals else 0

class SteerMLBollingerStrategy:
    """ML-enhanced Bollinger Bands strategy."""
    
    def __init__(self, 
                 intent_model: IntentPredictor,
                 feature_engineer,
                 n: int = 20,
                 k: float = 2.0,
                 ml_weight: float = 0.7,
                 traditional_weight: float = 0.3):
        self.intent_model = intent_model
        self.feature_engineer = feature_engineer
        self.n = n
        self.k = k
        self.ml_weight = ml_weight
        self.traditional_weight = traditional_weight
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate position ranges using ML-enhanced Bollinger Bands.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with position information
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else pd.Timestamp.now()
        
        # Traditional Bollinger Bands calculation
        sma = price_data['close'].rolling(self.n).mean().iloc[-1]
        std = price_data['close'].rolling(self.n).std().iloc[-1]
        bb_upper = sma + (self.k * std)
        bb_lower = sma - (self.k * std)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Traditional signal
        traditional_signal = 0
        if bb_position < 0.2:  # Oversold
            traditional_signal = 1
        elif bb_position > 0.8:  # Overbought
            traditional_signal = -1
        
        # ML prediction
        ml_signal = 0
        ml_confidence = 0.5
        
        try:
            if self.intent_model.is_trained:
                features_df = self.feature_engineer.create_features(price_data)
                if len(features_df) > 0:
                    latest_features = features_df.iloc[-1:].copy()
                    feature_cols = [col for col in latest_features.columns 
                                  if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    if len(feature_cols) > 0:
                        X = latest_features[feature_cols]
                        if self.feature_engineer.is_fitted:
                            X = self.feature_engineer.transform_features(latest_features)[feature_cols]
                        
                        ml_proba = self.intent_model.predict_proba(X)[0]
                        ml_confidence = ml_proba[1]
                        
                        if ml_confidence > 0.6:
                            ml_signal = 1
                        elif ml_confidence < 0.4:
                            ml_signal = -1
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
        
        # Combine signals
        combined_signal = (self.ml_weight * ml_signal + 
                          self.traditional_weight * traditional_signal)
        
        should_rebalance = abs(combined_signal) > 0.3
        
        # Calculate position width based on volatility
        volatility = std / sma
        position_width = min(0.5, max(0.05, volatility * 2))
        
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
        
        # Record decision
        decision_record = {
            'timestamp': current_time,
            'bb_position': bb_position,
            'traditional_signal': traditional_signal,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'combined_signal': combined_signal,
            'should_rebalance': should_rebalance,
            'position_width': position_width
        }
        self.decision_history.append(decision_record)
        
        # Calculate position ranges
        width_pct = position_width * 100
        lower_price = current_price * (1 - position_width)
        upper_price = current_price * (1 + position_width)
        
        return {
            'lower_price': lower_price,
            'upper_price': upper_price,
            'width_pct': width_pct,
            'should_rebalance': should_rebalance,
            'bb_position': bb_position,
            'traditional_signal': traditional_signal,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'combined_signal': combined_signal,
            'rebalance_count': self.rebalance_count
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_ml_confidence = np.mean([d['ml_confidence'] for d in self.decision_history]) if self.decision_history else 0
        avg_combined_signal = np.mean([d['combined_signal'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'ML-Enhanced Bollinger Bands',
            'n': self.n,
            'k': self.k,
            'ml_weight': self.ml_weight,
            'traditional_weight': self.traditional_weight,
            'rebalance_count': self.rebalance_count,
            'avg_ml_confidence': avg_ml_confidence,
            'avg_combined_signal': avg_combined_signal,
            'decision_history_count': len(self.decision_history)
        }
