"""
Machine learning models for AMM strategies.
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

class BaseMLModel(ABC):
    """Base class for ML models."""
    
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

class RebalancePredictor(BaseMLModel):
    """ML model for predicting optimal rebalancing decisions."""
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"rebalance_predictor_{model_type}")
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
        """Train the rebalancing prediction model."""
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
        """Predict rebalancing decisions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict rebalancing probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regression models, convert to probabilities
            predictions = self.model.predict(X)
            return np.column_stack([1 - predictions, predictions])

class VolatilityPredictor(BaseMLModel):
    """ML model for predicting future volatility."""
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"volatility_predictor_{model_type}")
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
        """Train the volatility prediction model."""
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
        """Predict future volatility."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)

class MLStrategy:
    """Machine learning-based AMM strategy."""
    
    def __init__(self, 
                 rebalance_model: RebalancePredictor,
                 volatility_model: VolatilityPredictor,
                 feature_engineer,
                 rebalance_threshold: float = 0.5,
                 volatility_threshold: float = 0.02,
                 min_rebalance_interval: int = 1):
        self.rebalance_model = rebalance_model
        self.volatility_model = volatility_model
        self.feature_engineer = feature_engineer
        self.rebalance_threshold = rebalance_threshold
        self.volatility_threshold = volatility_threshold
        self.min_rebalance_interval = min_rebalance_interval
        
        self.last_rebalance_time = None
        self.current_position_width = 0.1  # Default 10% width
        self.rebalance_history = []
        
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
            
            # Predict rebalancing probability
            if self.rebalance_model.is_trained:
                rebalance_proba = self.rebalance_model.predict_proba(X)[0]
                rebalance_decision = rebalance_proba[1] > self.rebalance_threshold
            else:
                rebalance_decision = False
                rebalance_proba = [0.5, 0.5]
            
            # Predict volatility for position sizing
            if self.volatility_model.is_trained:
                predicted_volatility = self.volatility_model.predict(X)[0]
                # Adjust position width based on predicted volatility
                volatility_adjusted_width = max(0.05, min(0.5, predicted_volatility * 2))
            else:
                predicted_volatility = 0.02
                volatility_adjusted_width = self.current_position_width
            
            decision_info = {
                'rebalance_probability': rebalance_proba[1],
                'predicted_volatility': predicted_volatility,
                'volatility_adjusted_width': volatility_adjusted_width,
                'current_position_width': self.current_position_width,
                'feature_count': len(feature_cols)
            }
            
            return rebalance_decision, decision_info
            
        except Exception as e:
            logger.error(f"Error in ML strategy decision: {e}")
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
            'avg_volatility_prediction': np.mean([r['decision_info'].get('predicted_volatility', 0) 
                                                for r in self.rebalance_history]),
            'avg_rebalance_probability': np.mean([r['decision_info'].get('rebalance_probability', 0) 
                                                for r in self.rebalance_history])
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
