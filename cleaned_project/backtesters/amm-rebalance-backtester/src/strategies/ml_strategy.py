"""
Machine learning-based AMM strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .base import BaseStrategy
from ..ml import FeatureEngineer, MLStrategy, RebalancePredictor, VolatilityPredictor, MLTrainer

logger = logging.getLogger(__name__)

class MLBasedStrategy(BaseStrategy):
    """Machine learning-based AMM strategy."""
    
    def __init__(self, 
                 ml_strategy: MLStrategy,
                 initial_width: float = 0.1,
                 min_width: float = 0.05,
                 max_width: float = 0.5,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        self.ml_strategy = ml_strategy
        self.initial_width = initial_width
        self.min_width = min_width
        self.max_width = max_width
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.current_width = initial_width
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate position ranges using ML predictions.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with position information
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Use ML strategy to determine if rebalancing is needed
        should_rebalance, decision_info = self.ml_strategy.should_rebalance(
            current_price, price_data, current_time
        )
        
        # Record decision
        decision_record = {
            'timestamp': current_time,
            'current_price': current_price,
            'should_rebalance': should_rebalance,
            'decision_info': decision_info
        }
        self.decision_history.append(decision_record)
        
        # Update position width based on ML predictions
        if should_rebalance and self._can_rebalance(current_time):
            # Use volatility-adjusted width from ML prediction
            volatility_adjusted_width = decision_info.get('volatility_adjusted_width', self.current_width)
            self.current_width = np.clip(volatility_adjusted_width, self.min_width, self.max_width)
            
            # Record rebalancing
            self.ml_strategy.record_rebalance(current_time, decision_info)
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
            
            logger.info(f"ML rebalancing triggered: width={self.current_width:.3f}, "
                       f"vol_pred={decision_info.get('predicted_volatility', 0):.4f}, "
                       f"prob={decision_info.get('rebalance_probability', 0):.3f}")
        
        # Calculate position ranges
        width_pct = self.current_width * 100
        lower_price = current_price * (1 - self.current_width)
        upper_price = current_price * (1 + self.current_width)
        
        return {
            'lower_price': lower_price,
            'upper_price': upper_price,
            'width_pct': width_pct,
            'should_rebalance': should_rebalance,
            'decision_info': decision_info,
            'rebalance_count': self.rebalance_count
        }
    
    def _can_rebalance(self, current_time: pd.Timestamp) -> bool:
        """Check if enough time has passed since last rebalancing."""
        if self.last_rebalance_time is None:
            return True
        
        time_diff = current_time - self.last_rebalance_time
        return time_diff >= timedelta(hours=self.rebalance_cooldown_hours)
    
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed."""
        # Use ML strategy to determine if rebalancing is needed
        should_rebalance, decision_info = self.ml_strategy.should_rebalance(
            current_price, kwargs.get('price_data', pd.DataFrame()), current_time
        )
        return should_rebalance
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        ml_stats = self.ml_strategy.get_strategy_stats()
        
        return {
            'strategy_type': 'ML-Based',
            'current_width': self.current_width,
            'rebalance_count': self.rebalance_count,
            'ml_stats': ml_stats,
            'decision_history_count': len(self.decision_history)
        }

class MLVolatilityStrategy(BaseStrategy):
    """ML-enhanced volatility-based strategy."""
    
    def __init__(self, 
                 volatility_model: VolatilityPredictor,
                 feature_engineer: FeatureEngineer,
                 base_k_width: float = 1.5,
                 volatility_multiplier: float = 2.0,
                 min_width: float = 0.05,
                 max_width: float = 0.5,
                 rebalance_cooldown_hours: int = 6):
        super().__init__()
        
        self.volatility_model = volatility_model
        self.feature_engineer = feature_engineer
        self.base_k_width = base_k_width
        self.volatility_multiplier = volatility_multiplier
        self.min_width = min_width
        self.max_width = max_width
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.volatility_history = []
        
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate position ranges using ML volatility predictions.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with position information
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Predict future volatility using ML model
        predicted_volatility = self._predict_volatility(price_data)
        
        # Calculate dynamic width based on predicted volatility
        if predicted_volatility is not None:
            # Use predicted volatility to adjust position width
            dynamic_width = predicted_volatility * self.volatility_multiplier
            dynamic_width = np.clip(dynamic_width, self.min_width, self.max_width)
        else:
            # Fallback to base width
            dynamic_width = self.min_width
        
        # Check if rebalancing is needed
        should_rebalance = self._should_rebalance(current_time, dynamic_width)
        
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
            
            vol_str = f"{predicted_volatility:.4f}" if predicted_volatility is not None else "None"
            logger.info(f"ML volatility rebalancing: predicted_vol={vol_str}, "
                       f"width={dynamic_width:.3f}")
        
        # Record volatility prediction
        self.volatility_history.append({
            'timestamp': current_time,
            'predicted_volatility': predicted_volatility,
            'dynamic_width': dynamic_width,
            'current_price': current_price
        })
        
        # Calculate position ranges
        width_pct = dynamic_width * 100
        lower_price = current_price * (1 - dynamic_width)
        upper_price = current_price * (1 + dynamic_width)
        
        return {
            'lower_price': lower_price,
            'upper_price': upper_price,
            'width_pct': width_pct,
            'should_rebalance': should_rebalance,
            'predicted_volatility': predicted_volatility,
            'dynamic_width': dynamic_width,
            'rebalance_count': self.rebalance_count
        }
    
    def _predict_volatility(self, price_data: pd.DataFrame) -> Optional[float]:
        """Predict future volatility using ML model."""
        try:
            if not self.volatility_model.is_trained:
                return None
            
            # Create features
            features_df = self.feature_engineer.create_features(price_data)
            
            if len(features_df) == 0:
                return None
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            
            # Remove non-feature columns
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return None
            
            X = latest_features[feature_cols]
            
            # Transform features if scaler is fitted
            if self.feature_engineer.is_fitted:
                X = self.feature_engineer.transform_features(latest_features)[feature_cols]
            
            # Predict volatility
            predicted_vol = self.volatility_model.predict(X)[0]
            
            return max(0, predicted_vol)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return None
    
    def _should_rebalance(self, current_time: pd.Timestamp, new_width: float) -> bool:
        """Determine if rebalancing should occur."""
        # Check cooldown period
        if self.last_rebalance_time is not None:
            time_diff = current_time - self.last_rebalance_time
            if time_diff < timedelta(hours=self.rebalance_cooldown_hours):
                return False
        
        # Check if width change is significant
        if len(self.volatility_history) > 0:
            last_width = self.volatility_history[-1]['dynamic_width']
            width_change = abs(new_width - last_width) / last_width
            if width_change < 0.1:  # Less than 10% change
                return False
        
        return True
    
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed."""
        # Check cooldown period
        if self.last_rebalance_time is not None:
            time_diff = current_time - self.last_rebalance_time
            if time_diff < timedelta(hours=self.rebalance_cooldown_hours):
                return False
        
        # Use volatility prediction to determine rebalancing
        price_data = kwargs.get('price_data', pd.DataFrame())
        if len(price_data) == 0:
            return False
        
        predicted_volatility = self._predict_volatility(price_data)
        if predicted_volatility is None:
            return False
        
        # Check if volatility change is significant
        if len(self.volatility_history) > 0:
            last_vol = self.volatility_history[-1]['predicted_volatility']
            if last_vol is not None:
                vol_change = abs(predicted_volatility - last_vol) / last_vol
                return vol_change > 0.1  # 10% volatility change threshold
        
        return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_predicted_vol = np.mean([h['predicted_volatility'] for h in self.volatility_history 
                                   if h['predicted_volatility'] is not None]) if self.volatility_history else 0
        
        return {
            'strategy_type': 'ML-Volatility',
            'base_k_width': self.base_k_width,
            'volatility_multiplier': self.volatility_multiplier,
            'rebalance_count': self.rebalance_count,
            'avg_predicted_volatility': avg_predicted_vol,
            'volatility_history_count': len(self.volatility_history)
        }

class MLHybridStrategy(BaseStrategy):
    """Hybrid strategy combining ML predictions with traditional indicators."""
    
    def __init__(self, 
                 ml_strategy: MLStrategy,
                 traditional_weight: float = 0.3,
                 ml_weight: float = 0.7,
                 base_width: float = 0.1,
                 rebalance_cooldown_hours: int = 4):
        super().__init__()
        
        self.ml_strategy = ml_strategy
        self.traditional_weight = traditional_weight
        self.ml_weight = ml_weight
        self.base_width = base_width
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.hybrid_decisions = []
        
    def calculate_ranges(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate position ranges using hybrid ML + traditional approach.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with position information
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Get ML prediction
        ml_should_rebalance, ml_decision_info = self.ml_strategy.should_rebalance(
            current_price, price_data, current_time
        )
        
        # Get traditional indicator
        traditional_decision = self._get_traditional_decision(price_data, current_price)
        
        # Combine decisions
        ml_score = ml_decision_info.get('rebalance_probability', 0.5)
        traditional_score = 1.0 if traditional_decision['should_rebalance'] else 0.0
        
        combined_score = (self.ml_weight * ml_score + 
                         self.traditional_weight * traditional_score)
        
        should_rebalance = combined_score > 0.5 and self._can_rebalance(current_time)
        
        # Calculate hybrid width
        ml_width = ml_decision_info.get('volatility_adjusted_width', self.base_width)
        traditional_width = traditional_decision.get('width', self.base_width)
        
        hybrid_width = (self.ml_weight * ml_width + 
                       self.traditional_weight * traditional_width)
        
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
        
        # Record hybrid decision
        decision_record = {
            'timestamp': current_time,
            'ml_score': ml_score,
            'traditional_score': traditional_score,
            'combined_score': combined_score,
            'ml_width': ml_width,
            'traditional_width': traditional_width,
            'hybrid_width': hybrid_width,
            'should_rebalance': should_rebalance
        }
        self.hybrid_decisions.append(decision_record)
        
        # Calculate position ranges
        width_pct = hybrid_width * 100
        lower_price = current_price * (1 - hybrid_width)
        upper_price = current_price * (1 + hybrid_width)
        
        return {
            'lower_price': lower_price,
            'upper_price': upper_price,
            'width_pct': width_pct,
            'should_rebalance': should_rebalance,
            'ml_score': ml_score,
            'traditional_score': traditional_score,
            'combined_score': combined_score,
            'hybrid_width': hybrid_width,
            'rebalance_count': self.rebalance_count
        }
    
    def _get_traditional_decision(self, price_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Get traditional technical analysis decision."""
        if len(price_data) < 20:
            return {'should_rebalance': False, 'width': self.base_width}
        
        # Simple moving average based decision
        sma_20 = price_data['close'].rolling(20).mean().iloc[-1]
        price_deviation = abs(current_price - sma_20) / sma_20
        
        # Volatility-based width
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        vol_width = min(0.5, max(0.05, volatility * 2))
        
        return {
            'should_rebalance': price_deviation > 0.02,  # 2% deviation threshold
            'width': vol_width
        }
    
    def _can_rebalance(self, current_time: pd.Timestamp) -> bool:
        """Check if enough time has passed since last rebalancing."""
        if self.last_rebalance_time is None:
            return True
        
        time_diff = current_time - self.last_rebalance_time
        return time_diff >= timedelta(hours=self.rebalance_cooldown_hours)
    
    def should_rebalance(self, current_price: float, current_time: pd.Timestamp, **kwargs) -> bool:
        """Determine if rebalancing is needed."""
        # Check cooldown period
        if self.last_rebalance_time is not None:
            time_diff = current_time - self.last_rebalance_time
            if time_diff < timedelta(hours=self.rebalance_cooldown_hours):
                return False
        
        # Get ML prediction
        price_data = kwargs.get('price_data', pd.DataFrame())
        if len(price_data) == 0:
            return False
        
        ml_should_rebalance, ml_decision_info = self.ml_strategy.should_rebalance(
            current_price, price_data, current_time
        )
        
        # Get traditional indicator
        traditional_decision = self._get_traditional_decision(price_data, current_price)
        
        # Combine decisions
        ml_score = ml_decision_info.get('intent_probability', 0.5)
        traditional_score = 1.0 if traditional_decision['should_rebalance'] else 0.0
        
        combined_score = (self.ml_weight * ml_score + 
                         self.traditional_weight * traditional_score)
        
        return combined_score > 0.5
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_ml_score = np.mean([d['ml_score'] for d in self.hybrid_decisions]) if self.hybrid_decisions else 0
        avg_traditional_score = np.mean([d['traditional_score'] for d in self.hybrid_decisions]) if self.hybrid_decisions else 0
        avg_combined_score = np.mean([d['combined_score'] for d in self.hybrid_decisions]) if self.hybrid_decisions else 0
        
        return {
            'strategy_type': 'ML-Hybrid',
            'ml_weight': self.ml_weight,
            'traditional_weight': self.traditional_weight,
            'rebalance_count': self.rebalance_count,
            'avg_ml_score': avg_ml_score,
            'avg_traditional_score': avg_traditional_score,
            'avg_combined_score': avg_combined_score,
            'hybrid_decisions_count': len(self.hybrid_decisions)
        }
