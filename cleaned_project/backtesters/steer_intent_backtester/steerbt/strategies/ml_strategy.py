"""
Machine learning-based Steer Intent strategy implementations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

from .base import BaseStrategy
from ..ml import SteerFeatureEngineer, SteerMLStrategy, IntentPredictor, PricePredictor, SteerMLTrainer
from ..ml.models import SteerMLBollingerStrategy

logger = logging.getLogger(__name__)

class MLBollingerStrategy(BaseStrategy):
    """ML-enhanced Bollinger Bands strategy for Steer Intent."""
    
    def __init__(self, 
                 ml_bollinger_strategy: SteerMLBollingerStrategy,
                 n: int = 20,
                 k: float = 2.0,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        self.ml_bollinger_strategy = ml_bollinger_strategy
        self.n = n
        self.k = k
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using ML-enhanced Bollinger Bands.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) and liquidities is list of amounts
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Use ML Bollinger strategy to get ranges
        ranges_info = self.ml_bollinger_strategy.calculate_ranges(
            price_data=price_data,
            current_price=current_price,
            portfolio_value=portfolio_value
        )
        
        # Check cooldown period
        should_rebalance = ranges_info.get('should_rebalance', False)
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            should_rebalance = False
        
        # Record rebalancing if needed
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
            
            logger.info(f"ML Bollinger rebalancing: bb_pos={ranges_info.get('bb_position', 0):.3f}, "
                       f"ml_conf={ranges_info.get('ml_confidence', 0):.3f}, "
                       f"combined={ranges_info.get('combined_signal', 0):.3f}")
        
        # Record decision
        decision_record = {
            'timestamp': current_time,
            'current_price': current_price,
            'ranges_info': ranges_info
        }
        self.decision_history.append(decision_record)
        
        # Extract position information
        lower_price = ranges_info.get('lower_price', current_price * 0.95)
        upper_price = ranges_info.get('upper_price', current_price * 1.05)
        
        # Return in the expected format
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        ml_info = self.ml_bollinger_strategy.get_strategy_info()
        
        return {
            'strategy_type': 'ML-Enhanced Bollinger Bands',
            'n': self.n,
            'k': self.k,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'rebalance_count': self.rebalance_count,
            'decision_history_count': len(self.decision_history),
            **ml_info
        }

class MLKeltnerStrategy(BaseStrategy):
    """ML-enhanced Keltner Channels strategy for Steer Intent."""
    
    def __init__(self, 
                 intent_model: IntentPredictor,
                 feature_engineer: SteerFeatureEngineer,
                 n: int = 20,
                 m: float = 2.0,
                 ml_weight: float = 0.7,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        self.intent_model = intent_model
        self.feature_engineer = feature_engineer
        self.n = n
        self.m = m
        self.ml_weight = ml_weight
        self.traditional_weight = 1.0 - ml_weight
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using ML-enhanced Keltner Channels.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) and liquidities is list of amounts
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Traditional Keltner Channels calculation
        ema = price_data['close'].ewm(span=self.n).mean().iloc[-1]
        atr = self._calculate_atr(price_data, self.n).iloc[-1]
        
        keltner_upper = ema + (self.m * atr)
        keltner_lower = ema - (self.m * atr)
        keltner_position = (current_price - keltner_lower) / (keltner_upper - keltner_lower)
        
        # Traditional signal
        traditional_signal = 0
        if keltner_position < 0.2:  # Oversold
            traditional_signal = 1
        elif keltner_position > 0.8:  # Overbought
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
        
        # Check cooldown period
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            should_rebalance = False
        
        # Calculate position width based on ATR
        position_width = min(0.5, max(0.05, atr / current_price * 2))
        
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
            
            logger.info(f"ML Keltner rebalancing: keltner_pos={keltner_position:.3f}, "
                       f"ml_conf={ml_confidence:.3f}, combined={combined_signal:.3f}")
        
        # Record decision
        decision_record = {
            'timestamp': current_time,
            'keltner_position': keltner_position,
            'traditional_signal': traditional_signal,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'combined_signal': combined_signal,
            'should_rebalance': should_rebalance,
            'position_width': position_width
        }
        self.decision_history.append(decision_record)
        
        # Calculate position ranges
        lower_price = current_price * (1 - position_width)
        upper_price = current_price * (1 + position_width)
        
        # Return in the expected format
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_ml_confidence = np.mean([d['ml_confidence'] for d in self.decision_history]) if self.decision_history else 0
        avg_combined_signal = np.mean([d['combined_signal'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'ML-Enhanced Keltner Channels',
            'n': self.n,
            'm': self.m,
            'ml_weight': self.ml_weight,
            'traditional_weight': self.traditional_weight,
            'rebalance_count': self.rebalance_count,
            'avg_ml_confidence': avg_ml_confidence,
            'avg_combined_signal': avg_combined_signal,
            'decision_history_count': len(self.decision_history)
        }

class MLDonchianStrategy(BaseStrategy):
    """ML-enhanced Donchian Channels strategy for Steer Intent."""
    
    def __init__(self, 
                 intent_model: IntentPredictor,
                 feature_engineer: SteerFeatureEngineer,
                 n: int = 20,
                 ml_weight: float = 0.7,
                 rebalance_cooldown_hours: int = 1):
        super().__init__()
        
        self.intent_model = intent_model
        self.feature_engineer = feature_engineer
        self.n = n
        self.ml_weight = ml_weight
        self.traditional_weight = 1.0 - ml_weight
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.decision_history = []
        
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using ML-enhanced Donchian Channels.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) and liquidities is list of amounts
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Traditional Donchian Channels calculation
        donchian_high = price_data['high'].rolling(self.n).max().iloc[-1]
        donchian_low = price_data['low'].rolling(self.n).min().iloc[-1]
        donchian_position = (current_price - donchian_low) / (donchian_high - donchian_low)
        
        # Traditional signal
        traditional_signal = 0
        if donchian_position < 0.2:  # Near lower channel
            traditional_signal = 1
        elif donchian_position > 0.8:  # Near upper channel
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
        
        # Check cooldown period
        if (self.last_rebalance_time is not None and 
            (current_time - self.last_rebalance_time).total_seconds() < self.rebalance_cooldown_hours * 3600):
            should_rebalance = False
        
        # Calculate position width based on channel width
        channel_width = (donchian_high - donchian_low) / current_price
        position_width = min(0.5, max(0.05, channel_width * 0.5))
        
        if should_rebalance:
            self.last_rebalance_time = current_time
            self.rebalance_count += 1
            
            logger.info(f"ML Donchian rebalancing: donchian_pos={donchian_position:.3f}, "
                       f"ml_conf={ml_confidence:.3f}, combined={combined_signal:.3f}")
        
        # Record decision
        decision_record = {
            'timestamp': current_time,
            'donchian_position': donchian_position,
            'traditional_signal': traditional_signal,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'combined_signal': combined_signal,
            'should_rebalance': should_rebalance,
            'position_width': position_width
        }
        self.decision_history.append(decision_record)
        
        # Calculate position ranges
        lower_price = current_price * (1 - position_width)
        upper_price = current_price * (1 + position_width)
        
        # Return in the expected format
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        avg_ml_confidence = np.mean([d['ml_confidence'] for d in self.decision_history]) if self.decision_history else 0
        avg_combined_signal = np.mean([d['combined_signal'] for d in self.decision_history]) if self.decision_history else 0
        
        return {
            'strategy_type': 'ML-Enhanced Donchian Channels',
            'n': self.n,
            'ml_weight': self.ml_weight,
            'traditional_weight': self.traditional_weight,
            'rebalance_count': self.rebalance_count,
            'avg_ml_confidence': avg_ml_confidence,
            'avg_combined_signal': avg_combined_signal,
            'decision_history_count': len(self.decision_history)
        }

class MLHybridStrategy(BaseStrategy):
    """Hybrid ML strategy combining multiple traditional strategies."""
    
    def __init__(self, 
                 steer_ml_strategy: SteerMLStrategy,
                 traditional_weight: float = 0.3,
                 ml_weight: float = 0.7,
                 base_width: float = 0.1,
                 rebalance_cooldown_hours: int = 4):
        super().__init__()
        
        self.steer_ml_strategy = steer_ml_strategy
        self.traditional_weight = traditional_weight
        self.ml_weight = ml_weight
        self.base_width = base_width
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        
        self.last_rebalance_time = None
        self.rebalance_count = 0
        self.hybrid_decisions = []
        
    def calculate_range(self, 
                        price_data: pd.DataFrame, 
                        current_price: float, 
                        portfolio_value: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using hybrid ML + traditional approach.
        
        Args:
            price_data: Historical price data
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) and liquidities is list of amounts
        """
        current_time = price_data.index[-1] if len(price_data) > 0 else datetime.now()
        
        # Get ML prediction
        ml_should_rebalance, ml_decision_info = self.steer_ml_strategy.should_rebalance(
            current_price, price_data, current_time
        )
        
        # Get traditional indicator
        traditional_decision = self._get_traditional_decision(price_data, current_price)
        
        # Combine decisions
        ml_score = ml_decision_info.get('intent_probability', 0.5)
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
        lower_price = current_price * (1 - hybrid_width)
        upper_price = current_price * (1 + hybrid_width)
        
        # Return in the expected format
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.5]  # 50% of portfolio in this position
        
        return ranges, liquidities
    
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
