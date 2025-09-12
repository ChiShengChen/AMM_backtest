"""
Base strategy class for CLMM strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all CLMM strategies."""
    
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.parameters = kwargs
        self.initialized = False
        
        # Strategy state
        self.current_ranges: List[Tuple[float, float]] = []
        self.current_liquidities: List[float] = []
        self.last_update = None
        
        # Performance tracking
        self.rebalance_count = 0
        self.total_fees_earned = 0.0
        
    @abstractmethod
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate new position ranges and liquidities.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) and liquidities is list of amounts
        """
        pass
    
    def initialize(
        self,
        initial_price: float,
        portfolio_value: float,
        price_data: pd.DataFrame
    ):
        """Initialize strategy with initial parameters."""
        if self.initialized:
            return
        
        # Get initial ranges and liquidities
        ranges, liquidities = self.calculate_range(price_data, initial_price, portfolio_value)
        
        self.current_ranges = ranges
        self.current_liquidities = liquidities
        self.initialized = True
        
        logger.info(f"Initialized {self.name} with {len(ranges)} positions")
    
    def update(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        force_update: bool = False
    ) -> bool:
        """
        Update strategy and determine if rebalancing is needed.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            force_update: Force update regardless of conditions
            
        Returns:
            True if rebalancing is needed
        """
        if not self.initialized:
            self.initialize(current_price, portfolio_value, price_data)
            return True
        
        # Calculate new ranges and liquidities
        new_ranges, new_liquidities = self.calculate_range(price_data, current_price, portfolio_value)
        
        # Check if rebalancing is needed
        if force_update or self._should_rebalance(new_ranges, new_liquidities):
            self.current_ranges = new_ranges
            self.current_liquidities = new_liquidities
            self.rebalance_count += 1
            self.last_update = price_data.index[-1] if len(price_data) > 0 else None
            
            logger.debug(f"{self.name} rebalancing triggered (count: {self.rebalance_count})")
            return True
        
        return False
    
    def _should_rebalance(
        self,
        new_ranges: List[Tuple[float, float]],
        new_liquidities: List[float]
    ) -> bool:
        """
        Determine if rebalancing is needed by comparing new and current positions.
        
        Args:
            new_ranges: New position ranges
            new_liquidities: New liquidity amounts
            
        Returns:
            True if rebalancing is needed
        """
        if len(new_ranges) != len(self.current_ranges):
            return True
        
        if len(new_liquidities) != len(self.current_liquidities):
            return True
        
        # Check if ranges have changed significantly
        for i, (new_range, current_range) in enumerate(zip(new_ranges, self.current_ranges)):
            new_lower, new_upper = new_range
            current_lower, current_upper = current_range
            
            # Check if range boundaries have moved by more than 1%
            lower_change = abs(new_lower - current_lower) / current_lower
            upper_change = abs(new_upper - current_upper) / current_upper
            
            if lower_change > 0.01 or upper_change > 0.01:
                return True
        
        # Check if liquidity has changed significantly
        for i, (new_liq, current_liq) in enumerate(zip(new_liquidities, self.current_liquidities)):
            liq_change = abs(new_liq - current_liq) / current_liq
            if liq_change > 0.05:  # 5% threshold
                return True
        
        return False
    
    def get_current_positions(self) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Get current position ranges and liquidities."""
        return self.current_ranges, self.current_liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "initialized": self.initialized,
            "rebalance_count": self.rebalance_count,
            "total_fees_earned": self.total_fees_earned,
            "current_positions": len(self.current_ranges),
            "last_update": self.last_update
        }
    
    def reset(self):
        """Reset strategy state."""
        self.initialized = False
        self.current_ranges = []
        self.current_liquidities = []
        self.last_update = None
        self.rebalance_count = 0
        self.total_fees_earned = 0.0
    
    def add_fees(self, fees: float):
        """Add earned fees to strategy."""
        self.total_fees_earned += fees
    
    def _validate_parameters(self, required_params: List[str]):
        """Validate that required parameters are present."""
        missing_params = [param for param in required_params if param not in self.parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters for {self.name}: {missing_params}")
    
    def _get_parameter(self, name: str, default=None):
        """Get parameter value with default fallback."""
        return self.parameters.get(name, default)
    
    def _calculate_sma(self, data: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return data.iloc[-1] if len(data) > 0 else 0
        return data.rolling(window=period).mean().iloc[-1]
    
    def _calculate_ema(self, data: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return data.iloc[-1] if len(data) > 0 else 0
        return data.ewm(span=period).mean().iloc[-1]
    
    def _calculate_std(self, data: pd.Series, period: int) -> float:
        """Calculate rolling standard deviation."""
        if len(data) < period:
            return 0
        return data.rolling(window=period).std().iloc[-1]
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Calculate Average True Range."""
        if len(high) < period:
            return 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return true_range.rolling(window=period).mean().iloc[-1]
