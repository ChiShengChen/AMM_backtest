"""
Base strategy class for AMM strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all AMM strategies."""
    
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.parameters = kwargs
        self.initialized = False
        
        # Strategy state
        self.current_ranges: List[Tuple[float, float]] = []
        self.current_liquidities: List[float] = []
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_count = 0
        
        # Performance tracking
        self.total_fees_earned = 0.0
        self.total_gas_cost = 0.0
        self.total_slippage_cost = 0.0
        
        # Position tracking
        self.position_history: List[Dict[str, Any]] = []
        self.rebalance_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate new position ranges and liquidities.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Tuple of (ranges, liquidities) where ranges is list of (lower, upper) 
            and liquidities is list of amounts
        """
        pass
    
    @abstractmethod
    def should_rebalance(
        self,
        current_price: float,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        Determine if rebalancing is needed.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            **kwargs: Additional parameters
            
        Returns:
            True if rebalancing is needed
        """
        pass
    
    def initialize(
        self,
        initial_price: float,
        portfolio_value: float,
        price_data: pd.DataFrame,
        **kwargs
    ):
        """Initialize strategy with initial parameters."""
        if self.initialized:
            return
        
        # Get initial ranges and liquidities
        ranges, liquidities = self.calculate_ranges(
            price_data, initial_price, portfolio_value, **kwargs
        )
        
        self.current_ranges = ranges
        self.current_liquidities = liquidities
        self.initialized = True
        
        # Record initial position
        self._record_position(initial_price, portfolio_value, ranges, liquidities)
        
        logger.info(f"Initialized {self.name} with {len(ranges)} positions")
    
    def update(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        current_time: datetime,
        portfolio_value: float,
        force_update: bool = False,
        **kwargs
    ) -> bool:
        """
        Update strategy and determine if rebalancing is needed.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            current_time: Current timestamp
            portfolio_value: Current portfolio value
            force_update: Force update regardless of conditions
            **kwargs: Additional parameters
            
        Returns:
            True if rebalancing is needed
        """
        if not self.initialized:
            self.initialize(current_price, portfolio_value, price_data, **kwargs)
            return True
        
        # Check if rebalancing is needed
        if force_update or self.should_rebalance(current_price, current_time, **kwargs):
            # Calculate new ranges and liquidities
            new_ranges, new_liquidities = self.calculate_ranges(
                price_data, current_price, portfolio_value, **kwargs
            )
            
            # Record rebalance
            self._record_rebalance(
                current_time, current_price, portfolio_value,
                self.current_ranges, self.current_liquidities,
                new_ranges, new_liquidities
            )
            
            # Update current state
            self.current_ranges = new_ranges
            self.current_liquidities = new_liquidities
            self.last_rebalance = current_time
            self.rebalance_count += 1
            
            logger.info(f"{self.name} rebalanced at {current_time}: {len(new_ranges)} positions")
            return True
        
        return False
    
    def get_current_position(self) -> Dict[str, Any]:
        """Get current position information."""
        return {
            'ranges': self.current_ranges,
            'liquidities': self.current_liquidities,
            'last_rebalance': self.last_rebalance,
            'rebalance_count': self.rebalance_count
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        return {
            'total_fees_earned': self.total_fees_earned,
            'total_gas_cost': self.total_gas_cost,
            'total_slippage_cost': self.total_slippage_cost,
            'rebalance_count': self.rebalance_count
        }
    
    def _record_position(
        self,
        price: float,
        portfolio_value: float,
        ranges: List[Tuple[float, float]],
        liquidities: List[float]
    ):
        """Record position state."""
        position_record = {
            'timestamp': datetime.now(),
            'price': price,
            'portfolio_value': portfolio_value,
            'ranges': ranges.copy(),
            'liquidities': liquidities.copy(),
            'strategy': self.name
        }
        self.position_history.append(position_record)
    
    def _record_rebalance(
        self,
        timestamp: datetime,
        price: float,
        portfolio_value: float,
        old_ranges: List[Tuple[float, float]],
        old_liquidities: List[float],
        new_ranges: List[Tuple[float, float]],
        new_liquidities: List[float]
    ):
        """Record rebalance event."""
        rebalance_record = {
            'timestamp': timestamp,
            'price': price,
            'portfolio_value': portfolio_value,
            'old_ranges': old_ranges.copy(),
            'old_liquidities': old_liquidities.copy(),
            'new_ranges': new_ranges.copy(),
            'new_liquidities': new_liquidities.copy(),
            'strategy': self.name
        }
        self.rebalance_history.append(rebalance_record)
    
    def add_fees(self, fees: float):
        """Add earned fees."""
        self.total_fees_earned += fees
    
    def add_gas_cost(self, gas_cost: float):
        """Add gas cost."""
        self.total_gas_cost += gas_cost
    
    def add_slippage_cost(self, slippage_cost: float):
        """Add slippage cost."""
        self.total_slippage_cost += slippage_cost
    
    def reset_performance(self):
        """Reset performance tracking."""
        self.total_fees_earned = 0.0
        self.total_gas_cost = 0.0
        self.total_slippage_cost = 0.0
        self.rebalance_count = 0
        self.position_history.clear()
        self.rebalance_history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'initialized': self.initialized,
            'current_ranges': self.current_ranges,
            'current_liquidities': self.current_liquidities,
            'last_rebalance': self.last_rebalance,
            'rebalance_count': self.rebalance_count,
            'performance': self.get_performance_metrics()
        }
