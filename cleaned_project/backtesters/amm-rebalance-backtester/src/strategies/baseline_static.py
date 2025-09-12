"""
Baseline static strategy - passive ultra-wide positions.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .base import BaseStrategy

logger = logging.getLogger(__name__)

class BaselineStaticStrategy(BaseStrategy):
    """Baseline static strategy with ultra-wide positions and minimal rebalancing."""
    
    def __init__(self, width_pct: float = 500.0, rebalance_cooldown_hours: int = 168):
        """
        Initialize baseline static strategy.
        
        Args:
            width_pct: Position width as percentage of current price
            rebalance_cooldown_hours: Minimum hours between rebalances
        """
        super().__init__(
            width_pct=width_pct,
            rebalance_cooldown_hours=rebalance_cooldown_hours
        )
        
        self.width_pct = width_pct
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
    
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate static position ranges.
        
        Creates a single ultra-wide position centered around current price.
        """
        # Calculate width in price terms
        width = current_price * (self.width_pct / 100.0)
        
        # Calculate range bounds
        lower_price = current_price - width / 2
        upper_price = current_price + width / 2
        
        # Ensure positive prices
        lower_price = max(lower_price, current_price * 0.01)  # At least 1% of current price
        
        # Single position with full portfolio value
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value]
        
        return ranges, liquidities
    
    def should_rebalance(
        self,
        current_price: float,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        Check if rebalancing is needed.
        
        Only rebalances after cooldown period or if price moves outside range.
        """
        # Check cooldown
        if self.last_rebalance is not None:
            time_since_rebalance = current_time - self.last_rebalance
            if time_since_rebalance < timedelta(hours=self.rebalance_cooldown_hours):
                return False
        
        # Check if price is still within current range
        if self.current_ranges:
            lower_price, upper_price = self.current_ranges[0]
            if lower_price <= current_price <= upper_price:
                return False
        
        # Rebalance needed
        return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'strategy_type': 'Baseline Static',
            'description': 'Ultra-wide passive positions with minimal rebalancing',
            'width_pct': self.width_pct,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'rebalancing_frequency': 'Very Low',
            'expected_performance': 'Lower fees, higher IL risk'
        }
