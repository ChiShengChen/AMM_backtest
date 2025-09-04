"""
Baseline fixed strategy - fixed width with fixed triggers.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .base import BaseStrategy

logger = logging.getLogger(__name__)

class BaselineFixedStrategy(BaseStrategy):
    """Baseline fixed strategy with fixed width and fixed price deviation triggers."""
    
    def __init__(self, width_pct: float = 50.0, price_deviation_bps: float = 50.0, 
                 rebalance_cooldown_hours: int = 24):
        """
        Initialize baseline fixed strategy.
        
        Args:
            width_pct: Position width as percentage of current price
            price_deviation_bps: Rebalancing threshold in basis points
            rebalance_cooldown_hours: Minimum hours between rebalances
        """
        super().__init__(
            width_pct=width_pct,
            price_deviation_bps=price_deviation_bps,
            rebalance_cooldown_hours=rebalance_cooldown_hours
        )
        
        self.width_pct = width_pct
        self.price_deviation_bps = price_deviation_bps
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
    
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate fixed position ranges.
        
        Creates a single position with fixed width centered around current price.
        """
        # Calculate width in price terms
        width = current_price * (self.width_pct / 100.0)
        
        # Calculate range bounds
        lower_price = current_price - width / 2
        upper_price = current_price + width / 2
        
        # Ensure positive prices
        lower_price = max(lower_price, current_price * 0.01)
        
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
        
        Triggers on price deviation or cooldown expiration.
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
            'strategy_type': 'Baseline Fixed',
            'description': 'Fixed width positions with fixed price deviation triggers',
            'width_pct': self.width_pct,
            'price_deviation_bps': self.price_deviation_bps,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'rebalancing_frequency': 'Medium',
            'expected_performance': 'Platform-style approach, balanced risk'
        }
