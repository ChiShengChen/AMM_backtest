"""
Dynamic volatility strategy - volatility-adaptive width with price triggers.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .base import BaseStrategy

logger = logging.getLogger(__name__)

class DynamicVolatilityStrategy(BaseStrategy):
    """Dynamic volatility strategy with adaptive position width."""
    
    def __init__(self, vol_estimator: str = "ewma", k_width: float = 1.5, 
                 price_deviation_bps: float = 50.0, rebalance_cooldown_hours: int = 24,
                 vol_window_hours: int = 168):
        """
        Initialize dynamic volatility strategy.
        
        Args:
            vol_estimator: Volatility estimator ("ewma" or "rs")
            k_width: Multiplier for volatility-based width
            price_deviation_bps: Rebalancing threshold in basis points
            rebalance_cooldown_hours: Minimum hours between rebalances
            vol_window_hours: Window for volatility estimation
        """
        super().__init__(
            vol_estimator=vol_estimator,
            k_width=k_width,
            price_deviation_bps=price_deviation_bps,
            rebalance_cooldown_hours=rebalance_cooldown_hours,
            vol_window_hours=vol_window_hours
        )
        
        self.vol_estimator = vol_estimator
        self.k_width = k_width
        self.price_deviation_bps = price_deviation_bps
        self.rebalance_cooldown_hours = rebalance_cooldown_hours
        self.vol_window_hours = vol_window_hours
    
    def _estimate_volatility(self, price_data: pd.DataFrame) -> float:
        """Estimate volatility using specified method."""
        if len(price_data) < 2:
            return 0.1  # Default volatility
        
        if self.vol_estimator == "ewma":
            # Exponential Weighted Moving Average
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
            vol = returns.ewm(span=min(24, len(returns))).std().iloc[-1]
        elif self.vol_estimator == "rs":
            # Range-based volatility (Parkinson)
            high_low = np.log(price_data['high'] / price_data['low'])
            vol = np.sqrt(np.mean(high_low ** 2) / (4 * np.log(2)))
        else:
            # Simple standard deviation
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
            vol = returns.std()
        
        # Annualize (assuming hourly data)
        annualized_vol = vol * np.sqrt(24 * 365)
        return max(annualized_vol, 0.05)  # Minimum 5% volatility
    
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate dynamic position ranges based on volatility.
        
        Width = k * volatility * current_price
        """
        # Estimate volatility
        volatility = self._estimate_volatility(price_data)
        
        # Calculate dynamic width
        width_pct = self.k_width * volatility * 100
        width_pct = max(width_pct, 10.0)  # Minimum 10% width
        width_pct = min(width_pct, 200.0)  # Maximum 200% width
        
        # Calculate width in price terms
        width = current_price * (width_pct / 100.0)
        
        # Calculate range bounds
        lower_price = current_price - width / 2
        upper_price = current_price + width / 2
        
        # Ensure positive prices
        lower_price = max(lower_price, current_price * 0.01)
        
        # Single position with full portfolio value
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value]
        
        logger.debug(f"Volatility: {volatility:.4f}, Width: {width_pct:.2f}%, Range: [{lower_price:.2f}, {upper_price:.2f}]")
        
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
            'strategy_type': 'Dynamic Volatility',
            'description': 'Volatility-adaptive width with price deviation triggers',
            'vol_estimator': self.vol_estimator,
            'k_width': self.k_width,
            'price_deviation_bps': self.price_deviation_bps,
            'rebalance_cooldown_hours': self.rebalance_cooldown_hours,
            'vol_window_hours': self.vol_window_hours,
            'rebalancing_frequency': 'High',
            'expected_performance': 'Best APR/MDD balance, adaptive to market conditions'
        }
