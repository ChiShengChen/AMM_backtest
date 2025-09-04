"""
Donchian Channels strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
import logging

logger = logging.getLogger(__name__)

class DonchianStrategy(BaseStrategy):
    """
    Donchian Channels strategy using highest high and lowest low over N periods.
    
    Parameters:
    - n: Lookback period for high/low calculation
    - width_multiplier: Optional multiplier for channel width
    - curve_type: Liquidity distribution curve type
    - max_positions: Maximum number of positions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["n"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.n = self._get_parameter("n")  # Lookback period
        self.width_multiplier = self._get_parameter("width_multiplier", 1.0)
        
        # Optional parameters
        self.curve_type = self._get_parameter("curve_type", "uniform")
        self.curve_params = self._get_parameter("curve_params", {})
        self.max_positions = self._get_parameter("max_positions", 1)
        
        # Initialize curve
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Strategy state
        self.last_highest_high = None
        self.last_lowest_low = None
        self.last_center = None
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using Donchian Channels.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities)
        """
        if len(price_data) < self.n:
            # Not enough data, use simple range around current price
            width_pct = 5.0  # Default 5% width
            lower_price = current_price * (1 - width_pct / 200)
            upper_price = current_price * (1 + width_pct / 200)
            
            ranges = [(lower_price, upper_price)]
            liquidities = [portfolio_value * 0.95]
            
            return ranges, liquidities
        
        # Calculate Donchian Channels
        high_prices = price_data["high"]
        low_prices = price_data["low"]
        
        highest_high = high_prices.rolling(window=self.n).max().iloc[-1]
        lowest_low = low_prices.rolling(window=self.n).min().iloc[-1]
        
        # Apply width multiplier if specified
        if self.width_multiplier != 1.0:
            center = (highest_high + lowest_low) / 2
            width = (highest_high - lowest_low) * self.width_multiplier
            highest_high = center + width / 2
            lowest_low = center - width / 2
        
        # Store for reference
        self.last_highest_high = highest_high
        self.last_lowest_low = lowest_low
        self.last_center = (highest_high + lowest_low) / 2
        
        # Calculate position ranges
        if self.max_positions == 1:
            # Single position using the channels
            ranges = [(lowest_low, highest_high)]
            liquidities = [portfolio_value * 0.95]
        else:
            # Multiple positions using curve
            center_price = self.last_center
            width_pct = ((highest_high - lowest_low) / center_price) * 100
            
            distribution = self.curve.generate_distribution(
                center_price, width_pct, portfolio_value * 0.95
            )
            
            ranges = [(lower, upper) for lower, upper, _ in distribution]
            liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def get_channels_info(self) -> Dict[str, float]:
        """Get current Donchian Channels information."""
        return {
            "highest_high": self.last_highest_high,
            "lowest_low": self.last_lowest_low,
            "center": self.last_center,
            "channel_width": (self.last_highest_high - self.last_lowest_low) / self.last_center * 100 if self.last_center else 0
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "n": self.n,
            "width_multiplier": self.width_multiplier,
            "curve_type": self.curve_type,
            "max_positions": self.max_positions,
            "channels_info": self.get_channels_info()
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.last_highest_high = None
        self.last_lowest_low = None
        self.last_center = None
