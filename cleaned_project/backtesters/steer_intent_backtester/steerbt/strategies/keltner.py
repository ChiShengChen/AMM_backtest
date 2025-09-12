"""
Keltner Channels strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
import logging

logger = logging.getLogger(__name__)

class KeltnerStrategy(BaseStrategy):
    """
    Keltner Channels strategy using EMA ± m*ATR for position ranges.
    
    Parameters:
    - n: Lookback period for EMA calculation
    - m: ATR multiplier
    - curve_type: Liquidity distribution curve type
    - max_positions: Maximum number of positions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["n", "m"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.n = self._get_parameter("n")  # Lookback period for EMA
        self.m = self._get_parameter("m")  # ATR multiplier
        
        # Optional parameters
        self.curve_type = self._get_parameter("curve_type", "uniform")
        self.curve_params = self._get_parameter("curve_params", {})
        self.max_positions = self._get_parameter("max_positions", 1)
        
        # Initialize curve
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Strategy state
        self.last_ema = None
        self.last_atr = None
        self.last_upper_channel = None
        self.last_lower_channel = None
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using Keltner Channels.
        
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
        
        # Calculate Keltner Channels
        close_prices = price_data["close"]
        high_prices = price_data["high"]
        low_prices = price_data["low"]
        
        ema = self._calculate_ema(close_prices, self.n)
        atr = self._calculate_atr(high_prices, low_prices, close_prices, self.n)
        
        # Calculate channels
        upper_channel = ema + self.m * atr
        lower_channel = ema - self.m * atr
        
        # Store for reference
        self.last_ema = ema
        self.last_atr = atr
        self.last_upper_channel = upper_channel
        self.last_lower_channel = lower_channel
        
        # Calculate position ranges
        if self.max_positions == 1:
            # Single position using the channels
            ranges = [(lower_channel, upper_channel)]
            liquidities = [portfolio_value * 0.95]
        else:
            # Multiple positions using curve
            center_price = ema
            width_pct = ((upper_channel - lower_channel) / center_price) * 100
            
            distribution = self.curve.generate_distribution(
                center_price, width_pct, portfolio_value * 0.95
            )
            
            ranges = [(lower, upper) for lower, upper, _ in distribution]
            liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def get_channels_info(self) -> Dict[str, float]:
        """Get current Keltner Channels information."""
        return {
            "ema": self.last_ema,
            "atr": self.last_atr,
            "upper_channel": self.last_upper_channel,
            "lower_channel": self.last_lower_channel,
            "channel_width": (self.last_upper_channel - self.last_lower_channel) / self.last_ema * 100 if self.last_ema else 0
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "n": self.n,
            "m": self.m,
            "curve_type": self.curve_type,
            "max_positions": self.max_positions,
            "channels_info": self.get_channels_info()
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.last_ema = None
        self.last_atr = None
        self.last_upper_channel = None
        self.last_lower_channel = None
