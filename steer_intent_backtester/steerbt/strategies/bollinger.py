"""
Bollinger Bands strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
import logging

logger = logging.getLogger(__name__)

class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands strategy using SMA ± k*std for position ranges.
    
    Parameters:
    - n: Lookback period for SMA calculation
    - k: Standard deviation multiplier
    - curve_type: Liquidity distribution curve type
    - max_positions: Maximum number of positions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["n", "k"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.n = self._get_parameter("n")  # Lookback period
        self.k = self._get_parameter("k")  # Standard deviation multiplier
        
        # Optional parameters
        self.curve_type = self._get_parameter("curve_type", "uniform")
        self.curve_params = self._get_parameter("curve_params", {})
        self.max_positions = self._get_parameter("max_positions", 1)
        self.dynamic_width = self._get_parameter("dynamic_width", True)
        
        # Initialize curve
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Strategy state
        self.last_sma = None
        self.last_std = None
        self.last_upper_band = None
        self.last_lower_band = None
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges using Bollinger Bands.
        
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
        
        # Calculate Bollinger Bands
        close_prices = price_data["close"]
        sma = self._calculate_sma(close_prices, self.n)
        std = self._calculate_std(close_prices, self.n)
        
        # Calculate bands
        upper_band = sma + self.k * std
        lower_band = sma - self.k * std
        
        # Store for reference
        self.last_sma = sma
        self.last_std = std
        self.last_upper_band = upper_band
        self.last_lower_band = lower_band
        
        # Calculate position ranges
        if self.max_positions == 1:
            # Single position using the bands
            ranges = [(lower_band, upper_band)]
            liquidities = [portfolio_value * 0.95]
        else:
            # Multiple positions using curve
            center_price = sma
            width_pct = ((upper_band - lower_band) / center_price) * 100
            
            distribution = self.curve.generate_distribution(
                center_price, width_pct, portfolio_value * 0.95
            )
            
            ranges = [(lower, upper) for lower, upper, _ in distribution]
            liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def get_bands_info(self) -> Dict[str, float]:
        """Get current Bollinger Bands information."""
        return {
            "sma": self.last_sma,
            "std": self.last_std,
            "upper_band": self.last_upper_band,
            "lower_band": self.last_lower_band,
            "band_width": (self.last_upper_band - self.last_lower_band) / self.last_sma * 100 if self.last_sma else 0
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "n": self.n,
            "k": self.k,
            "curve_type": self.curve_type,
            "max_positions": self.max_positions,
            "dynamic_width": self.dynamic_width,
            "bands_info": self.get_bands_info()
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.last_sma = None
        self.last_std = None
        self.last_upper_band = None
        self.last_lower_band = None
