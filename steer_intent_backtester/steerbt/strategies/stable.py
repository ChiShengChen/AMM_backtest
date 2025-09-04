"""
Stable strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
import logging

logger = logging.getLogger(__name__)

class StableStrategy(BaseStrategy):
    """
    Stable strategy that computes a "peg" and opens multiple positions around it.
    
    Parameters:
    - peg_method: Method to compute peg (sma, ema, median, custom)
    - peg_period: Period for peg calculation
    - width_pct: Width around peg as percentage
    - curve_type: Liquidity distribution curve type (gaussian, linear)
    - bin_count: Number of bins/positions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["peg_method", "width_pct"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.peg_method = self._get_parameter("peg_method")
        self.peg_period = self._get_parameter("peg_period", 20)
        self.width_pct = self._get_parameter("width_pct")
        self.curve_type = self._get_parameter("curve_type", "gaussian")
        self.bin_count = self._get_parameter("bin_count", 5)
        
        # Optional parameters
        self.curve_params = self._get_parameter("curve_params", {})
        self.peg_offset = self._get_parameter("peg_offset", 0.0)  # Offset from calculated peg
        
        # Initialize curve
        self.curve_params["max_bins"] = self.bin_count
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Strategy state
        self.last_peg = None
        self.last_peg_timestamp = None
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges around computed peg.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities)
        """
        # Compute peg
        peg_price = self._compute_peg(price_data, current_price)
        
        # Store peg information
        self.last_peg = peg_price
        self.last_peg_timestamp = price_data.index[-1] if len(price_data) > 0 else None
        
        # Apply offset if specified
        if self.peg_offset != 0.0:
            peg_price = peg_price * (1 + self.peg_offset)
        
        # Generate position distribution using curve
        distribution = self.curve.generate_distribution(
            peg_price, self.width_pct, portfolio_value * 0.95
        )
        
        ranges = [(lower, upper) for lower, upper, _ in distribution]
        liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def _compute_peg(self, price_data: pd.DataFrame, current_price: float) -> float:
        """
        Compute peg price using specified method.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            
        Returns:
            Computed peg price
        """
        if len(price_data) < self.peg_period:
            return current_price
        
        close_prices = price_data["close"]
        
        if self.peg_method == "sma":
            return self._calculate_sma(close_prices, self.peg_period)
        
        elif self.peg_method == "ema":
            return self._calculate_ema(close_prices, self.peg_period)
        
        elif self.peg_method == "median":
            return close_prices.rolling(window=self.peg_period).median().iloc[-1]
        
        elif self.peg_method == "vwap":
            # Volume Weighted Average Price
            volume = price_data["volume"]
            vwap = (close_prices * volume).rolling(window=self.peg_period).sum() / volume.rolling(window=self.peg_period).sum()
            return vwap.iloc[-1]
        
        elif self.peg_method == "custom":
            # Custom peg calculation (e.g., based on external data)
            # For now, use SMA as fallback
            return self._calculate_sma(close_prices, self.peg_period)
        
        else:
            raise ValueError(f"Unknown peg method: {self.peg_method}")
    
    def get_peg_info(self) -> Dict[str, Any]:
        """Get current peg information."""
        return {
            "peg_price": self.last_peg,
            "peg_method": self.peg_method,
            "peg_period": self.peg_period,
            "peg_offset": self.peg_offset,
            "last_peg_timestamp": self.last_peg_timestamp
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "peg_method": self.peg_method,
            "peg_period": self.peg_period,
            "width_pct": self.width_pct,
            "curve_type": self.curve_type,
            "bin_count": self.bin_count,
            "peg_offset": self.peg_offset,
            "peg_info": self.get_peg_info()
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.last_peg = None
        self.last_peg_timestamp = None
