"""
Channel Multiplier strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class ChannelMultiplierStrategy(BaseStrategy):
    """
    Channel Multiplier strategy with single symmetric percentage width around price.
    
    Parameters:
    - width_pct: Percentage width around current price
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["width_pct"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.width_pct = self._get_parameter("width_pct")
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate single position range with symmetric width.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities)
        """
        # Calculate range boundaries
        half_width = self.width_pct / 200  # Convert to decimal and divide by 2
        lower_price = current_price * (1 - half_width)
        upper_price = current_price * (1 + half_width)
        
        # Single position
        ranges = [(lower_price, upper_price)]
        liquidities = [portfolio_value * 0.95]  # 95% of portfolio value
        
        return ranges, liquidities
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "width_pct": self.width_pct
        })
        return info
