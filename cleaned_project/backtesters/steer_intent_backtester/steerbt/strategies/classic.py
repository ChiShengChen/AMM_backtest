"""
Classic rebalancing strategy for CLMM backtesting.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..curves import CurveFactory
from ..triggers import TriggerManager, GapFromCenterTrigger, RangeInactiveTrigger, PercentDriftTrigger, ElapsedTimeTrigger
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class ClassicStrategy(BaseStrategy):
    """
    Classic rebalancing strategy with configurable width modes and placement strategies.
    
    Width modes:
    - percent: Width as percentage of current price
    - multiplier: Width as multiplier of base width
    - static_ticks: Fixed width in price ticks
    
    Placement modes:
    - center: Center around current price
    - recenter: Recenter when out of range
    - dynamic: Dynamic placement based on market conditions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate required parameters
        required_params = ["width_mode", "width_value", "placement_mode"]
        self._validate_parameters(required_params)
        
        # Strategy parameters
        self.width_mode = self._get_parameter("width_mode")
        self.width_value = self._get_parameter("width_value")
        self.placement_mode = self._get_parameter("placement_mode")
        
        # Optional parameters
        self.curve_type = self._get_parameter("curve_type", "uniform")
        self.curve_params = self._get_parameter("curve_params", {})
        self.max_positions = self._get_parameter("max_positions", 1)
        self.recenter_threshold = self._get_parameter("recenter_threshold", 0.05)  # 5%
        
        # Initialize curve
        self.curve = CurveFactory.create_curve(self.curve_type, **self.curve_params)
        
        # Initialize triggers
        self.trigger_manager = TriggerManager()
        self._setup_triggers()
    
    def _setup_triggers(self):
        """Setup rebalancing triggers."""
        # Gap from center trigger
        gap_trigger = GapFromCenterTrigger(gap_bps=self._get_parameter("gap_bps", 100))
        self.trigger_manager.add_trigger("gap_from_center", gap_trigger)
        
        # Range inactive trigger
        range_trigger = RangeInactiveTrigger()
        self.trigger_manager.add_trigger("range_inactive", range_trigger)
        
        # Percent drift trigger
        drift_trigger = PercentDriftTrigger(drift_threshold_pct=self._get_parameter("drift_threshold_pct", 5.0))
        self.trigger_manager.add_trigger("percent_drift", drift_trigger)
        
        # Time elapsed trigger
        time_trigger = ElapsedTimeTrigger(timedelta(hours=self._get_parameter("time_threshold_hours", 24)))
        self.trigger_manager.add_trigger("time_elapsed", time_trigger)
    
    def calculate_range(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges and liquidities based on strategy parameters.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (ranges, liquidities)
        """
        # Calculate width based on mode
        width_pct = self._calculate_width(current_price)
        
        # Calculate center price based on placement mode
        center_price = self._calculate_center_price(price_data, current_price)
        
        # Generate position distribution using curve
        if self.max_positions == 1:
            # Single position - still use curve for consistent liquidity scaling
            distribution = self.curve.generate_distribution(
                center_price, width_pct, portfolio_value * 0.95
            )
            
            # For single position, use the first (and only) bin
            if distribution:
                lower_price, upper_price, liquidity = distribution[0]
                ranges = [(lower_price, upper_price)]
                liquidities = [liquidity]
            else:
                # Fallback if curve generation fails
                lower_price = center_price * (1 - width_pct / 200)
                upper_price = center_price * (1 + width_pct / 200)
                ranges = [(lower_price, upper_price)]
                liquidities = [portfolio_value * 0.95 * 0.001]  # Apply scaling manually
        else:
            # Multiple positions using curve
            distribution = self.curve.generate_distribution(
                center_price, width_pct, portfolio_value * 0.95
            )
            
            ranges = [(lower, upper) for lower, upper, _ in distribution]
            liquidities = [liq for _, _, liq in distribution]
        
        return ranges, liquidities
    
    def _calculate_width(self, current_price: float) -> float:
        """Calculate position width based on width mode."""
        if self.width_mode == "percent":
            return self.width_value
        
        elif self.width_mode == "multiplier":
            base_width = 5.0  # Base width of 5%
            return base_width * self.width_value
        
        elif self.width_mode == "static_ticks":
            # Convert ticks to percentage (approximate)
            tick_size = 0.0001  # Approximate tick size
            width_in_ticks = self.width_value
            return (width_in_ticks * tick_size / current_price) * 100
        
        else:
            raise ValueError(f"Unknown width mode: {self.width_mode}")
    
    def _calculate_center_price(
        self,
        price_data: pd.DataFrame,
        current_price: float
    ) -> float:
        """Calculate center price based on placement mode."""
        if self.placement_mode == "center":
            return current_price
        
        elif self.placement_mode == "recenter":
            # Check if current price is outside current range
            if self.current_ranges:
                current_lower, current_upper = self.current_ranges[0]
                if current_price < current_lower or current_price > current_upper:
                    return current_price  # Recenter
                else:
                    # Keep current center
                    return (current_lower + current_upper) / 2
            else:
                return current_price
        
        elif self.placement_mode == "dynamic":
            # Use moving average as center
            if len(price_data) >= 20:
                return price_data["close"].rolling(window=20).mean().iloc[-1]
            else:
                return current_price
        
        else:
            raise ValueError(f"Unknown placement mode: {self.placement_mode}")
    
    def should_rebalance(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float
    ) -> bool:
        """
        Determine if rebalancing is needed based on triggers.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            True if rebalancing is needed
        """
        if not self.initialized:
            return True
        
        # Check triggers
        current_state = {
            "current_price": current_price,
            "position_center": (self.current_ranges[0][0] + self.current_ranges[0][1]) / 2 if self.current_ranges else current_price,
            "lower_price": self.current_ranges[0][0] if self.current_ranges else 0,
            "upper_price": self.current_ranges[0][1] if self.current_ranges else 0,
            "position_value": portfolio_value,
            "current_timestamp": price_data.index[-1] if len(price_data) > 0 else None
        }
        
        should_trigger, triggered_names = self.trigger_manager.should_trigger_any(current_state)
        
        if should_trigger:
            logger.debug(f"Classic strategy triggers activated: {triggered_names}")
        
        return should_trigger
    
    def update(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        force_update: bool = False
    ) -> bool:
        """
        Update strategy and check if rebalancing is needed.
        
        Args:
            price_data: Historical price data
            current_price: Current market price
            portfolio_value: Current portfolio value
            force_update: Force update regardless of conditions
            
        Returns:
            True if rebalancing is needed
        """
        if force_update or self.should_rebalance(price_data, current_price, portfolio_value):
            return super().update(price_data, current_price, portfolio_value, force_update)
        
        return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics."""
        info = super().get_strategy_info()
        info.update({
            "width_mode": self.width_mode,
            "width_value": self.width_value,
            "placement_mode": self.placement_mode,
            "curve_type": self.curve_type,
            "max_positions": self.max_positions,
            "trigger_stats": self.trigger_manager.get_trigger_stats()
        })
        return info
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.trigger_manager.reset_all()
