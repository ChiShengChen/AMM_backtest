"""
Dynamic inventory strategy - inventory skew + fee density triggers.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .base import BaseStrategy

logger = logging.getLogger(__name__)

class DynamicInventoryStrategy(BaseStrategy):
    """Dynamic inventory strategy with inventory skew and fee density triggers."""
    
    def __init__(self, skew_threshold_pct: float = 15.0, fee_density_window_h: int = 24,
                 reinvest_frequency_h: int = 48, target_skew: float = 0.5):
        """
        Initialize dynamic inventory strategy.
        
        Args:
            skew_threshold_pct: Inventory skew threshold as percentage
            fee_density_window_h: Window for fee density calculation
            reinvest_frequency_h: Frequency for fee reinvestment
            target_skew: Target inventory ratio (token0:token1)
        """
        super().__init__(
            skew_threshold_pct=skew_threshold_pct,
            fee_density_window_h=fee_density_window_h,
            reinvest_frequency_h=reinvest_frequency_h,
            target_skew=target_skew
        )
        
        self.skew_threshold_pct = skew_threshold_pct
        self.fee_density_window_h = fee_density_window_h
        self.reinvest_frequency_h = reinvest_frequency_h
        self.target_skew = target_skew
        self.last_reinvest = None
    
    def _calculate_inventory_skew(self, current_price: float, portfolio_value: float) -> float:
        """Calculate current inventory skew."""
        # Simplified calculation - in real implementation, this would use actual token amounts
        # For now, assume 50/50 split and calculate deviation
        target_value = portfolio_value / 2
        current_skew = 0.5  # Placeholder - would be calculated from actual positions
        
        return abs(current_skew - self.target_skew) * 100
    
    def _calculate_fee_density(self, price_data: pd.DataFrame, pool_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate fee density over the specified window."""
        if pool_data is None or len(pool_data) < 2:
            # Use volume as proxy for fee generation
            recent_volume = price_data['volume'].tail(self.fee_density_window_h).sum()
            return recent_volume / self.fee_density_window_h
        else:
            # Use actual pool data if available
            recent_pool = pool_data.tail(self.fee_density_window_h)
            if 'volume_token0' in recent_pool.columns and 'volume_token1' in recent_pool.columns:
                total_volume = recent_pool['volume_token0'].sum() + recent_pool['volume_token1'].sum()
                return total_volume / self.fee_density_window_h
            else:
                # Fallback to price data volume
                recent_volume = price_data['volume'].tail(self.fee_density_window_h).sum()
                return recent_volume / self.fee_density_window_h
    
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        pool_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate position ranges based on inventory skew and fee density.
        """
        # Calculate inventory skew
        skew = self._calculate_inventory_skew(current_price, portfolio_value)
        
        # Calculate fee density
        fee_density = self._calculate_fee_density(price_data, pool_data)
        
        # Adjust width based on skew and fee density
        base_width_pct = 30.0  # Base width
        
        # Wider positions when skew is high (need more room for rebalancing)
        skew_adjustment = min(skew / self.skew_threshold_pct, 2.0)
        
        # Wider positions when fee density is high (more active trading)
        fee_adjustment = min(fee_density / 1000, 1.5)  # Normalize fee density
        
        # Calculate final width
        width_pct = base_width_pct * (1 + skew_adjustment + fee_adjustment)
        width_pct = max(width_pct, 20.0)  # Minimum 20% width
        width_pct = min(width_pct, 150.0)  # Maximum 150% width
        
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
        
        logger.debug(f"Skew: {skew:.2f}%, Fee Density: {fee_density:.2f}, Width: {width_pct:.2f}%")
        
        return ranges, liquidities
    
    def should_rebalance(
        self,
        current_price: float,
        current_time: datetime,
        price_data: pd.DataFrame,
        pool_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> bool:
        """
        Check if rebalancing is needed.
        
        Triggers on inventory skew, fee density, or cooldown expiration.
        """
        # Check cooldown
        if self.last_rebalance is not None:
            time_since_rebalance = current_time - self.last_rebalance
            if time_since_rebalance < timedelta(hours=6):  # More frequent rebalancing
                return False
        
        # Check inventory skew
        skew = self._calculate_inventory_skew(current_price, 100000)  # Placeholder portfolio value
        if skew > self.skew_threshold_pct:
            logger.info(f"Inventory skew trigger: {skew:.2f}% > {self.skew_threshold_pct}%")
            return True
        
        # Check fee density
        fee_density = self._calculate_fee_density(price_data, pool_data)
        if fee_density > 2000:  # High fee density threshold
            logger.info(f"Fee density trigger: {fee_density:.2f} > 2000")
            return True
        
        # Check if price is still within current range
        if self.current_ranges:
            lower_price, upper_price = self.current_ranges[0]
            if lower_price <= current_price <= upper_price:
                return False
        
        # Rebalance needed
        return True
    
    def should_reinvest_fees(self, current_time: datetime) -> bool:
        """Check if fees should be reinvested."""
        if self.last_reinvest is None:
            return True
        
        time_since_reinvest = current_time - self.last_reinvest
        return time_since_reinvest >= timedelta(hours=self.reinvest_frequency_h)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'strategy_type': 'Dynamic Inventory',
            'description': 'Inventory skew + fee density triggers with low-frequency reinvestment',
            'skew_threshold_pct': self.skew_threshold_pct,
            'fee_density_window_h': self.fee_density_window_h,
            'reinvest_frequency_h': self.reinvest_frequency_h,
            'target_skew': self.target_skew,
            'rebalancing_frequency': 'Medium-High',
            'expected_performance': 'Lowest MDD, smart inventory management'
        }
