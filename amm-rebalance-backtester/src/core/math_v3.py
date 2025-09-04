"""
Uniswap V3 mathematical calculations for CLMM positions.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Uniswap V3 constants
Q96 = 2**96
Q192 = 2**192
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

class UniswapV3Math:
    """Uniswap V3 mathematical calculations."""
    
    @staticmethod
    def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
        """Convert sqrt price in X96 format to decimal price."""
        return (sqrt_price_x96 / Q96) ** 2
    
    @staticmethod
    def price_to_sqrt_price_x96(price: float) -> int:
        """Convert decimal price to sqrt price in X96 format."""
        return int(np.sqrt(price) * Q96)
    
    @staticmethod
    def tick_to_sqrt_price_x96(tick: int) -> int:
        """Convert tick to sqrt price in X96 format."""
        if tick < MIN_TICK or tick > MAX_TICK:
            raise ValueError(f"Tick {tick} out of range [{MIN_TICK}, {MAX_TICK}]")
        
        return int(1.0001 ** (tick / 2) * Q96)
    
    @staticmethod
    def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
        """Convert sqrt price in X96 format to tick."""
        if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 > MAX_SQRT_RATIO:
            raise ValueError(f"Sqrt price {sqrt_price_x96} out of valid range")
        
        return int(np.log((sqrt_price_x96 / Q96) ** 2) / np.log(1.0001))
    
    @staticmethod
    def get_amount0_for_liquidity(
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        liquidity: int
    ) -> int:
        """Calculate amount0 for given liquidity and price range."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_a_x96 == sqrt_price_b_x96:
            return 0
        
        # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) / (sqrt_price_a_x96 * sqrt_price_b_x96)
        amount0 = liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) / (sqrt_price_a_x96 * sqrt_price_b_x96)
        return int(amount0)
    
    @staticmethod
    def get_amount1_for_liquidity(
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        liquidity: int
    ) -> int:
        """Calculate amount1 for given liquidity and price range."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_a_x96 == sqrt_price_b_x96:
            return 0
        
        # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96)
        amount1 = liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) / Q96
        return int(amount1)
    
    @staticmethod
    def get_liquidity_for_amount0(
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        amount0: int
    ) -> int:
        """Calculate liquidity for given amount0 and price range."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_a_x96 == sqrt_price_b_x96:
            return 0
        
        # Formula: amount0 * (sqrt_price_a_x96 * sqrt_price_b_x96) / (sqrt_price_b_x96 - sqrt_price_a_x96)
        liquidity = amount0 * (sqrt_price_a_x96 * sqrt_price_b_x96) / (sqrt_price_b_x96 - sqrt_price_a_x96)
        return int(liquidity)
    
    @staticmethod
    def get_liquidity_for_amount1(
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        amount1: int
    ) -> int:
        """Calculate liquidity for given amount1 and price range."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_a_x96 == sqrt_price_b_x96:
            return 0
        
        # Formula: amount1 / (sqrt_price_b_x96 - sqrt_price_a_x96) * Q96
        liquidity = amount1 * Q96 / (sqrt_price_b_x96 - sqrt_price_a_x96)
        return int(liquidity)
    
    @staticmethod
    def get_liquidity_for_amounts(
        sqrt_price_current_x96: int,
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        amount0: int,
        amount1: int
    ) -> int:
        """Calculate optimal liquidity for given amounts and price range."""
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_current_x96 <= sqrt_price_a_x96:
            # Current price below range, only token0
            return UniswapV3Math.get_liquidity_for_amount0(sqrt_price_a_x96, sqrt_price_b_x96, amount0)
        elif sqrt_price_current_x96 >= sqrt_price_b_x96:
            # Current price above range, only token1
            return UniswapV3Math.get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount1)
        else:
            # Current price in range, calculate both
            liquidity0 = UniswapV3Math.get_liquidity_for_amount0(sqrt_price_current_x96, sqrt_price_b_x96, amount0)
            liquidity1 = UniswapV3Math.get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_current_x96, amount1)
            return min(liquidity0, liquidity1)
    
    @staticmethod
    def calculate_position_value(
        sqrt_price_current_x96: int,
        sqrt_price_a_x96: int,
        sqrt_price_b_x96: int,
        liquidity: int,
        price: float
    ) -> Tuple[float, float]:
        """
        Calculate current position value in both tokens.
        
        Returns:
            Tuple of (amount0, amount1)
        """
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
        
        if sqrt_price_current_x96 <= sqrt_price_a_x96:
            # Only token0
            amount0 = UniswapV3Math.get_amount0_for_liquidity(
                sqrt_price_a_x96, sqrt_price_b_x96, liquidity
            ) / 1e18  # Assuming 18 decimals
            return amount0, 0.0
        elif sqrt_price_current_x96 >= sqrt_price_b_x96:
            # Only token1
            amount1 = UniswapV3Math.get_amount1_for_liquidity(
                sqrt_price_a_x96, sqrt_price_b_x96, liquidity
            ) / 1e18  # Assuming 18 decimals
            return 0.0, amount1
        else:
            # Both tokens
            amount0 = UniswapV3Math.get_amount0_for_liquidity(
                sqrt_price_current_x96, sqrt_price_b_x96, liquidity
            ) / 1e18
            amount1 = UniswapV3Math.get_amount1_for_liquidity(
                sqrt_price_a_x96, sqrt_price_current_x96, liquidity
            ) / 1e18
            return amount0, amount1
    
    @staticmethod
    def calculate_fees_earned(
        fee_growth_global0_x128: int,
        fee_growth_global1_x128: int,
        fee_growth_inside0_x128: int,
        fee_growth_inside1_x128: int,
        liquidity: int
    ) -> Tuple[float, float]:
        """
        Calculate fees earned by a position.
        
        Returns:
            Tuple of (fees0, fees1)
        """
        # Calculate fee growth outside the position
        fee_growth_outside0_x128 = fee_growth_global0_x128 - fee_growth_inside0_x128
        fee_growth_outside1_x128 = fee_growth_global1_x128 - fee_growth_inside1_x128
        
        # Calculate fees earned
        fees0 = (fee_growth_outside0_x128 * liquidity) / (2**128)
        fees1 = (fee_growth_outside1_x128 * liquidity) / (2**128)
        
        return fees0, fees1
    
    @staticmethod
    def calculate_optimal_width(
        volatility: float,
        k_multiplier: float = 1.0,
        min_width_pct: float = 10.0,
        max_width_pct: float = 500.0
    ) -> float:
        """
        Calculate optimal position width based on volatility.
        
        Args:
            volatility: Annualized volatility
            k_multiplier: Multiplier for volatility
            min_width_pct: Minimum width as percentage
            max_width_pct: Maximum width as percentage
            
        Returns:
            Optimal width as percentage
        """
        # Convert annualized volatility to position width
        # Higher volatility -> wider position
        optimal_width = k_multiplier * volatility * 100
        
        # Apply bounds
        optimal_width = max(min_width_pct, min(max_width_pct, optimal_width))
        
        return optimal_width
    
    @staticmethod
    def calculate_price_deviation(
        current_price: float,
        target_price: float
    ) -> float:
        """Calculate price deviation as percentage."""
        return abs(current_price - target_price) / target_price * 100
    
    @staticmethod
    def should_rebalance(
        current_price: float,
        target_price: float,
        threshold_bps: float,
        cooldown_hours: int,
        last_rebalance: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> bool:
        """
        Determine if rebalancing is needed.
        
        Args:
            current_price: Current market price
            target_price: Target/center price
            threshold_bps: Rebalancing threshold in basis points
            cooldown_hours: Minimum hours between rebalances
            last_rebalance: Timestamp of last rebalance
            current_time: Current timestamp
            
        Returns:
            True if rebalancing is needed
        """
        # Check price deviation
        deviation_bps = abs(current_price - target_price) / target_price * 10000
        price_trigger = deviation_bps >= threshold_bps
        
        # Check cooldown
        time_trigger = True
        if last_rebalance is not None and current_time is not None:
            hours_since_rebalance = (current_time - last_rebalance) / 3600
            time_trigger = hours_since_rebalance >= cooldown_hours
        
        return price_trigger and time_trigger
