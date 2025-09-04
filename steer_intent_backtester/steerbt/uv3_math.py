"""
CLMM position valuation using Uniswap V3 math formulas.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Uniswap V3 constants
Q96 = 2**96
Q192 = 2**192

def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    """
    Convert sqrt price in X96 format to decimal price.
    
    Args:
        sqrt_price_x96: Square root of price in X96 format
        
    Returns:
        Decimal price
    """
    return (sqrt_price_x96 / Q96) ** 2

def price_to_sqrt_price_x96(price: float) -> int:
    """
    Convert decimal price to sqrt price in X96 format.
    
    Args:
        price: Decimal price
        
    Returns:
        Square root of price in X96 format
    """
    return int(np.sqrt(price) * Q96)

def get_amount0_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int
) -> int:
    """
    Calculate amount0 for given liquidity and price range.
    
    Args:
        sqrt_price_a_x96: Square root of lower price in X96 format
        sqrt_price_b_x96: Square root of upper price in X96 format  
        liquidity: Liquidity amount
        
    Returns:
        Amount of token0
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    # Use float division to maintain precision
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    if diff_x96 == 0:
        return 0
    
    # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) / (sqrt_price_a_x96 * sqrt_price_b_x96)
    amount0_float = liquidity * (diff_x96 / Q96) / ((sqrt_price_a_x96 / Q96) * (sqrt_price_b_x96 / Q96))
    
    return int(amount0_float)

def get_amount1_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int
) -> int:
    """
    Calculate amount1 for given liquidity and price range.
    
    Args:
        sqrt_price_a_x96: Square root of lower price in X96 format
        sqrt_price_b_x96: Square root of upper price in X96 format
        liquidity: Liquidity amount
        
    Returns:
        Amount of token1
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    # Use float division to maintain precision
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    if diff_x96 == 0:
        return 0
    
    # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96)
    amount1_float = liquidity * (diff_x96 / Q96)
    
    return int(amount1_float)

def get_liquidity_for_amount0(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    amount0: int
) -> int:
    """
    Calculate liquidity for given amount0 and price range.
    
    Args:
        sqrt_price_a_x96: Square root of lower price in X96 format
        sqrt_price_b_x96: Square root of upper price in X96 format
        amount0: Amount of token0
        
    Returns:
        Liquidity amount
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    if diff_x96 == 0:
        return 0
    
    # Use float division to maintain precision
    # Formula: amount0 * (sqrt_price_a_x96 * sqrt_price_b_x96) / (sqrt_price_b_x96 - sqrt_price_a_x96)
    liquidity_float = (amount0 * (sqrt_price_a_x96 / Q96) * (sqrt_price_b_x96 / Q96)) / (diff_x96 / Q96)
    
    # Convert back to integer
    return int(liquidity_float)

def get_liquidity_for_amount1(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    amount1: int
) -> int:
    """
    Calculate liquidity for given amount1 and price range.
    
    Args:
        sqrt_price_a_x96: Square root of lower price in X96 format
        sqrt_price_b_x96: Square root of upper price in X96 format
        amount1: Amount of token1
        
    Returns:
        Liquidity amount
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    # Convert to float to avoid integer division precision issues
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    if diff_x96 == 0:
        return 0
    
    # Use float division to maintain precision
    liquidity_float = amount1 / (diff_x96 / Q96)
    
    # Convert back to integer
    return int(liquidity_float)

def get_amounts_for_liquidity(
    sqrt_price_x96: int,
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int
) -> Tuple[int, int]:
    """
    Calculate amounts of both tokens for given liquidity and current price.
    
    Args:
        sqrt_price_x96: Current square root price in X96 format
        sqrt_price_a_x96: Square root of lower price in X96 format
        sqrt_price_b_x96: Square root of upper price in X96 format
        liquidity: Liquidity amount
        
    Returns:
        Tuple of (amount0, amount1)
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    amount0 = 0
    amount1 = 0
    
    if sqrt_price_x96 <= sqrt_price_a_x96:
        # Price is below range - all liquidity in token0
        amount0 = get_amount0_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
    elif sqrt_price_x96 >= sqrt_price_b_x96:
        # Price is above range - all liquidity in token1
        amount1 = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
    else:
        # Price is in range - split between tokens
        amount0 = get_amount0_for_liquidity(sqrt_price_x96, sqrt_price_b_x96, liquidity)
        amount1 = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_x96, liquidity)
    
    return amount0, amount1

def calculate_position_value(
    price: float,
    lower_price: float,
    upper_price: float,
    liquidity: float,
    fee_tier_bps: int = 500
) -> Tuple[float, float, float]:
    """
    Calculate position value and amounts for given parameters.
    
    Args:
        price: Current price
        lower_price: Lower price bound
        upper_price: Upper price bound
        liquidity: Liquidity amount
        fee_tier_bps: Fee tier in basis points (default: 0.05%)
        
    Returns:
        Tuple of (amount0, amount1, total_value_usd)
    """
    # Convert to X96 format
    sqrt_price_x96 = price_to_sqrt_price_x96(price)
    sqrt_lower_x96 = price_to_sqrt_price_x96(lower_price)
    sqrt_upper_x96 = price_to_sqrt_price_x96(upper_price)
    
    # Calculate amounts
    amount0, amount1 = get_amounts_for_liquidity(
        sqrt_price_x96, sqrt_lower_x96, sqrt_upper_x96, int(liquidity)
    )
    
    # Convert from X96 format
    amount0_decimal = amount0 / Q96
    amount1_decimal = amount1 / Q96
    
    # Calculate total value in USD terms (assuming token1 is USD)
    total_value = amount0_decimal * price + amount1_decimal
    
    return amount0_decimal, amount1_decimal, total_value

def calculate_fees_earned(
    volume_in_range: float,
    liquidity_share: float,
    fee_tier_bps: int
) -> float:
    """
    Calculate fees earned for a given volume and liquidity share.
    
    Args:
        volume_in_range: Trading volume within the position's price range
        liquidity_share: Share of total liquidity in the range
        fee_tier_bps: Fee tier in basis points
        
    Returns:
        Fees earned in USD
    """
    fee_rate = fee_tier_bps / 10000.0
    return volume_in_range * liquidity_share * fee_rate

def calculate_impermanent_loss(
    initial_price: float,
    current_price: float,
    initial_amount0: float,
    initial_amount1: float
) -> float:
    """
    Calculate impermanent loss for a position.
    
    Args:
        initial_price: Price when position was opened
        current_price: Current price
        initial_amount0: Initial amount of token0
        initial_amount1: Initial amount of token1
        
    Returns:
        Impermanent loss as a percentage
    """
    # Value if held
    held_value = initial_amount0 * current_price + initial_amount1
    
    # Value if rebalanced to 50:50 at current price
    total_value = initial_amount0 * initial_price + initial_amount1
    current_amount0 = total_value / (2 * current_price)
    current_amount1 = total_value / 2
    
    rebalanced_value = current_amount0 * current_price + current_amount1
    
    # Impermanent loss
    il = (rebalanced_value - held_value) / held_value
    
    return il

def calculate_lvr_proxy(
    hodl_50_50_value: float,
    clmm_no_fee_value: float
) -> float:
    """
    Calculate LVR (Loss-Versus-Rebalancing) proxy.
    
    Args:
        hodl_50_50_value: Value of HODL 50:50 strategy
        clmm_no_fee_value: Value of CLMM position without fees
        
    Returns:
        LVR proxy value
    """
    return hodl_50_50_value - clmm_no_fee_value
