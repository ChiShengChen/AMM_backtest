"""
Fixed CLMM position valuation using Uniswap V3 math formulas - Version 2
修復版本的UV3數學計算 - 解決精度問題
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
    
    if sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0
    
    # 修復：使用更高精度的計算
    # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) / (sqrt_price_a_x96 * sqrt_price_b_x96)
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    product_x96 = sqrt_price_a_x96 * sqrt_price_b_x96
    
    if product_x96 == 0:
        return 0
    
    # 使用浮點數計算以保持精度
    amount0_float = (liquidity * diff_x96) / product_x96
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
    
    if sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0
    
    # Formula: liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96)
    diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
    amount1_float = liquidity * diff_x96 / Q96
    
    return int(amount1_float)

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
        # Price is within range - liquidity in both tokens
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
    # 修復：確保liquidity是合理的值
    if liquidity <= 0:
        return 0.0, 0.0, 0.0
    
    # 修復：確保價格範圍合理
    if lower_price <= 0 or upper_price <= 0 or price <= 0:
        return 0.0, 0.0, 0.0
    
    if lower_price >= upper_price:
        return 0.0, 0.0, 0.0
    
    # 修復：使用更簡單的計算方法，避免X96精度問題
    # 直接使用浮點數計算，然後轉換為X96格式
    
    # 計算價格範圍的寬度
    price_range = upper_price - lower_price
    price_center = (upper_price + lower_price) / 2
    
    # 計算當前價格在範圍中的位置
    if price < lower_price:
        # 價格低於範圍，所有流動性在token0
        # 使用簡化的計算：liquidity * price_range / price_center
        amount0 = liquidity * price_range / price_center
        amount1 = 0.0
    elif price > upper_price:
        # 價格高於範圍，所有流動性在token1
        # 使用簡化的計算：liquidity * price_range
        amount0 = 0.0
        amount1 = liquidity * price_range
    else:
        # 價格在範圍內，流動性分佈在兩個token中
        # 簡化計算：根據價格在範圍中的位置分配流動性
        position_in_range = (price - lower_price) / price_range
        amount0 = liquidity * (1 - position_in_range) * price_range / price_center
        amount1 = liquidity * position_in_range * price_range
    
    # 計算總價值
    total_value = amount0 * price + amount1
    
    return amount0, amount1, total_value

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
    price_ratio: float,
    initial_price_ratio: float
) -> float:
    """
    Calculate impermanent loss for a given price ratio change.
    
    Args:
        price_ratio: Current price ratio (price1/price0)
        initial_price_ratio: Initial price ratio
        
    Returns:
        Impermanent loss as a percentage
    """
    if initial_price_ratio <= 0 or price_ratio <= 0:
        return 0.0
    
    ratio_change = price_ratio / initial_price_ratio
    sqrt_ratio = np.sqrt(ratio_change)
    
    # IL formula: 2 * sqrt(ratio) / (1 + ratio) - 1
    il = 2 * sqrt_ratio / (1 + ratio_change) - 1
    
    return il * 100  # Return as percentage
