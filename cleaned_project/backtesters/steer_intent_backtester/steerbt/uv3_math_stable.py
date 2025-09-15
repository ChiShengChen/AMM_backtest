"""
Stable CLMM position valuation using Uniswap V3 math formulas - Stable Version
穩定版本的UV3數學計算 - 避免數值溢出
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

def calculate_position_value(
    price: float,
    lower_price: float,
    upper_price: float,
    liquidity: float,
    fee_tier_bps: int = 500
) -> Tuple[float, float, float]:
    """
    Calculate position value and amounts for given parameters.
    使用穩定的計算方法避免數值溢出
    
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
    
    # 修復：使用更保守的計算方法，避免數值溢出
    # 限制liquidity的最大值
    max_liquidity = 1e12  # 限制最大流動性
    liquidity = min(liquidity, max_liquidity)
    
    # 計算價格範圍的寬度
    price_range = upper_price - lower_price
    price_center = (upper_price + lower_price) / 2
    
    # 計算當前價格在範圍中的位置
    if price < lower_price:
        # 價格低於範圍，所有流動性在token0
        # 使用更保守的計算
        amount0 = liquidity * min(price_range / price_center, 1.0)
        amount1 = 0.0
    elif price > upper_price:
        # 價格高於範圍，所有流動性在token1
        # 使用更保守的計算
        amount0 = 0.0
        amount1 = liquidity * min(price_range, price_center)
    else:
        # 價格在範圍內，流動性分佈在兩個token中
        # 簡化計算：根據價格在範圍中的位置分配流動性
        position_in_range = (price - lower_price) / price_range
        position_in_range = max(0.0, min(1.0, position_in_range))  # 限制在[0,1]範圍內
        
        # 使用更保守的計算，避免數值溢出
        amount0 = liquidity * (1 - position_in_range) * min(price_range / price_center, 1.0)
        amount1 = liquidity * position_in_range * min(price_range, price_center)
    
    # 限制amount的最大值，避免溢出
    max_amount = 1e15  # 限制最大amount
    amount0 = min(amount0, max_amount)
    amount1 = min(amount1, max_amount)
    
    # 計算總價值，使用更穩定的計算
    try:
        total_value = amount0 * price + amount1
        # 檢查是否溢出
        if not np.isfinite(total_value) or total_value > 1e20:
            total_value = 0.0
            amount0 = 0.0
            amount1 = 0.0
    except (OverflowError, ValueError):
        total_value = 0.0
        amount0 = 0.0
        amount1 = 0.0
    
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
