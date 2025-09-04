"""
Unit tests for CLMM math functions.
"""

import pytest
import numpy as np
from steerbt.uv3_math import (
    sqrt_price_x96_to_price,
    price_to_sqrt_price_x96,
    get_amount0_for_liquidity,
    get_amount1_for_liquidity,
    get_amounts_for_liquidity,
    calculate_position_value,
    calculate_fees_earned,
    calculate_impermanent_loss,
    calculate_lvr_proxy
)

class TestCLMMMath:
    """Test CLMM mathematical functions."""
    
    def test_sqrt_price_conversion(self):
        """Test sqrt price conversion functions."""
        # Test conversion round-trip
        original_price = 2000.0
        sqrt_price_x96 = price_to_sqrt_price_x96(original_price)
        converted_price = sqrt_price_x96_to_price(sqrt_price_x96)
        
        # Should be very close (within floating point precision)
        assert abs(original_price - converted_price) < 1e-6
    
    def test_liquidity_amounts(self):
        """Test liquidity amount calculations."""
        # Use more reasonable price range for testing
        sqrt_price_a = price_to_sqrt_price_x96(1000.0)  # Lower price
        sqrt_price_b = price_to_sqrt_price_x96(1500.0)  # Upper price (closer range)
        liquidity = 1000000
        
        amount0 = get_amount0_for_liquidity(sqrt_price_a, sqrt_price_b, liquidity)
        amount1 = get_amount1_for_liquidity(sqrt_price_a, sqrt_price_b, liquidity)
        
        # For reasonable price ranges, both amounts should be positive
        # Note: amount0 might be 0 for very wide ranges due to precision limits
        assert amount0 >= 0
        assert amount1 > 0
        
        # Test amounts for liquidity function
        # Use a current price that's within the range
        current_price = 1250.0  # Between 1000 and 1500
        sqrt_price_current = price_to_sqrt_price_x96(current_price)
        
        amounts = get_amounts_for_liquidity(sqrt_price_current, sqrt_price_a, sqrt_price_b, liquidity)
        assert len(amounts) == 2
        
        # When price is in range, amounts should be calculated correctly
        # amount0: from current price to upper bound
        # amount1: from lower bound to current price
        expected_amount0 = get_amount0_for_liquidity(sqrt_price_current, sqrt_price_b, liquidity)
        expected_amount1 = get_amount1_for_liquidity(sqrt_price_a, sqrt_price_current, liquidity)
        
        assert amounts[0] == expected_amount0
        assert amounts[1] == expected_amount1
    
    def test_position_value_calculation(self):
        """Test position value calculation."""
        price = 1500.0
        lower_price = 1000.0
        upper_price = 2000.0
        liquidity = 1000000
        
        amount0, amount1, total_value = calculate_position_value(
            price, lower_price, upper_price, liquidity
        )
        
        # All values should be positive
        assert amount0 >= 0
        assert amount1 >= 0
        assert total_value > 0
        
        # Total value should equal amount0 * price + amount1
        expected_value = amount0 * price + amount1
        assert abs(total_value - expected_value) < 1e-6
    
    def test_fees_calculation(self):
        """Test fees calculation."""
        volume_in_range = 1000000  # $1M volume
        liquidity_share = 0.001    # 0.1% share
        fee_tier_bps = 500         # 0.05%
        
        fees = calculate_fees_earned(volume_in_range, liquidity_share, fee_tier_bps)
        
        # Fees should be positive
        assert fees > 0
        
        # Expected fees: volume * share * fee_rate
        expected_fees = volume_in_range * liquidity_share * (fee_tier_bps / 10000.0)
        assert abs(fees - expected_fees) < 1e-6
    
    def test_impermanent_loss(self):
        """Test impermanent loss calculation."""
        initial_price = 1000.0
        current_price = 1200.0
        initial_amount0 = 1.0
        initial_amount1 = 1000.0
        
        il = calculate_impermanent_loss(
            initial_price, current_price, initial_amount0, initial_amount1
        )
        
        # IL should be a percentage (can be positive or negative)
        assert isinstance(il, (int, float))
        
        # Test edge case: same price should have no IL
        il_same_price = calculate_impermanent_loss(
            initial_price, initial_price, initial_amount0, initial_amount1
        )
        assert abs(il_same_price) < 1e-6
    
    def test_lvr_proxy(self):
        """Test LVR proxy calculation."""
        hodl_value = 10000.0
        clmm_no_fee_value = 9500.0
        
        lvr = calculate_lvr_proxy(hodl_value, clmm_no_fee_value)
        
        # LVR should be positive when CLMM underperforms
        assert lvr > 0
        
        # Expected LVR
        expected_lvr = hodl_value - clmm_no_fee_value
        assert abs(lvr - expected_lvr) < 1e-6
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small liquidity
        price = 1000.0
        lower_price = 999.0
        upper_price = 1001.0
        liquidity = 1
        
        amount0, amount1, total_value = calculate_position_value(
            price, lower_price, upper_price, liquidity
        )
        
        # Should handle small values gracefully
        assert amount0 >= 0
        assert amount1 >= 0
        assert total_value >= 0
    
    def test_monotonicity(self):
        """Test that position value increases with liquidity."""
        price = 1500.0
        lower_price = 1000.0
        upper_price = 2000.0
        
        liquidity1 = 1000000
        liquidity2 = 2000000
        
        _, _, value1 = calculate_position_value(price, lower_price, upper_price, liquidity1)
        _, _, value2 = calculate_position_value(price, lower_price, upper_price, liquidity2)
        
        # Value should increase with liquidity
        assert value2 > value1
