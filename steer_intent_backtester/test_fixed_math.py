#!/usr/bin/env python3
"""
Test script to verify fixed Uniswap V3 math functions.
"""

import sys
import os
import numpy as np

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.uv3_math import *

def test_fixed_functions():
    """Test the fixed liquidity calculation functions."""
    print("🧪 Testing Fixed Uniswap V3 Math Functions...")
    print("=" * 60)
    
    # Test cases with different price ranges
    test_cases = [
        (3000.0, 3010.0, "0.33% range"),
        (3000.0, 3150.0, "5% range"),
        (3000.0, 3300.0, "10% range"),
        (3000.0, 3600.0, "20% range")
    ]
    
    for price_lower, price_upper, description in test_cases:
        print(f"\n📊 Testing {description}: ${price_lower:.2f} - ${price_upper:.2f}")
        print("-" * 50)
        
        try:
            # Convert to X96 format
            sqrt_price_a_x96 = price_to_sqrt_price_x96(price_lower)
            sqrt_price_b_x96 = price_to_sqrt_price_x96(price_upper)
            
            # Test with different amounts
            test_amounts = [1, 10, 100, 1000, 10000]
            
            for amount in test_amounts:
                print(f"\n💰 Testing with Amount1: ${amount}")
                
                # Calculate liquidity using fixed function
                liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount)
                print(f"💧 Liquidity: {liquidity}")
                
                if liquidity > 0:
                    # Test reverse calculation
                    calc_amount = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
                    print(f"💱 Calculated Amount: {calc_amount}")
                    print(f"✅ Match: {calc_amount == amount}")
                    
                    # Test position value calculation
                    current_price = (price_lower + price_upper) / 2
                    current_price_x96 = price_to_sqrt_price_x96(current_price)
                    amount0, amount1 = get_amounts_for_liquidity(
                        current_price_x96, sqrt_price_a_x96, sqrt_price_b_x96, liquidity
                    )
                    
                    position_value = (amount0 / Q96) * current_price + amount1
                    print(f"💵 Position Value: ${position_value:.2f}")
                    print(f"🎯 Expected Value: ${amount:.2f}")
                    print(f"📊 Accuracy: {abs(position_value - amount) / amount * 100:.2f}%")
                else:
                    print("❌ Liquidity is still 0 - function not fixed!")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_liquidity_consistency():
    """Test liquidity calculation consistency."""
    print("\n🧪 Testing Liquidity Consistency...")
    print("=" * 60)
    
    # Use a reasonable price range
    price_lower = 3000.0
    price_upper = 3150.0  # 5% range
    amount1 = 1000  # $1000 USDC
    
    print(f"💰 Price Range: ${price_lower:.2f} - ${price_upper:.2f}")
    print(f"💰 Amount1: ${amount1}")
    
    try:
        # Convert to X96 format
        sqrt_price_a_x96 = price_to_sqrt_price_x96(price_lower)
        sqrt_price_b_x96 = price_to_sqrt_price_x96(price_upper)
        
        # Calculate liquidity
        liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount1)
        print(f"💧 Calculated Liquidity: {liquidity}")
        
        if liquidity > 0:
            # Test with different current prices
            test_prices = [3000, 3075, 3150, 3200, 2800]
            
            print(f"\n📊 Testing Position Values at Different Prices:")
            for price in test_prices:
                current_price_x96 = price_to_sqrt_price_x96(price)
                amount0, amount1 = get_amounts_for_liquidity(
                    current_price_x96, sqrt_price_a_x96, sqrt_price_b_x96, liquidity
                )
                
                position_value = (amount0 / Q96) * price + amount1
                print(f"   Price ${price}: ${position_value:.2f}")
                
        else:
            print("❌ Cannot test - liquidity is 0!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🚀 Starting Fixed Math Functions Test")
    print("=" * 70)
    
    # Test fixed functions
    test_fixed_functions()
    
    # Test liquidity consistency
    test_liquidity_consistency()
    
    print("\n🧪 Test completed!")
    print("\n💡 Expected Results:")
    print("   1. get_liquidity_for_amount1 should now return non-zero values")
    print("   2. Position value calculations should be accurate")
    print("   3. Strategies should work without overflow errors")

if __name__ == "__main__":
    main()
