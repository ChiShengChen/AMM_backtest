#!/usr/bin/env python3
"""
Deep debug script to analyze Uniswap V3 math precision issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.uv3_math import *

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_x96_precision():
    """Debug X96 format precision issues."""
    print("🔍 Debugging X96 Format Precision...")
    print("=" * 60)
    
    # Test different price ranges
    test_cases = [
        (3000.0, 3010.0, "0.33% range"),
        (3000.0, 3150.0, "5% range"),
        (3000.0, 3300.0, "10% range"),
        (3000.0, 3600.0, "20% range"),
        (3000.0, 4500.0, "50% range")
    ]
    
    for price_lower, price_upper, description in test_cases:
        print(f"\n📊 Testing {description}: ${price_lower:.2f} - ${price_upper:.2f}")
        print("-" * 50)
        
        try:
            # Convert to X96 format
            sqrt_price_a_x96 = price_to_sqrt_price_x96(price_lower)
            sqrt_price_b_x96 = price_to_sqrt_price_x96(price_upper)
            
            print(f"🔢 Sqrt Price A (X96): {sqrt_price_a_x96}")
            print(f"🔢 Sqrt Price B (X96): {sqrt_price_b_x96}")
            print(f"🔢 Difference (X96): {sqrt_price_b_x96 - sqrt_price_a_x96}")
            
            # Test with different amounts
            test_amounts = [1, 10, 100, 1000, 10000]
            
            for amount in test_amounts:
                print(f"\n💰 Testing with Amount1: {amount}")
                
                # Calculate liquidity
                liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount)
                print(f"💧 Liquidity: {liquidity}")
                
                if liquidity > 0:
                    # Test reverse calculation
                    calc_amount = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
                    print(f"💱 Calculated Amount: {calc_amount}")
                    print(f"✅ Match: {calc_amount == amount}")
                else:
                    print("❌ Liquidity is 0 - precision issue!")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def debug_liquidity_calculation():
    """Debug liquidity calculation step by step."""
    print("\n🔍 Debugging Liquidity Calculation...")
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
        
        print(f"\n🔢 X96 Conversion:")
        print(f"   Price A: ${price_lower:.2f} -> Sqrt: {np.sqrt(price_lower):.6f} -> X96: {sqrt_price_a_x96}")
        print(f"   Price B: ${price_upper:.2f} -> Sqrt: {np.sqrt(price_upper):.6f} -> X96: {sqrt_price_b_x96}")
        
        # Calculate difference
        diff_x96 = sqrt_price_b_x96 - sqrt_price_a_x96
        print(f"\n🔢 Difference Calculation:")
        print(f"   Sqrt B - Sqrt A = {diff_x96}")
        print(f"   As float: {diff_x96 / Q96:.6f}")
        
        # Test liquidity calculation
        print(f"\n💧 Liquidity Calculation:")
        print(f"   Formula: amount1 / (sqrt_price_b_x96 - sqrt_price_a_x96)")
        print(f"   = {amount1} / {diff_x96}")
        
        # Manual calculation
        liquidity_float = amount1 / (diff_x96 / Q96)
        print(f"   Manual float calc: {amount1} / {diff_x96 / Q96:.6f} = {liquidity_float:.6f}")
        
        # Actual function call
        liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount1)
        print(f"   Function result: {liquidity}")
        
        if liquidity == 0:
            print("❌ PROBLEM: Function returns 0 due to integer division!")
            print("   This happens because amount1 / diff_x96 < 1 in integer arithmetic")
            
            # Test with larger amounts
            print(f"\n🔍 Testing with larger amounts:")
            for large_amount in [100000, 1000000, 10000000]:
                large_liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, large_amount)
                print(f"   Amount ${large_amount:,}: Liquidity = {large_liquidity}")
                
        else:
            print("✅ Liquidity calculated successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def debug_alternative_approach():
    """Debug alternative approach to liquidity calculation."""
    print("\n🔍 Debugging Alternative Approach...")
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
        
        # Alternative: scale up the calculation to avoid precision loss
        scale_factor = 10**18  # Use 18 decimal places
        
        # Scale up amount1
        scaled_amount1 = int(amount1 * scale_factor)
        
        print(f"\n🔧 Alternative Calculation with Scaling:")
        print(f"   Original Amount1: {amount1}")
        print(f"   Scaled Amount1: {scaled_amount1}")
        print(f"   Scale Factor: {scale_factor}")
        
        # Calculate liquidity with scaling
        liquidity_scaled = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, scaled_amount1)
        print(f"   Scaled Liquidity: {liquidity_scaled}")
        
        if liquidity_scaled > 0:
            # Scale back down
            actual_liquidity = liquidity_scaled / scale_factor
            print(f"   Actual Liquidity: {actual_liquidity}")
            
            # Test reverse calculation
            calc_amount = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity_scaled)
            calc_amount_scaled = calc_amount / scale_factor
            print(f"   Calculated Amount: ${calc_amount_scaled:.2f}")
            print(f"   Original Amount: ${amount1}")
            print(f"   Match: {abs(calc_amount_scaled - amount1) < 0.01}")
        else:
            print("❌ Even scaling didn't help - fundamental issue!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def debug_strategy_impact():
    """Debug how this affects strategy calculations."""
    print("\n🔍 Debugging Strategy Impact...")
    print("=" * 60)
    
    # Simulate what happens in strategy
    portfolio_value = 10000.0
    liquidity_scale = 0.001
    
    print(f"💰 Portfolio Value: ${portfolio_value:.2f}")
    print(f"💧 Liquidity Scale: {liquidity_scale}")
    
    # Calculate target liquidity
    target_liquidity = portfolio_value * liquidity_scale
    print(f"🎯 Target Liquidity: ${target_liquidity:.6f}")
    
    # Test with different price ranges
    test_ranges = [
        (3000.0, 3010.0, "0.33% range"),
        (3000.0, 3150.0, "5% range"),
        (3000.0, 3300.0, "10% range")
    ]
    
    for price_lower, price_upper, description in test_ranges:
        print(f"\n📊 Testing {description}: ${price_lower:.2f} - ${price_upper:.2f}")
        
        try:
            # Convert to X96 format
            sqrt_price_a_x96 = price_to_sqrt_price_x96(price_lower)
            sqrt_price_b_x96 = price_to_sqrt_price_x96(price_upper)
            
            # Calculate liquidity for target amount
            target_amount = int(target_liquidity)
            liquidity = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, target_amount)
            
            print(f"   Target Amount: ${target_amount}")
            print(f"   Calculated Liquidity: {liquidity}")
            
            if liquidity > 0:
                # Calculate position value
                amount0, amount1 = get_amounts_for_liquidity(
                    sqrt_price_a_x96, sqrt_price_a_x96, sqrt_price_a_x96, sqrt_price_b_x96, liquidity
                )
                
                current_price = (price_lower + price_upper) / 2
                position_value = (amount0 / Q96) * current_price + amount1
                
                print(f"   Position Value: ${position_value:.2f}")
                print(f"   Expected Value: ${target_liquidity:.2f}")
                print(f"   Match: {abs(position_value - target_liquidity) < 0.01}")
            else:
                print("   ❌ Cannot calculate position value - liquidity is 0!")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Main debug function."""
    print("🚀 Starting Deep Debug Session")
    print("=" * 70)
    
    # Debug X96 precision
    debug_x96_precision()
    
    # Debug liquidity calculation
    debug_liquidity_calculation()
    
    # Debug alternative approach
    debug_alternative_approach()
    
    # Debug strategy impact
    debug_strategy_impact()
    
    print("\n🔍 Deep debug session completed!")
    print("\n💡 Key Findings:")
    print("   1. X96 format has precision limitations for small price differences")
    print("   2. get_liquidity_for_amount1 returns 0 for small amounts due to integer division")
    print("   3. This causes strategies to fail when calculating position values")
    print("   4. Need to either scale up calculations or use alternative approach")

if __name__ == "__main__":
    main()
