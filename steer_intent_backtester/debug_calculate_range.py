#!/usr/bin/env python3
"""
Debug script to test strategy calculate_range method directly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.strategies.classic import ClassicStrategy

def test_calculate_range():
    """Test strategy calculate_range method directly."""
    print("🔍 Testing Strategy calculate_range Method...")
    print("=" * 60)
    
    # Create strategy
    strategy_params = {
        "width_mode": "percent",
        "width_value": 50.0,  # Very wide range: 50%
        "placement_mode": "center",
        "curve_type": "uniform",
        "liquidity_scale": 1.0,  # Full scaling
        "max_positions": 1
    }
    
    try:
        print("🔄 Creating ClassicStrategy...")
        strategy = ClassicStrategy(**strategy_params)
        
        print(f"✅ Strategy created successfully")
        print(f"   Width mode: {strategy.width_mode}")
        print(f"   Width value: {strategy.width_value}")
        print(f"   Placement mode: {strategy.placement_mode}")
        print(f"   Curve type: {strategy.curve_type}")
        print(f"   Max positions: {strategy.max_positions}")
        
        # Create dummy price data
        print("\n📊 Creating test price data...")
        timestamps = pd.date_range(start='2025-07-01', periods=5, freq='1h')
        price_data = pd.DataFrame({
            'close': [100, 105, 95, 110, 90],
            'open': [100, 105, 95, 110, 90],
            'high': [102, 107, 97, 112, 92],
            'low': [98, 103, 93, 108, 88]
        }, index=timestamps)
        
        print(f"   Price data created: {len(price_data)} records")
        print(f"   Price range: ${price_data['close'].min():.2f} to ${price_data['close'].max():.2f}")
        
        # Test calculate_range method
        print("\n🔄 Testing calculate_range method...")
        
        current_price = 100.0
        portfolio_value = 10000.0
        
        print(f"   Current price: ${current_price}")
        print(f"   Portfolio value: ${portfolio_value}")
        
        ranges, liquidities = strategy.calculate_range(price_data, current_price, portfolio_value)
        
        print(f"\n📊 calculate_range Results:")
        print(f"   Ranges: {ranges}")
        print(f"   Liquidities: {liquidities}")
        
        if ranges and liquidities:
            print(f"   Number of positions: {len(ranges)}")
            
            for i, (lower, upper) in enumerate(ranges):
                liquidity = liquidities[i] if i < len(liquidities) else 0
                width_pct = ((upper - lower) / lower) * 100
                print(f"   Position {i+1}: ${lower:.2f} - ${upper:.2f} (width: {width_pct:.1f}%), Liquidity: {liquidity}")
        else:
            print(f"   ⚠️  No ranges or liquidities returned!")
            
        # Test curve generation directly
        print(f"\n🔄 Testing curve generation directly...")
        
        if hasattr(strategy, 'curve'):
            print(f"   Curve type: {type(strategy.curve).__name__}")
            print(f"   Curve params: {strategy.curve.__dict__}")
            
            # Test curve generate_distribution
            if hasattr(strategy.curve, 'generate_distribution'):
                center_price = current_price
                width_pct = strategy.width_value
                total_liquidity = portfolio_value * 0.95
                
                print(f"   Testing curve.generate_distribution:")
                print(f"     Center price: ${center_price}")
                print(f"     Width: {width_pct}%")
                print(f"     Total liquidity: ${total_liquidity}")
                
                try:
                    distribution = strategy.curve.generate_distribution(center_price, width_pct, total_liquidity)
                    print(f"     Distribution result: {distribution}")
                    
                    if distribution:
                        print(f"     Number of bins: {len(distribution)}")
                        for i, (lower, upper, liq) in enumerate(distribution):
                            print(f"       Bin {i+1}: ${lower:.2f} - ${upper:.2f}, Liquidity: {liq}")
                    else:
                        print(f"     ⚠️  No distribution generated!")
                        
                except Exception as e:
                    print(f"     ❌ Error generating distribution: {e}")
            else:
                print(f"   ⚠️  Curve has no generate_distribution method!")
        else:
            print(f"   ⚠️  Strategy has no curve attribute!")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🚀 Starting calculate_range Method Test")
    print("=" * 70)
    
    test_calculate_range()
    
    print("\n🔍 Test completed!")
    print("\n💡 Expected Findings:")
    print("   1. Strategy should create valid price ranges")
    print("   2. Curve should generate distribution")
    print("   3. Should see realistic position bounds")

if __name__ == "__main__":
    main()
