#!/usr/bin/env python3
"""
Debug script to analyze strategy issues step by step.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester
from steerbt.strategies.classic import ClassicStrategy
from steerbt.portfolio import Portfolio, Position
from steerbt.uv3_math import *

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_classic_strategy():
    """Debug ClassicStrategy step by step."""
    print("🔍 Debugging ClassicStrategy...")
    print("=" * 50)
    
    # Load minimal data
    data_file = "data/ETHUSDC_1h.csv"
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 7 days for debugging
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=7)
    data = data[data.index >= start_date]
    
    print(f"📊 Using debug data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Create strategy with more reasonable settings
    strategy_params = {
        "width_mode": "percent",
        "width_value": 5.0,  # 5% range - more reasonable
        "placement_mode": "center",
        "curve_type": "uniform",
        "liquidity_scale": 0.001  # More reasonable scaling
    }
    
    # Create strategy instance
    strategy = ClassicStrategy(**strategy_params)
    
    # Test with first few data points
    for i in range(min(5, len(data))):
        row = data.iloc[i]
        current_price = row['close']
        timestamp = data.index[i]
        
        print(f"\n📅 Timestamp: {timestamp}")
        print(f"💰 Current Price: ${current_price:.2f}")
        
        try:
            # Create dummy price data for strategy
            price_data = data.iloc[:i+1] if i > 0 else data.iloc[:1]
            
            # Calculate range
            ranges, liquidities = strategy.calculate_range(price_data, current_price, 10000.0)
            print(f"📊 Ranges: {len(ranges)} positions")
            
            for j, (lower, upper) in enumerate(ranges):
                print(f"   Position {j+1}: {lower:.2f} - {upper:.2f}")
                if j < len(liquidities):
                    print(f"   Liquidity: {liquidities[j]:.2e}")
            
            # Test portfolio value calculation
            portfolio = Portfolio(10000.0)
            
            # Add a small position
            if ranges:
                lower, upper = ranges[0]
                liquidity = liquidities[0] if liquidities else 1000.0
                
                # Create position
                position = Position(lower, upper, liquidity)
                portfolio.add_position(position)
                
                # Calculate position value
                amount0, amount1, total_value = position.get_value(current_price)
                
                print(f"💱 Amount0: {amount0:.6f}")
                print(f"💱 Amount1: {amount1:.2f}")
                print(f"💵 Position Value: ${total_value:.2f}")
                
                # Check if this is reasonable
                if total_value > 1000000:  # > $1M is suspicious
                    print("⚠️  WARNING: Position value seems too high!")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

def debug_uv3_math():
    """Debug Uniswap V3 math functions."""
    print("\n🔍 Debugging Uniswap V3 Math...")
    print("=" * 50)
    
    # Test with more reasonable values
    price_lower = 3000.0  # $3000
    price_upper = 3150.0  # $3150 (5% range)
    current_price = 3075.0  # Middle of range
    
    print(f"💰 Price Range: ${price_lower:.2f} - ${price_upper:.2f}")
    print(f"💰 Current Price: ${current_price:.2f}")
    
    # Test liquidity calculation
    try:
        # Small amounts
        amount0 = 1.0  # 1 ETH
        amount1 = 3000.0  # $3000 USDC
        
        # Convert to X96 format
        sqrt_price_a_x96 = price_to_sqrt_price_x96(price_lower)
        sqrt_price_b_x96 = price_to_sqrt_price_x96(price_upper)
        
        print(f"🔢 Sqrt Price A (X96): {sqrt_price_a_x96}")
        print(f"🔢 Sqrt Price B (X96): {sqrt_price_b_x96}")
        
        # Calculate liquidity for amount0
        liquidity0 = get_liquidity_for_amount0(sqrt_price_a_x96, sqrt_price_b_x96, int(amount0 * Q96))
        print(f"💧 Liquidity for Amount0: {liquidity0:.2e}")
        
        # Calculate liquidity for amount1
        liquidity1 = get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, int(amount1))
        print(f"💧 Liquidity for Amount1: {liquidity1:.2e}")
        
        # Use the smaller liquidity
        liquidity = min(liquidity0, liquidity1)
        print(f"💧 Final Liquidity: {liquidity:.2e}")
        
        # Test reverse calculation
        calc_amount0 = get_amount0_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
        calc_amount1 = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
        
        print(f"💱 Calculated Amount0: {calc_amount0 / Q96:.6f}")
        print(f"💱 Calculated Amount1: {calc_amount1:.2f}")
        
        # Test with very small liquidity
        tiny_liquidity = int(liquidity * 0.001)
        print(f"💧 Tiny Liquidity: {tiny_liquidity:.2e}")
        
        tiny_amount0 = get_amount0_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, tiny_liquidity)
        tiny_amount1 = get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, tiny_liquidity)
        
        print(f"💱 Tiny Amount0: {tiny_amount0 / Q96:.6f}")
        print(f"💱 Tiny Amount1: {tiny_amount1:.2f}")
        
        # Calculate position value
        position_value = (tiny_amount0 / Q96) * current_price + tiny_amount1
        print(f"💵 Tiny Position Value: ${position_value:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def debug_portfolio():
    """Debug portfolio calculations."""
    print("\n🔍 Debugging Portfolio...")
    print("=" * 50)
    
    portfolio = Portfolio(10000.0)
    
    # Test adding positions
    try:
        # Add a reasonable position
        position = Position(3000, 3150, 1000)
        portfolio.add_position(position)
        
        current_price = 3075.0
        portfolio_value = portfolio.get_total_value(current_price)
        
        print(f"💰 Portfolio Value: ${portfolio_value:.2f}")
        print(f"📊 Position Count: {len(portfolio.positions)}")
        
        # Check if this is reasonable
        if portfolio_value > 1000000:  # > $1M is suspicious
            print("⚠️  WARNING: Portfolio value seems too high!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🚀 Starting Strategy Debug Session")
    print("=" * 60)
    
    # Debug Uniswap V3 math first
    debug_uv3_math()
    
    # Debug portfolio
    debug_portfolio()
    
    # Debug strategy
    debug_classic_strategy()
    
    print("\n🔍 Debug session completed!")

if __name__ == "__main__":
    main()
