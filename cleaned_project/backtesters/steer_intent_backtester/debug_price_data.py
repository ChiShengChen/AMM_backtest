#!/usr/bin/env python3
"""
Debug script to check price data volatility.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_price_data():
    """Analyze price data for volatility."""
    print("🔍 Analyzing Price Data Volatility...")
    print("=" * 60)
    
    # Load data
    data_file = "data/ETHUSDC_1h.csv"
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    print(f"📊 Data Overview:")
    print(f"   Total Records: {len(data)}")
    print(f"   Date Range: {data.index[0]} to {data.index[-1]}")
    print(f"   Columns: {list(data.columns)}")
    
    # Analyze price data
    if 'close' in data.columns:
        prices = data['close']
        
        print(f"\n💰 Price Analysis:")
        print(f"   Initial Price: ${prices.iloc[0]:.2f}")
        print(f"   Final Price: ${prices.iloc[-1]:.2f}")
        print(f"   Min Price: ${prices.min():.2f}")
        print(f"   Max Price: ${prices.max():.2f}")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        print(f"\n📈 Return Analysis:")
        print(f"   Mean Return: {returns.mean()*100:.4f}%")
        print(f"   Std Return: {returns.std()*100:.4f}%")
        print(f"   Min Return: {returns.min()*100:.4f}%")
        print(f"   Max Return: {returns.max()*100:.4f}%")
        
        # Check for negative returns
        negative_returns = (returns < 0).sum()
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        
        print(f"\n📊 Return Distribution:")
        print(f"   Positive Returns: {positive_returns} ({positive_returns/total_returns*100:.2f}%)")
        print(f"   Negative Returns: {negative_returns} ({negative_returns/total_returns*100:.2f}%)")
        print(f"   Zero Returns: {total_returns - positive_returns - negative_returns}")
        
        # Check for significant drops
        significant_drops = (returns < -0.05).sum()  # >5% drops
        print(f"   Significant Drops (>5%): {significant_drops}")
        
        # Sample some price changes
        print(f"\n📋 Sample Price Changes (first 20):")
        for i in range(min(20, len(returns))):
            if i < len(prices) - 1:
                change = returns.iloc[i] * 100
                print(f"   {i}: ${prices.iloc[i]:.2f} → ${prices.iloc[i+1]:.2f} ({change:+.2f}%)")
        
        # Check recent data
        print(f"\n📋 Recent Price Changes (last 20):")
        for i in range(max(0, len(returns)-20), len(returns)):
            if i < len(prices) - 1:
                change = returns.iloc[i] * 100
                print(f"   {i}: ${prices.iloc[i]:.2f} → ${prices.iloc[i+1]:.2f} ({change:+.2f}%)")
                
        # Check for any large movements
        large_moves = returns[abs(returns) > 0.1]  # >10% moves
        if not large_moves.empty:
            print(f"\n🚨 Large Price Movements (>10%):")
            for idx, move in large_moves.head(10).items():
                print(f"   {idx}: {move*100:+.2f}%")
        else:
            print(f"\n📊 No large price movements (>10%) found")
            
    else:
        print("❌ No 'close' column found in data!")

def main():
    """Main function."""
    print("🚀 Starting Price Data Analysis")
    print("=" * 70)
    
    analyze_price_data()
    
    print("\n🔍 Analysis completed!")
    print("\n💡 Expected Findings:")
    print("   1. Should see both positive and negative returns")
    print("   2. Should see some significant price drops")
    print("   3. Price data should show realistic volatility")

if __name__ == "__main__":
    main()
