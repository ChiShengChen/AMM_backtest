#!/usr/bin/env python3
"""
Real imperfect strategy implementation by modifying strategy logic.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import warnings
import random
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steerbt.backtester import Backtester
from steerbt.strategies.classic import ClassicStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealImperfectClassicStrategy(ClassicStrategy):
    """Classic strategy with real logic modifications to generate MDD."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.imperfection_level = kwargs.get('imperfection_level', 0.5)
        self.rebalance_failure_rate = kwargs.get('rebalance_failure_rate', 0.3)
        self.liquidity_shortage_rate = kwargs.get('liquidity_shortage_rate', 0.2)
        self.market_impact_rate = kwargs.get('market_impact_rate', 0.1)
        self.rebalance_count = 0
        self.failed_rebalances = 0
        self.last_rebalance_time = None
        self.forced_no_rebalance_periods = 0
        
    def update(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float) -> bool:
        """Override update method with real logic modifications."""
        # Call parent method to get the original decision
        original_should_rebalance = super().update(price_data, current_price, portfolio_value)
        
        if not original_should_rebalance:
            return False
            
        # Apply real imperfection logic
        
        # 1. Random rebalancing failure
        if random.random() < self.rebalance_failure_rate:
            logger.info(f"🔄 Rebalancing failed due to random failure (level: {self.imperfection_level})")
            self.failed_rebalances += 1
            return False
        
        # 2. Force no rebalancing during certain periods (simulate market stress)
        current_time = price_data.index[-1] if not price_data.empty else None
        if current_time:
            # Force no rebalancing during the first 20% of the backtest period
            total_periods = len(price_data)
            current_period = len(price_data)
            if current_period < total_periods * 0.2:
                if random.random() < 0.8:  # 80% chance of no rebalancing in early periods
                    logger.info(f"⏰ Forced no rebalancing in early period {current_period}/{total_periods}")
                    self.forced_no_rebalance_periods += 1
                    return False
            
            # Force no rebalancing during market stress (simulate)
            if random.random() < 0.1:  # 10% chance of market stress
                logger.info(f"📉 Market stress detected, forcing no rebalancing")
                self.forced_no_rebalance_periods += 1
                return False
        
        # 3. Reduce rebalancing frequency based on imperfection level
        if self.last_rebalance_time and current_time:
            time_since_last = (current_time - self.last_rebalance_time).total_seconds() / 3600
            min_interval = 24 * (1 + self.imperfection_level * 2)  # 24-72 hours based on imperfection
            
            if time_since_last < min_interval:
                logger.info(f"⏳ Too soon to rebalance: {time_since_last:.1f}h < {min_interval:.1f}h")
                return False
        
        # 4. Sometimes ignore the trigger completely
        if random.random() < self.imperfection_level * 0.3:
            logger.info(f"🎲 Ignoring rebalancing trigger due to imperfection")
            return False
        
        # If we get here, allow rebalancing
        self.rebalance_count += 1
        self.last_rebalance_time = current_time
        return True
    
    def calculate_range(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float):
        """Override calculate_range with real logic modifications."""
        ranges, liquidities = super().calculate_range(price_data, current_price, portfolio_value)
        
        if not ranges or not liquidities:
            return ranges, liquidities
        
        # Apply real imperfection to ranges and liquidities
        
        # 1. Sometimes create suboptimal ranges
        if random.random() < self.imperfection_level * 0.4:
            logger.info(f"🔧 Creating suboptimal ranges due to imperfection")
            
            adjusted_ranges = []
            for lower, upper in ranges:
                range_width = upper - lower
                
                # 30% chance to make range too wide (less effective)
                if random.random() < 0.3:
                    adjusted_lower = lower - range_width * 0.3
                    adjusted_upper = upper + range_width * 0.3
                    logger.info(f"   Widening range: {lower:.2f}-{upper:.2f} -> {adjusted_lower:.2f}-{adjusted_upper:.2f}")
                
                # 30% chance to make range too narrow (more concentrated risk)
                elif random.random() < 0.3:
                    adjusted_lower = lower + range_width * 0.2
                    adjusted_upper = upper - range_width * 0.2
                    logger.info(f"   Narrowing range: {lower:.2f}-{upper:.2f} -> {adjusted_lower:.2f}-{adjusted_upper:.2f}")
                
                # 40% chance to keep original range
                else:
                    adjusted_lower, adjusted_upper = lower, upper
                
                adjusted_ranges.append((adjusted_lower, adjusted_upper))
            
            ranges = adjusted_ranges
        
        # 2. Reduce liquidity effectiveness
        imperfection_factor = 1.0 - (self.imperfection_level * 0.6)
        adjusted_liquidities = [liq * imperfection_factor for liq in liquidities]
        
        # 3. Sometimes add random noise to liquidities
        if random.random() < self.imperfection_level * 0.3:
            logger.info(f"📊 Adding noise to liquidities due to imperfection")
            noise_factor = 0.8 + random.random() * 0.4  # 0.8 to 1.2
            adjusted_liquidities = [liq * noise_factor for liq in adjusted_liquidities]
        
        return ranges, adjusted_liquidities

def main():
    """Main function for real imperfect strategy simulation."""
    print("🚀 Starting Real Imperfect Strategy MDD Generation Test")
    print("=" * 80)
    
    # Load ETHUSDC data
    data_file = "data/ETHUSDC_1h.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Use only last 3 months for testing
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=90)
    
    # Filter data
    data = data[data.index >= start_date]
    
    print(f"📈 Using limited data: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Imperfection configurations with real logic modifications
    imperfection_configs = {
        'classic_perfect': {
            'level': 0.0,
            'rebalance_failure_rate': 0.0,
            'liquidity_shortage_rate': 0.0,
            'market_impact_rate': 0.0
        },
        'classic_slight_imperfect': {
            'level': 0.3,
            'rebalance_failure_rate': 0.15,
            'liquidity_shortage_rate': 0.1,
            'market_impact_rate': 0.05
        },
        'classic_moderate_imperfect': {
            'level': 0.6,
            'rebalance_failure_rate': 0.35,
            'liquidity_shortage_rate': 0.25,
            'market_impact_rate': 0.15
        },
        'classic_highly_imperfect': {
            'level': 0.9,
            'rebalance_failure_rate': 0.6,
            'liquidity_shortage_rate': 0.5,
            'market_impact_rate': 0.3
        }
    }
    
    print(f"\n💡 Real Imperfect Strategy Analysis:")
    print(f"   🔍 Key factors for MDD generation:")
    print(f"      - Real rebalancing failures ({imperfection_configs['classic_highly_imperfect']['rebalance_failure_rate']*100:.0f}% rate)")
    print(f"      - Forced no-rebalancing periods")
    print(f"      - Suboptimal range creation")
    print(f"      - Reduced liquidity effectiveness")
    print(f"      - Market stress simulation")

if __name__ == "__main__":
    main()
