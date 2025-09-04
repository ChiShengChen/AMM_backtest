"""
Imperfect Classic Strategy with real logic modifications to generate MDD.
"""

import random
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from .classic import ClassicStrategy

logger = logging.getLogger(__name__)

class ImperfectClassicStrategy(ClassicStrategy):
    """Classic strategy with real logic modifications to generate MDD."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Imperfection parameters
        self.imperfection_level = kwargs.get('imperfection_level', 0.5)  # 0-1 scale
        self.rebalance_failure_rate = kwargs.get('rebalance_failure_rate', 0.3)
        self.liquidity_shortage_rate = kwargs.get('liquidity_shortage_rate', 0.2)
        self.market_impact_rate = kwargs.get('market_impact_rate', 0.1)
        self.forced_no_rebalance_chance = kwargs.get('forced_no_rebalance_chance', 0.2)
        
        # Internal state
        self.rebalance_count = 0
        self.failed_rebalances = 0
        self.last_rebalance_time = None
        self.forced_no_rebalance_periods = 0
        self.consecutive_failures = 0
        self.market_stress_mode = False
        
        # Initialize random seed for reproducibility
        random.seed(42)
        
        logger.info(f"Initialized ImperfectClassicStrategy with imperfection level: {self.imperfection_level}")
    
    def update(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float) -> bool:
        """Override update method with real logic modifications."""
        # Call parent method to get the original decision
        original_should_rebalance = super().update(price_data, current_price, portfolio_value)
        
        if not original_should_rebalance:
            return False
            
        # Apply real imperfection logic - BALANCED approach
        
        # 1. Random rebalancing failure - reduced frequency
        if random.random() < self.imperfection_level * 0.2:  # Reduced from 0.3 to 0.2
            logger.info(f"🔄 Rebalancing failed due to random failure (level: {self.imperfection_level})")
            self.failed_rebalances += 1
            self.consecutive_failures += 1
            return False
        
        # 2. Force no rebalancing during certain periods - reduced frequency
        current_time = price_data.index[-1] if not price_data.empty else None
        if current_time:
            # Force no rebalancing during the first 20% of the backtest period (reduced from 40%)
            total_periods = len(price_data)
            current_period = len(price_data)
            if current_period < total_periods * 0.2:
                if random.random() < 0.6:  # Reduced from 0.9 to 0.6
                    logger.info(f"⏰ Forced no rebalancing in early period {current_period}/{total_periods}")
                    self.forced_no_rebalance_periods += 1
                    return False
            
            # Force no rebalancing during market stress - reduced frequency
            if random.random() < 0.15:  # Reduced from 0.25 to 0.15
                logger.info(f"📉 Market stress detected, forcing no rebalancing")
                self.market_stress_mode = True
                self.forced_no_rebalance_periods += 1
                return False
            
            # Exit market stress mode after some time
            if self.market_stress_mode and random.random() < 0.1:  # Increased from 0.05 to 0.1
                self.market_stress_mode = False
                logger.info(f"✅ Market stress mode ended")
        
        # 3. Reduce rebalancing frequency - more reasonable intervals
        if self.last_rebalance_time and current_time:
            time_since_last = (current_time - self.last_rebalance_time).total_seconds() / 3600
            min_interval = 24 * (1 + self.imperfection_level * 2)  # Reduced from 4 to 2 (24-72 hours)
            
            if time_since_last < min_interval:
                logger.info(f"⏳ Too soon to rebalance: {time_since_last:.1f}h < {min_interval:.1f}h")
                return False
        
        # 4. Sometimes ignore the trigger - reduced chance
        if random.random() < self.forced_no_rebalance_chance * 0.8:  # Reduced by 20%
            logger.info(f"🎲 Ignoring rebalancing trigger due to imperfection")
            return False
        
        # 5. Consecutive failure penalty - less aggressive
        if self.consecutive_failures > 2:  # Increased from 1 to 2
            penalty_chance = min(0.7, self.consecutive_failures * 0.2)  # Reduced from 0.3 to 0.2
            if random.random() < penalty_chance:
                logger.info(f"🚫 Consecutive failure penalty: {self.consecutive_failures} failures")
                return False
        
        # 6. Price-based rebalancing inhibition - reduced frequency
        if len(price_data) > 10:
            recent_prices = price_data['close'].tail(10)
            price_volatility = recent_prices.std() / recent_prices.mean()
            
            # Inhibit rebalancing during high volatility periods - reduced frequency
            if price_volatility > 0.05:  # 5% volatility threshold
                if random.random() < 0.4:  # Reduced from 0.7 to 0.4
                    logger.info(f"📈 High volatility detected ({price_volatility:.3f}), inhibiting rebalancing")
                    return False
        
        # 7. Portfolio value-based inhibition - reduced frequency
        if portfolio_value > 0:
            # Sometimes inhibit rebalancing when portfolio is performing well - reduced frequency
            if random.random() < self.imperfection_level * 0.2:  # Reduced from 0.4 to 0.2
                logger.info(f"💰 Portfolio performing well, inhibiting rebalancing")
                return False
        
        # If we get here, allow rebalancing
        self.rebalance_count += 1
        self.last_rebalance_time = current_time
        self.consecutive_failures = 0  # Reset consecutive failures
        
        logger.info(f"✅ Rebalancing allowed (imperfection level: {self.imperfection_level})")
        return True
    
    def calculate_range(self, price_data: pd.DataFrame, current_price: float, portfolio_value: float):
        """Override calculate_range with real logic modifications."""
        ranges, liquidities = super().calculate_range(price_data, current_price, portfolio_value)
        
        if not ranges or not liquidities:
            return ranges, liquidities
        
        # Apply real imperfection to ranges and liquidities
        
        # 1. Sometimes create suboptimal ranges
        if random.random() < self.imperfection_level * 0.5:
            logger.info(f"🔧 Creating suboptimal ranges due to imperfection")
            
            adjusted_ranges = []
            for lower, upper in ranges:
                range_width = upper - lower
                
                # 40% chance to make range too wide (less effective)
                if random.random() < 0.4:
                    adjusted_lower = lower - range_width * 0.4
                    adjusted_upper = upper + range_width * 0.4
                    logger.info(f"   Widening range: {lower:.2f}-{upper:.2f} -> {adjusted_lower:.2f}-{adjusted_upper:.2f}")
                
                # 30% chance to make range too narrow (more concentrated risk)
                elif random.random() < 0.3:
                    adjusted_lower = lower + range_width * 0.25
                    adjusted_upper = upper - range_width * 0.25
                    logger.info(f"   Narrowing range: {lower:.2f}-{upper:.2f} -> {adjusted_lower:.2f}-{adjusted_upper:.2f}")
                
                # 30% chance to keep original range
                else:
                    adjusted_lower, adjusted_upper = lower, upper
                
                adjusted_ranges.append((adjusted_lower, adjusted_upper))
            
            ranges = adjusted_ranges
        
        # 2. Reduce liquidity effectiveness
        imperfection_factor = 1.0 - (self.imperfection_level * 0.7)
        adjusted_liquidities = [liq * imperfection_factor for liq in liquidities]
        
        # 3. Sometimes add random noise to liquidities
        if random.random() < self.imperfection_level * 0.4:
            logger.info(f"📊 Adding noise to liquidities due to imperfection")
            noise_factor = 0.7 + random.random() * 0.6  # 0.7 to 1.3
            adjusted_liquidities = [liq * noise_factor for liq in adjusted_liquidities]
        
        # 4. Market stress mode effects
        if self.market_stress_mode:
            logger.info(f"📉 Market stress mode: reducing liquidity effectiveness")
            stress_factor = 0.5 + random.random() * 0.3  # 0.5 to 0.8
            adjusted_liquidities = [liq * stress_factor for liq in adjusted_liquidities]
        
        # 5. NEW: Force "bad" liquidity distribution for high imperfection levels
        if self.imperfection_level > 0.7:
            if random.random() < 0.3:  # 30% chance to create very bad distribution
                logger.info(f"💥 FORCING BAD liquidity distribution for high imperfection!")
                
                # Create extremely wide ranges that are ineffective
                bad_ranges = []
                for lower, upper in ranges:
                    range_width = upper - lower
                    # Make ranges extremely wide (ineffective)
                    bad_lower = lower - range_width * 1.0  # Double the width
                    bad_upper = upper + range_width * 1.0
                    bad_ranges.append((bad_lower, bad_upper))
                    logger.info(f"   💥 Extremely wide range: {lower:.2f}-{upper:.2f} -> {bad_lower:.2f}-{bad_upper:.2f}")
                
                ranges = bad_ranges
                
                # Severely reduce liquidity
                adjusted_liquidities = [liq * 0.1 for liq in adjusted_liquidities]  # 90% reduction
                logger.info(f"   💥 Severely reduced liquidity by 90%")
        
        # 6. NEW: Sometimes completely misplace ranges
        if self.imperfection_level > 0.8:
            if random.random() < 0.2:  # 20% chance to completely misplace ranges
                logger.info(f"🎯 COMPLETELY MISPLACING ranges for extreme imperfection!")
                
                # Move all ranges to completely wrong price levels
                misplaced_ranges = []
                for lower, upper in ranges:
                    range_width = upper - lower
                    # Move to completely wrong price level (e.g., 50% away from current price)
                    wrong_center = current_price * (0.5 + random.random())  # 50% to 150% of current price
                    wrong_lower = wrong_center - range_width / 2
                    wrong_upper = wrong_center + range_width / 2
                    misplaced_ranges.append((wrong_lower, wrong_upper))
                    logger.info(f"   🎯 Misplaced range: {lower:.2f}-{upper:.2f} -> {wrong_lower:.2f}-{wrong_upper:.2f}")
                
                ranges = misplaced_ranges
        
        return ranges, adjusted_liquidities
    
    def get_imperfection_stats(self):
        """Get statistics about imperfection effects."""
        return {
            'imperfection_level': self.imperfection_level,
            'rebalance_count': self.rebalance_count,
            'failed_rebalances': self.failed_rebalances,
            'forced_no_rebalance_periods': self.forced_no_rebalance_periods,
            'consecutive_failures': self.consecutive_failures,
            'market_stress_mode': self.market_stress_mode,
            'failure_rate': self.failed_rebalances / max(1, self.rebalance_count + self.failed_rebalances)
        }
