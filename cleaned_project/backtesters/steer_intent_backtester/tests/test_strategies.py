"""
Unit tests for strategy implementations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from steerbt.strategies import (
    ClassicStrategy, ChannelMultiplierStrategy, BollingerStrategy,
    KeltnerStrategy, DonchianStrategy, StableStrategy, FluidStrategy
)

class TestStrategies:
    """Test strategy implementations."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        base_price = 2000.0
        returns = np.random.normal(0, 0.01, len(dates))  # 1% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.price_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'high': prices * (1 + abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates)),
            'quote_volume': prices * np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        self.current_price = prices[-1]
        self.portfolio_value = 10000.0
    
    def test_classic_strategy(self):
        """Test classic rebalancing strategy."""
        strategy = ClassicStrategy(
            width_mode="percent",
            width_value=10.0,
            placement_mode="center"
        )
        
        # Test initialization
        assert not strategy.initialized
        
        # Test range calculation
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        assert len(ranges) == len(liquidities)
        
        # Test that ranges are valid
        for lower, upper in ranges:
            assert lower < upper
            assert lower > 0
            assert upper > 0
        
        # Test initialization
        strategy.initialize(self.current_price, self.portfolio_value, self.price_data)
        assert strategy.initialized
    
    def test_channel_multiplier_strategy(self):
        """Test channel multiplier strategy."""
        strategy = ChannelMultiplierStrategy(width_pct=15.0)
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) == 1  # Single position
        assert len(liquidities) == 1
        
        lower, upper = ranges[0]
        center = (lower + upper) / 2
        width_pct = (upper - lower) / center * 100
        
        # Width should be approximately 15%
        assert abs(width_pct - 15.0) < 1.0
    
    def test_bollinger_strategy(self):
        """Test Bollinger Bands strategy."""
        strategy = BollingerStrategy(n=20, k=2.0)
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        
        # Test bands info
        bands_info = strategy.get_bands_info()
        assert 'sma' in bands_info
        assert 'std' in bands_info
        assert 'upper_band' in bands_info
        assert 'lower_band' in bands_info
    
    def test_keltner_strategy(self):
        """Test Keltner Channels strategy."""
        strategy = KeltnerStrategy(n=20, m=2.0)
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        
        # Test channels info
        channels_info = strategy.get_channels_info()
        assert 'ema' in channels_info
        assert 'atr' in channels_info
        assert 'upper_channel' in channels_info
        assert 'lower_channel' in channels_info
    
    def test_donchian_strategy(self):
        """Test Donchian Channels strategy."""
        strategy = DonchianStrategy(n=20)
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        
        # Test channels info
        channels_info = strategy.get_channels_info()
        assert 'highest_high' in channels_info
        assert 'lowest_low' in channels_info
        assert 'center' in channels_info
    
    def test_stable_strategy(self):
        """Test stable strategy."""
        strategy = StableStrategy(
            peg_method="sma",
            width_pct=20.0,
            curve_type="gaussian"
        )
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        
        # Test peg info
        peg_info = strategy.get_peg_info()
        assert 'peg_price' in peg_info
        assert 'peg_method' in peg_info
    
    def test_fluid_strategy(self):
        """Test fluid strategy."""
        strategy = FluidStrategy(
            ideal_ratio=1.0,
            acceptable_ratio=0.1,
            sprawl_type="dynamic"
        )
        
        ranges, liquidities = strategy.calculate_range(
            self.price_data, self.current_price, self.portfolio_value
        )
        
        assert len(ranges) > 0
        assert len(liquidities) > 0
        
        # Test strategy info
        strategy_info = strategy.get_strategy_info()
        assert 'ideal_ratio' in strategy_info
        assert 'current_state' in strategy_info
    
    def test_strategy_validation(self):
        """Test strategy parameter validation."""
        # Test missing required parameters
        with pytest.raises(ValueError):
            ClassicStrategy()  # Missing required params
        
        with pytest.raises(ValueError):
            BollingerStrategy()  # Missing required params
    
    def test_strategy_reset(self):
        """Test strategy reset functionality."""
        strategy = ClassicStrategy(
            width_mode="percent",
            width_value=10.0,
            placement_mode="center"
        )
        
        # Initialize strategy
        strategy.initialize(self.current_price, self.portfolio_value, self.price_data)
        assert strategy.initialized
        
        # Reset strategy
        strategy.reset()
        assert not strategy.initialized
        assert strategy.rebalance_count == 0
    
    def test_strategy_update(self):
        """Test strategy update functionality."""
        strategy = ClassicStrategy(
            width_mode="percent",
            width_value=10.0,
            placement_mode="center"
        )
        
        # First update should initialize
        should_rebalance = strategy.update(
            self.price_data, self.current_price, self.portfolio_value
        )
        assert should_rebalance
        assert strategy.initialized
        
        # Second update with same data should not rebalance
        should_rebalance = strategy.update(
            self.price_data, self.current_price, self.portfolio_value
        )
        assert not should_rebalance
    
    def test_strategy_parameters(self):
        """Test strategy parameter handling."""
        strategy = ClassicStrategy(
            width_mode="percent",
            width_value=10.0,
            placement_mode="center",
            curve_type="gaussian",
            max_positions=3
        )
        
        strategy_info = strategy.get_strategy_info()
        assert strategy_info['width_mode'] == 'percent'
        assert strategy_info['width_value'] == 10.0
        assert strategy_info['placement_mode'] == 'center'
        assert strategy_info['curve_type'] == 'gaussian'
        assert strategy_info['max_positions'] == 3
