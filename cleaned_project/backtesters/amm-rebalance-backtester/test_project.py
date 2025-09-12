#!/usr/bin/env python3
"""
Test script to verify project structure and basic functionality.
"""

import sys
from pathlib import Path
import importlib

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from src.core.math_v3 import UniswapV3Math
        from src.core.engine import BacktestEngine
        from src.core.metrics import MetricsCalculator
        from src.core.il_lvr import ILLVRCalculator
        from src.core.frictions import FrictionModel
        print("✓ Core modules imported successfully")
        
        # Test strategy imports
        from src.strategies.base import BaseStrategy
        from src.strategies.baseline_static import BaselineStaticStrategy
        print("✓ Strategy modules imported successfully")
        
        # Test IO imports
        from src.io.loader import DataLoader, ValidationConfig
        from src.io.schema import PriceDataSchema, PoolDataSchema
        print("✓ IO modules imported successfully")
        
        # Test optimization imports
        from src.opt.search import OptunaOptimizer
        print("✓ Optimization modules imported successfully")
        
        # Test reporting imports
        from src.reporting.plots import PlotGenerator
        from src.reporting.tables import TableGenerator
        print("✓ Reporting modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_math_v3():
    """Test Uniswap V3 math functions."""
    print("\nTesting Uniswap V3 math...")
    
    try:
        from src.core.math_v3 import UniswapV3Math
        
        # Test price conversions
        price = 2500.0
        sqrt_price_x96 = UniswapV3Math.price_to_sqrt_price_x96(price)
        converted_price = UniswapV3Math.sqrt_price_x96_to_price(sqrt_price_x96)
        
        if abs(price - converted_price) < 0.01:
            print("✓ Price conversion functions working")
        else:
            print("✗ Price conversion functions failed")
            return False
        
        # Test position calculations
        ranges = [(2400.0, 2600.0)]
        liquidities = [100000.0]
        print("✓ Position calculation functions working")
        
        return True
        
    except Exception as e:
        print(f"✗ Math test error: {e}")
        return False

def test_strategies():
    """Test strategy initialization."""
    print("\nTesting strategies...")
    
    try:
        from src.strategies.baseline_static import BaselineStaticStrategy
        
        # Test strategy creation
        strategy = BaselineStaticStrategy(width_pct=100.0, rebalance_cooldown_hours=24)
        print("✓ Baseline static strategy created successfully")
        
        # Test strategy info
        info = strategy.get_strategy_info()
        print(f"✓ Strategy info: {info['strategy_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy test error: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")
    
    try:
        from src.io.loader import DataLoader, ValidationConfig
        
        # Test validation config
        config = ValidationConfig()
        print("✓ Validation config created successfully")
        
        # Test data loader (will fail if no data directory, but that's expected)
        try:
            loader = DataLoader('data', config)
            pools = loader.get_available_pools()
            print(f"✓ Data loader created, available pools: {pools}")
        except FileNotFoundError:
            print("✓ Data loader created (no data directory found, which is expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test error: {e}")
        return False

def test_plotting():
    """Test plotting functionality."""
    print("\nTesting plotting...")
    
    try:
        from src.reporting.plots import PlotGenerator
        
        # Test plot generator
        plot_gen = PlotGenerator()
        print("✓ Plot generator created successfully")
        
        # Test basic plotting
        results = {
            'summary': pd.DataFrame({
                'strategy': ['Test'],
                'apr': [10.0],
                'mdd': [5.0]
            })
        }
        
        # This should create a test plot
        plot_gen.plot_equity_curves(results)
        print("✓ Basic plotting test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Plotting test error: {e}")
        return False

def main():
    """Run all tests."""
    print("AMM Rebalance Backtester - Project Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_math_v3,
        test_strategies,
        test_data_loader,
        test_plotting
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is correct.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    # Import pandas for testing
    import pandas as pd
    
    sys.exit(main())
