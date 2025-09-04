#!/usr/bin/env python3
"""
Simple test runner for Steer Intent Backtester.
"""

import sys
import os
import subprocess

def run_tests():
    """Run the test suite."""
    print("Running Steer Intent Backtester Tests")
    print("=" * 50)
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Try to import pytest
        import pytest
        
        # Run tests
        test_dir = os.path.join(os.path.dirname(__file__), "tests")
        if os.path.exists(test_dir):
            print(f"Running tests from: {test_dir}")
            result = pytest.main([test_dir, "-v"])
            return result == 0
        else:
            print("Tests directory not found")
            return False
            
    except ImportError:
        print("pytest not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
            import pytest
            
            # Run tests again
            test_dir = os.path.join(os.path.dirname(__file__), "tests")
            if os.path.exists(test_dir):
                print(f"Running tests from: {test_dir}")
                result = pytest.main([test_dir, "-v"])
                return result == 0
            else:
                print("Tests directory not found")
                return False
                
        except Exception as e:
            print(f"Failed to install pytest: {e}")
            return False

def run_example():
    """Run the example script."""
    print("\nRunning Example Script")
    print("=" * 30)
    
    try:
        # Import and run example
        from example import run_basic_backtest, run_multiple_strategies
        
        print("Running basic backtest...")
        run_basic_backtest()
        
        print("\nRunning multiple strategies comparison...")
        run_multiple_strategies()
        
        print("\nExample completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Steer Intent Backtester - Test Runner")
    print("=" * 50)
    
    # Run tests
    tests_passed = run_tests()
    
    if tests_passed:
        print("\n✅ All tests passed!")
        
        # Run example
        print("\n" + "=" * 50)
        example_success = run_example()
        
        if example_success:
            print("\n✅ Example completed successfully!")
            print("\n🎉 All checks passed! The system is ready to use.")
        else:
            print("\n❌ Example failed. Check the output above.")
            sys.exit(1)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
