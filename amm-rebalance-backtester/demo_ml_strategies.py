#!/usr/bin/env python3
"""
Demo script for ML-based AMM strategies.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ml import FeatureEngineer, MLTrainer, RebalancePredictor, VolatilityPredictor, MLStrategy
from src.strategies import MLBasedStrategy, MLVolatilityStrategy, MLHybridStrategy
from src.io.loader import DataLoader, ValidationConfig
from src.core.engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(days: int = 365) -> pd.DataFrame:
    """Create sample price data for demonstration."""
    logger.info(f"Creating {days} days of sample price data...")
    
    # Generate realistic price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Start with base price
    base_price = 2000.0
    prices = [base_price]
    
    # Generate price movements with some trend and volatility
    for i in range(1, days):
        # Add some trend and volatility
        trend = 0.0001  # Slight upward trend
        volatility = 0.02  # 2% daily volatility
        random_shock = np.random.normal(0, volatility)
        
        new_price = prices[-1] * (1 + trend + random_shock)
        prices.append(max(new_price, 100))  # Minimum price floor
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_vol = abs(np.random.normal(0, 0.01))
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        open_price = close * (1 + np.random.normal(0, 0.005))
        
        # Generate volume (higher on volatile days)
        base_volume = 1000000
        volume_multiplier = 1 + abs(random_shock) * 10
        volume = int(base_volume * volume_multiplier)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Created sample data: {len(df)} days, price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def train_ml_models(price_data: pd.DataFrame) -> tuple:
    """Train ML models on the price data."""
    logger.info("Training ML models...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(lookback_periods=50)
    
    # Initialize trainer
    trainer = MLTrainer(feature_engineer, models_dir="models")
    
    # Train complete ML strategy
    ml_strategy = trainer.train_ml_strategy(
        price_data=price_data,
        rebalance_model_type='random_forest',
        volatility_model_type='random_forest',
        rebalance_threshold=0.02
    )
    
    # Get evaluation results
    evaluation_results = trainer.evaluate_models()
    
    logger.info("ML model training completed")
    logger.info(f"Evaluation results: {evaluation_results}")
    
    return ml_strategy, trainer, evaluation_results

def test_ml_strategies(price_data: pd.DataFrame, ml_strategy: MLStrategy) -> dict:
    """Test different ML strategies."""
    logger.info("Testing ML strategies...")
    
    # Split data for testing
    train_size = int(len(price_data) * 0.7)
    train_data = price_data.iloc[:train_size]
    test_data = price_data.iloc[train_size:]
    
    logger.info(f"Training on {len(train_data)} days, testing on {len(test_data)} days")
    
    # Test ML-based strategy
    ml_based_strategy = MLBasedStrategy(
        ml_strategy=ml_strategy,
        initial_width=0.1,
        rebalance_cooldown_hours=1
    )
    
    # Test ML volatility strategy
    ml_volatility_strategy = MLVolatilityStrategy(
        volatility_model=ml_strategy.volatility_model,
        feature_engineer=ml_strategy.feature_engineer,
        base_k_width=1.5,
        rebalance_cooldown_hours=6
    )
    
    # Test hybrid strategy
    ml_hybrid_strategy = MLHybridStrategy(
        ml_strategy=ml_strategy,
        traditional_weight=0.3,
        ml_weight=0.7,
        rebalance_cooldown_hours=4
    )
    
    # Simulate strategy decisions on test data
    strategies = {
        'ML-Based': ml_based_strategy,
        'ML-Volatility': ml_volatility_strategy,
        'ML-Hybrid': ml_hybrid_strategy
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testing {strategy_name} strategy...")
        
        rebalance_decisions = []
        position_widths = []
        
        # Test on a subset of test data
        test_subset = test_data.iloc[:100]  # Test on first 100 days
        
        for i in range(20, len(test_subset)):  # Start after lookback period
            current_data = test_subset.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            # Get strategy decision
            ranges = strategy.calculate_ranges(
                price_data=current_data,
                current_price=current_price,
                portfolio_value=10000
            )
            
            rebalance_decisions.append(ranges['should_rebalance'])
            position_widths.append(ranges.get('width_pct', 0) / 100)
        
        # Calculate strategy statistics
        rebalance_rate = np.mean(rebalance_decisions)
        avg_width = np.mean(position_widths)
        width_std = np.std(position_widths)
        
        results[strategy_name] = {
            'rebalance_rate': rebalance_rate,
            'avg_position_width': avg_width,
            'width_std': width_std,
            'total_decisions': len(rebalance_decisions),
            'strategy_info': strategy.get_strategy_info()
        }
        
        logger.info(f"{strategy_name}: rebalance_rate={rebalance_rate:.3f}, "
                   f"avg_width={avg_width:.3f}, width_std={width_std:.3f}")
    
    return results

def run_ml_backtest_demo():
    """Run complete ML backtesting demonstration."""
    logger.info("Starting ML AMM Strategy Demo")
    logger.info("=" * 50)
    
    try:
        # Create sample data
        price_data = create_sample_data(days=500)
        
        # Train ML models
        ml_strategy, trainer, evaluation_results = train_ml_models(price_data)
        
        # Test ML strategies
        strategy_results = test_ml_strategies(price_data, ml_strategy)
        
        # Print results
        print("\n" + "=" * 50)
        print("ML STRATEGY DEMO RESULTS")
        print("=" * 50)
        
        print("\nModel Evaluation Results:")
        for model_type, results in evaluation_results.items():
            print(f"\n{model_type}:")
            for metric, value in results.items():
                if isinstance(value, dict):
                    print(f"  {metric}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {metric}: {value}")
        
        print("\nStrategy Performance:")
        for strategy_name, results in strategy_results.items():
            print(f"\n{strategy_name}:")
            print(f"  Rebalance Rate: {results['rebalance_rate']:.3f}")
            print(f"  Avg Position Width: {results['avg_position_width']:.3f}")
            print(f"  Width Std Dev: {results['width_std']:.3f}")
            print(f"  Total Decisions: {results['total_decisions']}")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_ml_backtest_demo()
    sys.exit(0 if success else 1)
