#!/usr/bin/env python3
"""
Demo script for ML-based Steer Intent strategies.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from steerbt.ml import SteerFeatureEngineer, SteerMLTrainer, IntentPredictor, PricePredictor, SteerMLStrategy
from steerbt.ml.models import SteerMLBollingerStrategy
from steerbt.strategies import MLBollingerStrategy, MLKeltnerStrategy, MLDonchianStrategy, MLHybridStrategy
from steerbt.data.binance import BinanceDataFetcher

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

def train_steer_ml_models(price_data: pd.DataFrame) -> tuple:
    """Train Steer ML models on the price data."""
    logger.info("Training Steer ML models...")
    
    # Initialize feature engineer
    feature_engineer = SteerFeatureEngineer(lookback_periods=50)
    
    # Initialize trainer
    trainer = SteerMLTrainer(feature_engineer, models_dir="models")
    
    # Train complete Steer ML strategy
    steer_ml_strategy = trainer.train_steer_ml_strategy(
        price_data=price_data,
        strategy_type='bollinger',
        intent_model_type='random_forest',
        price_model_type='random_forest'
    )
    
    # Train ML Bollinger strategy
    ml_bollinger_strategy = trainer.train_ml_bollinger_strategy(
        price_data=price_data,
        intent_model_type='random_forest',
        n=20,
        k=2.0,
        ml_weight=0.7
    )
    
    # Get evaluation results
    evaluation_results = trainer.evaluate_models()
    
    logger.info("Steer ML model training completed")
    logger.info(f"Evaluation results: {evaluation_results}")
    
    return steer_ml_strategy, ml_bollinger_strategy, trainer, evaluation_results

def test_steer_ml_strategies(price_data: pd.DataFrame, 
                           steer_ml_strategy: SteerMLStrategy,
                           ml_bollinger_strategy: SteerMLBollingerStrategy) -> dict:
    """Test different Steer ML strategies."""
    logger.info("Testing Steer ML strategies...")
    
    # Split data for testing
    train_size = int(len(price_data) * 0.7)
    train_data = price_data.iloc[:train_size]
    test_data = price_data.iloc[train_size:]
    
    logger.info(f"Training on {len(train_data)} days, testing on {len(test_data)} days")
    
    # Test ML Bollinger strategy
    ml_bollinger = MLBollingerStrategy(
        ml_bollinger_strategy=ml_bollinger_strategy,
        n=20,
        k=2.0,
        rebalance_cooldown_hours=1
    )
    
    # Test ML Keltner strategy
    ml_keltner = MLKeltnerStrategy(
        intent_model=steer_ml_strategy.intent_model,
        feature_engineer=steer_ml_strategy.feature_engineer,
        n=20,
        m=2.0,
        ml_weight=0.7,
        rebalance_cooldown_hours=1
    )
    
    # Test ML Donchian strategy
    ml_donchian = MLDonchianStrategy(
        intent_model=steer_ml_strategy.intent_model,
        feature_engineer=steer_ml_strategy.feature_engineer,
        n=20,
        ml_weight=0.7,
        rebalance_cooldown_hours=1
    )
    
    # Test hybrid strategy
    ml_hybrid = MLHybridStrategy(
        steer_ml_strategy=steer_ml_strategy,
        traditional_weight=0.3,
        ml_weight=0.7,
        rebalance_cooldown_hours=4
    )
    
    # Simulate strategy decisions on test data
    strategies = {
        'ML-Bollinger': ml_bollinger,
        'ML-Keltner': ml_keltner,
        'ML-Donchian': ml_donchian,
        'ML-Hybrid': ml_hybrid
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testing {strategy_name} strategy...")
        
        rebalance_decisions = []
        position_widths = []
        ml_confidences = []
        
        # Test on a subset of test data
        test_subset = test_data.iloc[:100]  # Test on first 100 days
        
        for i in range(20, len(test_subset)):  # Start after lookback period
            current_data = test_subset.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            # Get strategy decision
            ranges, liquidities = strategy.calculate_range(
                price_data=current_data,
                current_price=current_price,
                portfolio_value=10000
            )
            
            # Extract position information
            if ranges:
                lower_price, upper_price = ranges[0]
                position_width = (upper_price - lower_price) / (2 * current_price)
                position_widths.append(position_width)
            else:
                position_widths.append(0.1)  # Default width
            
            # For Steer Intent strategies, we need to check if rebalancing is needed
            # by comparing with previous position or using strategy's internal logic
            should_rebalance = len(ranges) > 0  # Simple check for now
            rebalance_decisions.append(should_rebalance)
            
            # Extract ML confidence if available (this would need to be stored in strategy)
            ml_confidence = 0.5  # Default confidence
            ml_confidences.append(ml_confidence)
        
        # Calculate strategy statistics
        rebalance_rate = np.mean(rebalance_decisions)
        avg_width = np.mean(position_widths)
        width_std = np.std(position_widths)
        avg_ml_confidence = np.mean(ml_confidences)
        
        results[strategy_name] = {
            'rebalance_rate': rebalance_rate,
            'avg_position_width': avg_width,
            'width_std': width_std,
            'avg_ml_confidence': avg_ml_confidence,
            'total_decisions': len(rebalance_decisions),
            'strategy_info': strategy.get_strategy_info()
        }
        
        logger.info(f"{strategy_name}: rebalance_rate={rebalance_rate:.3f}, "
                   f"avg_width={avg_width:.3f}, width_std={width_std:.3f}, "
                   f"ml_conf={avg_ml_confidence:.3f}")
    
    return results

def run_steer_ml_backtest_demo():
    """Run complete Steer ML backtesting demonstration."""
    logger.info("Starting Steer ML Strategy Demo")
    logger.info("=" * 50)
    
    try:
        # Create sample data
        price_data = create_sample_data(days=500)
        
        # Train Steer ML models
        steer_ml_strategy, ml_bollinger_strategy, trainer, evaluation_results = train_steer_ml_models(price_data)
        
        # Test Steer ML strategies
        strategy_results = test_steer_ml_strategies(price_data, steer_ml_strategy, ml_bollinger_strategy)
        
        # Print results
        print("\n" + "=" * 50)
        print("STEER ML STRATEGY DEMO RESULTS")
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
            print(f"  Avg ML Confidence: {results['avg_ml_confidence']:.3f}")
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
    success = run_steer_ml_backtest_demo()
    sys.exit(0 if success else 1)
