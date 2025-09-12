"""
Demo script for quantum machine learning strategies in AMM backtester.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create DataFrame
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def test_quantum_models():
    """Test quantum models without strategies."""
    try:
        from src.ml import (
            QUANTUM_AVAILABLE,
            QNNRebalancePredictor,
            QSVMVolatilityPredictor,
            PennyLaneQNNPredictor
        )
        
        if not QUANTUM_AVAILABLE:
            logger.warning("Quantum libraries not available. Skipping quantum model tests.")
            return
        
        logger.info("Testing quantum models...")
        
        # Generate sample data
        data = generate_sample_data(500)
        
        # Create features
        from src.ml import FeatureEngineer
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(data)
        
        if len(features_df) == 0:
            logger.error("No features generated")
            return
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) == 0:
            logger.error("No feature columns found")
            return
        
        X = features_df[feature_cols].values
        y_rebalance = (features_df['close'].pct_change().abs() > 0.01).astype(int).values[1:]
        y_volatility = (features_df['close'].rolling(20).std() > features_df['close'].rolling(20).std().mean()).astype(int).values[20:]
        
        # Adjust X to match y lengths
        X_rebalance = X[1:len(y_rebalance)+1]
        X_volatility = X[20:len(y_volatility)+20]
        
        logger.info(f"Training data shapes: X_rebalance={X_rebalance.shape}, y_rebalance={y_rebalance.shape}")
        logger.info(f"Training data shapes: X_volatility={X_volatility.shape}, y_volatility={y_volatility.shape}")
        
        # Test QNN Rebalance Predictor
        try:
            logger.info("Testing QNN Rebalance Predictor...")
            qnn_model = QNNRebalancePredictor(n_qubits=2, n_layers=1, feature_dim=4)
            
            # Use smaller dataset for quantum training
            X_small = X_rebalance[:50]
            y_small = y_rebalance[:50]
            
            # Ensure correct feature dimension
            if X_small.shape[1] > 4:
                X_small = X_small[:, :4]
            elif X_small.shape[1] < 4:
                padding = np.zeros((X_small.shape[0], 4 - X_small.shape[1]))
                X_small = np.hstack([X_small, padding])
            
            qnn_model.fit(X_small, y_small)
            
            # Test predictions
            test_X = X_small[:5]
            predictions = qnn_model.predict(test_X)
            probabilities = qnn_model.predict_proba(test_X)
            
            logger.info(f"QNN Predictions: {predictions}")
            logger.info(f"QNN Probabilities shape: {probabilities.shape}")
            
        except Exception as e:
            logger.error(f"QNN test failed: {e}")
        
        # Test QSVM Volatility Predictor
        try:
            logger.info("Testing QSVM Volatility Predictor...")
            qsvm_model = QSVMVolatilityPredictor(n_qubits=2, feature_dim=4)
            
            # Use smaller dataset
            X_small = X_volatility[:50]
            y_small = y_volatility[:50]
            
            # Ensure correct feature dimension
            if X_small.shape[1] > 4:
                X_small = X_small[:, :4]
            elif X_small.shape[1] < 4:
                padding = np.zeros((X_small.shape[0], 4 - X_small.shape[1]))
                X_small = np.hstack([X_small, padding])
            
            qsvm_model.fit(X_small, y_small)
            
            # Test predictions
            test_X = X_small[:5]
            predictions = qsvm_model.predict(test_X)
            probabilities = qsvm_model.predict_proba(test_X)
            
            logger.info(f"QSVM Predictions: {predictions}")
            logger.info(f"QSVM Probabilities shape: {probabilities.shape}")
            
        except Exception as e:
            logger.error(f"QSVM test failed: {e}")
        
        logger.info("Quantum model tests completed")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")

def test_quantum_strategies():
    """Test quantum strategies."""
    try:
        from src.strategies import (
            QUANTUM_STRATEGIES_AVAILABLE,
            QuantumBasedStrategy,
            QuantumVolatilityStrategy,
            QuantumHybridStrategy
        )
        
        if not QUANTUM_STRATEGIES_AVAILABLE:
            logger.warning("Quantum strategies not available. Skipping quantum strategy tests.")
            return
        
        logger.info("Testing quantum strategies...")
        
        # Generate sample data
        data = generate_sample_data(200)
        
        # Test QuantumBasedStrategy
        try:
            logger.info("Testing QuantumBasedStrategy...")
            strategy = QuantumBasedStrategy(
                quantum_model_type="qnn_rebalance",
                n_qubits=2,
                n_layers=1,
                feature_dim=4,
                rebalance_threshold=0.3
            )
            
            # Test strategy decisions
            rebalance_decisions = []
            position_widths = []
            
            for i in range(50, len(data)):
                current_data = data.iloc[:i+1]
                current_price = data['close'].iloc[i]
                current_time = data.index[i]
                
                # Test should_rebalance
                should_rebalance = strategy.should_rebalance(
                    current_price=current_price,
                    current_time=current_time,
                    price_data=current_data
                )
                
                # Test calculate_ranges
                ranges, liquidities = strategy.calculate_ranges(
                    price_data=current_data,
                    current_price=current_price,
                    portfolio_value=10000
                )
                
                rebalance_decisions.append(should_rebalance)
                if ranges:
                    width_pct = (ranges[0][1] - ranges[0][0]) / current_price
                    position_widths.append(width_pct)
                else:
                    position_widths.append(0)
            
            logger.info(f"QuantumBasedStrategy - Rebalance decisions: {sum(rebalance_decisions)}/{len(rebalance_decisions)}")
            logger.info(f"QuantumBasedStrategy - Avg position width: {np.mean(position_widths):.3f}")
            
            # Get strategy info
            strategy_info = strategy.get_strategy_info()
            logger.info(f"QuantumBasedStrategy info: {strategy_info}")
            
        except Exception as e:
            logger.error(f"QuantumBasedStrategy test failed: {e}")
        
        # Test QuantumVolatilityStrategy
        try:
            logger.info("Testing QuantumVolatilityStrategy...")
            strategy = QuantumVolatilityStrategy(
                quantum_model_type="qsvm_volatility",
                n_qubits=2,
                feature_dim=4,
                base_width=0.1,
                volatility_sensitivity=2.0
            )
            
            # Test strategy decisions
            rebalance_decisions = []
            position_widths = []
            
            for i in range(50, len(data)):
                current_data = data.iloc[:i+1]
                current_price = data['close'].iloc[i]
                current_time = data.index[i]
                
                # Test should_rebalance
                should_rebalance = strategy.should_rebalance(
                    current_price=current_price,
                    current_time=current_time,
                    price_data=current_data
                )
                
                # Test calculate_ranges
                ranges, liquidities = strategy.calculate_ranges(
                    price_data=current_data,
                    current_price=current_price,
                    portfolio_value=10000
                )
                
                rebalance_decisions.append(should_rebalance)
                if ranges:
                    width_pct = (ranges[0][1] - ranges[0][0]) / current_price
                    position_widths.append(width_pct)
                else:
                    position_widths.append(0)
            
            logger.info(f"QuantumVolatilityStrategy - Rebalance decisions: {sum(rebalance_decisions)}/{len(rebalance_decisions)}")
            logger.info(f"QuantumVolatilityStrategy - Avg position width: {np.mean(position_widths):.3f}")
            
            # Get strategy info
            strategy_info = strategy.get_strategy_info()
            logger.info(f"QuantumVolatilityStrategy info: {strategy_info}")
            
        except Exception as e:
            logger.error(f"QuantumVolatilityStrategy test failed: {e}")
        
        # Test QuantumHybridStrategy
        try:
            logger.info("Testing QuantumHybridStrategy...")
            strategy = QuantumHybridStrategy(
                rebalance_model_type="qnn_rebalance",
                volatility_model_type="qsvm_volatility",
                n_qubits=2,
                n_layers=1,
                feature_dim=4,
                rebalance_weight=0.6,
                volatility_weight=0.4
            )
            
            # Test strategy decisions
            rebalance_decisions = []
            position_widths = []
            
            for i in range(50, len(data)):
                current_data = data.iloc[:i+1]
                current_price = data['close'].iloc[i]
                current_time = data.index[i]
                
                # Test should_rebalance
                should_rebalance = strategy.should_rebalance(
                    current_price=current_price,
                    current_time=current_time,
                    price_data=current_data
                )
                
                # Test calculate_ranges
                ranges, liquidities = strategy.calculate_ranges(
                    price_data=current_data,
                    current_price=current_price,
                    portfolio_value=10000
                )
                
                rebalance_decisions.append(should_rebalance)
                if ranges:
                    width_pct = (ranges[0][1] - ranges[0][0]) / current_price
                    position_widths.append(width_pct)
                else:
                    position_widths.append(0)
            
            logger.info(f"QuantumHybridStrategy - Rebalance decisions: {sum(rebalance_decisions)}/{len(rebalance_decisions)}")
            logger.info(f"QuantumHybridStrategy - Avg position width: {np.mean(position_widths):.3f}")
            
            # Get strategy info
            strategy_info = strategy.get_strategy_info()
            logger.info(f"QuantumHybridStrategy info: {strategy_info}")
            
        except Exception as e:
            logger.error(f"QuantumHybridStrategy test failed: {e}")
        
        logger.info("Quantum strategy tests completed")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")

def main():
    """Main demo function."""
    logger.info("Starting quantum strategies demo...")
    
    # Test quantum models
    test_quantum_models()
    
    # Test quantum strategies
    test_quantum_strategies()
    
    logger.info("Quantum strategies demo completed!")

if __name__ == "__main__":
    main()
