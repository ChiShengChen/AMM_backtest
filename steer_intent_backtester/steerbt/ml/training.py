"""
Machine learning model training utilities for Steer Intent strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from .feature_engineering import SteerFeatureEngineer
from .models import IntentPredictor, PricePredictor, SteerMLStrategy, SteerMLBollingerStrategy

logger = logging.getLogger(__name__)

class SteerMLTrainer:
    """Trainer for ML-based Steer Intent strategies."""
    
    def __init__(self, 
                 feature_engineer: SteerFeatureEngineer,
                 models_dir: str = "models",
                 test_size: float = 0.2,
                 validation_size: float = 0.2):
        self.feature_engineer = feature_engineer
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.test_size = test_size
        self.validation_size = validation_size
        
        self.training_results = {}
        
    def prepare_training_data(self, 
                            price_data: pd.DataFrame,
                            strategy_type: str = 'bollinger') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for Steer ML models.
        
        Args:
            price_data: Historical price data
            strategy_type: Type of strategy to create targets for
            
        Returns:
            Tuple of (features_df, intent_targets, price_targets)
        """
        logger.info(f"Preparing training data for {strategy_type} strategy...")
        
        # Create features
        features_df = self.feature_engineer.create_features(price_data)
        
        # Create targets
        targets_df = self.feature_engineer.create_targets(features_df, strategy_type)
        
        # Separate features and targets
        feature_cols = [col for col in targets_df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'future_return_1', 'future_return_5', 'future_return_20',
                                     'should_rebalance', 'bb_signal', 'keltner_signal', 'donchian_signal']]
        
        X = targets_df[feature_cols].copy()
        y_intent = targets_df['should_rebalance'].copy()
        y_price = targets_df['future_return_1'].copy()
        
        # Remove rows with NaN targets
        valid_mask = ~(y_intent.isna() | y_price.isna())
        X = X[valid_mask]
        y_intent = y_intent[valid_mask]
        y_price = y_price[valid_mask]
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Intent target distribution: {y_intent.value_counts().to_dict()}")
        logger.info(f"Price target stats: mean={y_price.mean():.4f}, std={y_price.std():.4f}")
        
        return X, y_intent, y_price
    
    def train_intent_model(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          model_type: str = 'random_forest') -> IntentPredictor:
        """
        Train intent prediction model.
        
        Args:
            X: Feature matrix
            y: Intent targets (binary)
            model_type: Type of model to train
            
        Returns:
            Trained IntentPredictor
        """
        logger.info(f"Training intent prediction model ({model_type})...")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Initialize model
        model = IntentPredictor(model_type=model_type)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Intent model accuracy: {accuracy:.4f}")
        
        # Store results
        self.training_results['intent_model'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'feature_importance': model.feature_importance_,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_targets': y_test
        }
        
        # Save model
        model_path = self.models_dir / f"intent_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(str(model_path))
        
        return model
    
    def train_price_model(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         model_type: str = 'random_forest') -> PricePredictor:
        """
        Train price prediction model.
        
        Args:
            X: Feature matrix
            y: Price targets (continuous)
            model_type: Type of model to train
            
        Returns:
            Trained PricePredictor
        """
        logger.info(f"Training price prediction model ({model_type})...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        # Initialize model
        model = PricePredictor(model_type=model_type)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Price model RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Store results
        self.training_results['price_model'] = {
            'model_type': model_type,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': model.feature_importance_,
            'test_predictions': y_pred,
            'test_targets': y_test
        }
        
        # Save model
        model_path = self.models_dir / f"price_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(str(model_path))
        
        return model
    
    def train_steer_ml_strategy(self, 
                               price_data: pd.DataFrame,
                               strategy_type: str = 'bollinger',
                               intent_model_type: str = 'random_forest',
                               price_model_type: str = 'random_forest') -> SteerMLStrategy:
        """
        Train complete Steer ML strategy.
        
        Args:
            price_data: Historical price data
            strategy_type: Type of strategy
            intent_model_type: Type of intent model
            price_model_type: Type of price model
            
        Returns:
            Trained SteerMLStrategy
        """
        logger.info(f"Training complete Steer ML strategy ({strategy_type})...")
        
        # Prepare training data
        X, y_intent, y_price = self.prepare_training_data(price_data, strategy_type)
        
        # Fit feature scaler
        self.feature_engineer.fit_scaler(X)
        
        # Transform features
        X_scaled = self.feature_engineer.transform_features(X)
        
        # Train models
        intent_model = self.train_intent_model(X_scaled, y_intent, intent_model_type)
        price_model = self.train_price_model(X_scaled, y_price, price_model_type)
        
        # Create Steer ML strategy
        steer_ml_strategy = SteerMLStrategy(
            intent_model=intent_model,
            price_model=price_model,
            feature_engineer=self.feature_engineer,
            strategy_type=strategy_type,
            rebalance_threshold=0.5,  # Probability threshold
            price_threshold=0.02
        )
        
        logger.info("Steer ML strategy training completed successfully")
        
        return steer_ml_strategy
    
    def train_ml_bollinger_strategy(self, 
                                   price_data: pd.DataFrame,
                                   intent_model_type: str = 'random_forest',
                                   n: int = 20,
                                   k: float = 2.0,
                                   ml_weight: float = 0.7) -> SteerMLBollingerStrategy:
        """
        Train ML-enhanced Bollinger Bands strategy.
        
        Args:
            price_data: Historical price data
            intent_model_type: Type of intent model
            n: Bollinger Bands period
            k: Bollinger Bands standard deviation multiplier
            ml_weight: Weight for ML predictions
            
        Returns:
            Trained SteerMLBollingerStrategy
        """
        logger.info("Training ML-enhanced Bollinger Bands strategy...")
        
        # Prepare training data
        X, y_intent, y_price = self.prepare_training_data(price_data, 'bollinger')
        
        # Fit feature scaler
        self.feature_engineer.fit_scaler(X)
        
        # Transform features
        X_scaled = self.feature_engineer.transform_features(X)
        
        # Train intent model
        intent_model = self.train_intent_model(X_scaled, y_intent, intent_model_type)
        
        # Create ML Bollinger strategy
        ml_bollinger_strategy = SteerMLBollingerStrategy(
            intent_model=intent_model,
            feature_engineer=self.feature_engineer,
            n=n,
            k=k,
            ml_weight=ml_weight,
            traditional_weight=1.0 - ml_weight
        )
        
        logger.info("ML Bollinger strategy training completed successfully")
        
        return ml_bollinger_strategy
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models and return performance metrics."""
        if not self.training_results:
            logger.warning("No training results available for evaluation")
            return {}
        
        evaluation_results = {}
        
        # Evaluate intent model
        if 'intent_model' in self.training_results:
            intent_results = self.training_results['intent_model']
            evaluation_results['intent_model'] = {
                'accuracy': intent_results['accuracy'],
                'top_features': intent_results['feature_importance'].head(10).to_dict() if intent_results['feature_importance'] is not None else {}
            }
        
        # Evaluate price model
        if 'price_model' in self.training_results:
            price_results = self.training_results['price_model']
            evaluation_results['price_model'] = {
                'rmse': price_results['rmse'],
                'mae': price_results['mae'],
                'top_features': price_results['feature_importance'].head(10).to_dict() if price_results['feature_importance'] is not None else {}
            }
        
        return evaluation_results
    
    def save_training_results(self, filepath: str) -> None:
        """Save training results to file."""
        results_data = {
            'training_results': self.training_results,
            'feature_engineer_config': {
                'lookback_periods': self.feature_engineer.lookback_periods,
                'is_fitted': self.feature_engineer.is_fitted
            },
            'training_config': {
                'test_size': self.test_size,
                'validation_size': self.validation_size
            }
        }
        
        joblib.dump(results_data, filepath)
        logger.info(f"Saved training results to {filepath}")
    
    def load_training_results(self, filepath: str) -> None:
        """Load training results from file."""
        results_data = joblib.load(filepath)
        self.training_results = results_data['training_results']
        logger.info(f"Loaded training results from {filepath}")
    
    def create_model_comparison(self, 
                              X: pd.DataFrame, 
                              y_intent: pd.Series, 
                              y_price: pd.Series) -> Dict[str, Any]:
        """
        Compare different model types for both tasks.
        
        Args:
            X: Feature matrix
            y_intent: Intent targets
            y_price: Price targets
            
        Returns:
            Comparison results
        """
        logger.info("Creating model comparison...")
        
        # Model types to compare
        intent_models = ['random_forest', 'gradient_boosting', 'neural_network', 'svm']
        price_models = ['random_forest', 'gradient_boosting', 'neural_network', 'ridge']
        
        comparison_results = {
            'intent_models': {},
            'price_models': {}
        }
        
        # Compare intent models
        for model_type in intent_models:
            try:
                model = self.train_intent_model(X, y_intent, model_type)
                comparison_results['intent_models'][model_type] = {
                    'accuracy': self.training_results['intent_model']['accuracy']
                }
            except Exception as e:
                logger.error(f"Error training intent model {model_type}: {e}")
                comparison_results['intent_models'][model_type] = {'error': str(e)}
        
        # Compare price models
        for model_type in price_models:
            try:
                model = self.train_price_model(X, y_price, model_type)
                comparison_results['price_models'][model_type] = {
                    'rmse': self.training_results['price_model']['rmse'],
                    'mae': self.training_results['price_model']['mae']
                }
            except Exception as e:
                logger.error(f"Error training price model {model_type}: {e}")
                comparison_results['price_models'][model_type] = {'error': str(e)}
        
        return comparison_results
