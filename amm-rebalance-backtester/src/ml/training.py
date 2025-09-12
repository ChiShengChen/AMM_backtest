"""
Machine learning model training utilities.
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

from .feature_engineering import FeatureEngineer
from .models import RebalancePredictor, VolatilityPredictor, MLStrategy

logger = logging.getLogger(__name__)

class MLTrainer:
    """Trainer for ML-based AMM strategies."""
    
    def __init__(self, 
                 feature_engineer: FeatureEngineer,
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
                            rebalance_threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for ML models.
        
        Args:
            price_data: Historical price data
            rebalance_threshold: Threshold for rebalancing decisions
            
        Returns:
            Tuple of (features_df, rebalance_targets, volatility_targets)
        """
        logger.info("Preparing training data...")
        
        # Create features
        features_df = self.feature_engineer.create_features(price_data)
        
        # Create targets
        targets_df = self.feature_engineer.create_targets(features_df, rebalance_threshold)
        
        # Separate features and targets
        feature_cols = [col for col in targets_df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'future_return_1', 'future_return_5', 'future_return_20',
                                     'should_rebalance', 'future_volatility', 'future_max_drawdown']]
        
        X = targets_df[feature_cols].copy()
        y_rebalance = targets_df['should_rebalance'].copy()
        y_volatility = targets_df['future_volatility'].copy()
        
        # Remove rows with NaN targets
        valid_mask = ~(y_rebalance.isna() | y_volatility.isna())
        X = X[valid_mask]
        y_rebalance = y_rebalance[valid_mask]
        y_volatility = y_volatility[valid_mask]
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Rebalance target distribution: {y_rebalance.value_counts().to_dict()}")
        logger.info(f"Volatility target stats: mean={y_volatility.mean():.4f}, std={y_volatility.std():.4f}")
        
        return X, y_rebalance, y_volatility
    
    def train_rebalance_model(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            model_type: str = 'random_forest') -> RebalancePredictor:
        """
        Train rebalancing prediction model.
        
        Args:
            X: Feature matrix
            y: Rebalancing targets (binary)
            model_type: Type of model to train
            
        Returns:
            Trained RebalancePredictor
        """
        logger.info(f"Training rebalance prediction model ({model_type})...")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Initialize model
        model = RebalancePredictor(model_type=model_type)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Rebalance model accuracy: {accuracy:.4f}")
        
        # Store results
        self.training_results['rebalance_model'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'feature_importance': model.feature_importance_,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_targets': y_test
        }
        
        # Save model
        model_path = self.models_dir / f"rebalance_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(str(model_path))
        
        return model
    
    def train_volatility_model(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             model_type: str = 'random_forest') -> VolatilityPredictor:
        """
        Train volatility prediction model.
        
        Args:
            X: Feature matrix
            y: Volatility targets (continuous)
            model_type: Type of model to train
            
        Returns:
            Trained VolatilityPredictor
        """
        logger.info(f"Training volatility prediction model ({model_type})...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        # Initialize model
        model = VolatilityPredictor(model_type=model_type)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Volatility model RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Store results
        self.training_results['volatility_model'] = {
            'model_type': model_type,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': model.feature_importance_,
            'test_predictions': y_pred,
            'test_targets': y_test
        }
        
        # Save model
        model_path = self.models_dir / f"volatility_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(str(model_path))
        
        return model
    
    def train_ml_strategy(self, 
                         price_data: pd.DataFrame,
                         rebalance_model_type: str = 'random_forest',
                         volatility_model_type: str = 'random_forest',
                         rebalance_threshold: float = 0.02) -> MLStrategy:
        """
        Train complete ML strategy.
        
        Args:
            price_data: Historical price data
            rebalance_model_type: Type of rebalancing model
            volatility_model_type: Type of volatility model
            rebalance_threshold: Threshold for rebalancing decisions
            
        Returns:
            Trained MLStrategy
        """
        logger.info("Training complete ML strategy...")
        
        # Prepare training data
        X, y_rebalance, y_volatility = self.prepare_training_data(price_data, rebalance_threshold)
        
        # Fit feature scaler
        self.feature_engineer.fit_scaler(X)
        
        # Transform features
        X_scaled = self.feature_engineer.transform_features(X)
        
        # Train models
        rebalance_model = self.train_rebalance_model(X_scaled, y_rebalance, rebalance_model_type)
        volatility_model = self.train_volatility_model(X_scaled, y_volatility, volatility_model_type)
        
        # Create ML strategy
        ml_strategy = MLStrategy(
            rebalance_model=rebalance_model,
            volatility_model=volatility_model,
            feature_engineer=self.feature_engineer,
            rebalance_threshold=0.5,  # Probability threshold
            volatility_threshold=0.02
        )
        
        logger.info("ML strategy training completed successfully")
        
        return ml_strategy
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models and return performance metrics."""
        if not self.training_results:
            logger.warning("No training results available for evaluation")
            return {}
        
        evaluation_results = {}
        
        # Evaluate rebalance model
        if 'rebalance_model' in self.training_results:
            rebalance_results = self.training_results['rebalance_model']
            evaluation_results['rebalance_model'] = {
                'accuracy': rebalance_results['accuracy'],
                'top_features': rebalance_results['feature_importance'].head(10).to_dict() if rebalance_results['feature_importance'] is not None else {}
            }
        
        # Evaluate volatility model
        if 'volatility_model' in self.training_results:
            volatility_results = self.training_results['volatility_model']
            evaluation_results['volatility_model'] = {
                'rmse': volatility_results['rmse'],
                'mae': volatility_results['mae'],
                'top_features': volatility_results['feature_importance'].head(10).to_dict() if volatility_results['feature_importance'] is not None else {}
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
                              y_rebalance: pd.Series, 
                              y_volatility: pd.Series) -> Dict[str, Any]:
        """
        Compare different model types for both tasks.
        
        Args:
            X: Feature matrix
            y_rebalance: Rebalancing targets
            y_volatility: Volatility targets
            
        Returns:
            Comparison results
        """
        logger.info("Creating model comparison...")
        
        # Model types to compare
        rebalance_models = ['random_forest', 'gradient_boosting', 'neural_network', 'svm']
        volatility_models = ['random_forest', 'gradient_boosting', 'neural_network', 'ridge']
        
        comparison_results = {
            'rebalance_models': {},
            'volatility_models': {}
        }
        
        # Compare rebalancing models
        for model_type in rebalance_models:
            try:
                model = self.train_rebalance_model(X, y_rebalance, model_type)
                comparison_results['rebalance_models'][model_type] = {
                    'accuracy': self.training_results['rebalance_model']['accuracy']
                }
            except Exception as e:
                logger.error(f"Error training rebalance model {model_type}: {e}")
                comparison_results['rebalance_models'][model_type] = {'error': str(e)}
        
        # Compare volatility models
        for model_type in volatility_models:
            try:
                model = self.train_volatility_model(X, y_volatility, model_type)
                comparison_results['volatility_models'][model_type] = {
                    'rmse': self.training_results['volatility_model']['rmse'],
                    'mae': self.training_results['volatility_model']['mae']
                }
            except Exception as e:
                logger.error(f"Error training volatility model {model_type}: {e}")
                comparison_results['volatility_models'][model_type] = {'error': str(e)}
        
        return comparison_results
