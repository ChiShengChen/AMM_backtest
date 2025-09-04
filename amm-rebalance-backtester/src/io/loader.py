"""
Data loader and validator for AMM backtester.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime, timezone
import yaml

from .schema import PriceDataSchema, PoolDataSchema, DataQualityReport, ValidationConfig

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality assessment."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_price_data(self, df: pd.DataFrame) -> Tuple[bool, DataQualityReport]:
        """Validate price data quality."""
        warnings = []
        errors = []
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, DataQualityReport(
                total_rows=len(df),
                missing_values={},
                outliers={},
                data_quality_score=0.0,
                warnings=warnings,
                errors=errors
            )
        
        # Convert timestamp
        try:
            if df['timestamp'].dtype == 'int64':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            errors.append(f"Timestamp conversion failed: {e}")
        
        # Check for missing values
        missing_values = df[required_cols].isnull().sum().to_dict()
        if any(missing_values.values()):
            warnings.append(f"Missing values detected: {missing_values}")
        
        # Check for outliers
        outliers = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = self.config.outlier_threshold
                outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
                outliers[col] = outlier_mask.sum()
        
        # Check price logic
        if 'high' in df.columns and 'low' in df.columns:
            invalid_high_low = (df['high'] < df['low']).sum()
            if invalid_high_low > 0:
                errors.append(f"Invalid high/low relationship: {invalid_high_low} rows")
        
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            invalid_close = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
            if invalid_close > 0:
                warnings.append(f"Close outside high/low range: {invalid_close} rows")
        
        # Calculate quality score
        total_checks = len(required_cols) + 2  # +2 for logic checks
        passed_checks = total_checks - len(errors)
        quality_score = passed_checks / total_checks
        
        report = DataQualityReport(
            total_rows=len(df),
            missing_values=missing_values,
            outliers=outliers,
            data_quality_score=quality_score,
            warnings=warnings,
            errors=errors
        )
        
        return len(errors) == 0, report
    
    def validate_pool_data(self, df: pd.DataFrame) -> Tuple[bool, DataQualityReport]:
        """Validate pool data quality."""
        warnings = []
        errors = []
        
        if 'timestamp' not in df.columns:
            errors.append("Missing timestamp column")
            return False, DataQualityReport(
                total_rows=len(df),
                missing_values={},
                outliers={},
                data_quality_score=0.0,
                warnings=warnings,
                errors=errors
            )
        
        # Convert timestamp
        try:
            if df['timestamp'].dtype == 'int64':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            errors.append(f"Timestamp conversion failed: {e}")
        
        # Check for missing values
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            warnings.append(f"Missing values detected: {missing_values}")
        
        # Check numeric columns
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    threshold = self.config.outlier_threshold
                    outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
                    outliers[col] = outlier_mask.sum()
        
        # Calculate quality score
        total_checks = 1  # timestamp check
        passed_checks = total_checks - len(errors)
        quality_score = passed_checks / total_checks
        
        report = DataQualityReport(
            total_rows=len(df),
            missing_values=missing_values,
            outliers=outliers,
            data_quality_score=quality_score,
            warnings=warnings,
            errors=errors
        )
        
        return len(errors) == 0, report

class DataLoader:
    """Data loader for AMM backtester."""
    
    def __init__(self, data_dir: str, config: ValidationConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.validator = DataValidator(config)
    
    def load_pool_data(self, pool_name: str, frequency: str = "1h") -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load price and pool data for a specific pool.
        
        Args:
            pool_name: Name of the pool (e.g., 'ETHUSDC')
            frequency: Data frequency (e.g., '1h', '1d')
            
        Returns:
            Tuple of (price_data, pool_data)
        """
        pool_dir = self.data_dir / pool_name
        
        if not pool_dir.exists():
            raise FileNotFoundError(f"Pool directory not found: {pool_dir}")
        
        # Load price data
        price_file = pool_dir / f"price_{frequency}.csv"
        if not price_file.exists():
            raise FileNotFoundError(f"Price file not found: {price_file}")
        
        price_df = pd.read_csv(price_file)
        is_valid, quality_report = self.validator.validate_price_data(price_df)
        
        if not is_valid:
            logger.warning(f"Price data quality issues: {quality_report.errors}")
            if quality_report.data_quality_score < self.config.min_data_quality:
                raise ValueError(f"Price data quality too low: {quality_report.data_quality_score}")
        
        # Set timestamp as index
        price_df.set_index('timestamp', inplace=True)
        price_df.sort_index(inplace=True)
        
        # Load pool data if available
        pool_file = pool_dir / f"pool_{frequency}.csv"
        pool_df = None
        
        if pool_file.exists():
            pool_df = pd.read_csv(pool_file)
            is_valid, quality_report = self.validator.validate_pool_data(pool_df)
            
            if not is_valid:
                logger.warning(f"Pool data quality issues: {quality_report.errors}")
                if quality_report.data_quality_score < self.config.min_data_quality:
                    logger.warning("Pool data quality too low, will use proxy mode")
                    pool_df = None
            else:
                pool_df.set_index('timestamp', inplace=True)
                pool_df.sort_index(inplace=True)
        
        # Align timestamps
        if pool_df is not None:
            price_df, pool_df = self._align_timestamps(price_df, pool_df)
        
        # Clean and validate data
        price_df = self._clean_price_data(price_df)
        if pool_df is not None:
            pool_df = self._clean_pool_data(pool_df)
        
        return price_df, pool_df
    
    def _align_timestamps(self, price_df: pd.DataFrame, pool_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align timestamps between price and pool data."""
        # Find common timestamps
        common_timestamps = price_df.index.intersection(pool_df.index)
        
        if len(common_timestamps) == 0:
            logger.warning("No common timestamps found between price and pool data")
            return price_df, None
        
        # Filter both dataframes to common timestamps
        price_df = price_df.loc[common_timestamps]
        pool_df = pool_df.loc[common_timestamps]
        
        logger.info(f"Aligned {len(common_timestamps)} timestamps between price and pool data")
        return price_df, pool_df
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data."""
        # Remove outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = self.config.outlier_threshold
                outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
                df.loc[outlier_mask, col] = np.nan
        
        # Forward fill and backward fill
        df = df.fillna(method='ffill', limit=self.config.forward_fill_limit)
        df = df.fillna(method='bfill', limit=self.config.backward_fill_limit)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        return df
    
    def _clean_pool_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean pool data."""
        # Forward fill and backward fill
        df = df.fillna(method='ffill', limit=self.config.forward_fill_limit)
        df = df.fillna(method='bfill', limit=self.config.backward_fill_limit)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        return df
    
    def get_available_pools(self) -> List[str]:
        """Get list of available pools."""
        if not self.data_dir.exists():
            return []
        
        pools = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return pools
    
    def get_available_frequencies(self, pool_name: str) -> List[str]:
        """Get available frequencies for a specific pool."""
        pool_dir = self.data_dir / pool_name
        if not pool_dir.exists():
            return []
        
        price_files = list(pool_dir.glob("price_*.csv"))
        frequencies = []
        
        for file in price_files:
            # Extract frequency from filename (e.g., "price_1h.csv" -> "1h")
            freq = file.stem.replace("price_", "")
            frequencies.append(freq)
        
        return frequencies
