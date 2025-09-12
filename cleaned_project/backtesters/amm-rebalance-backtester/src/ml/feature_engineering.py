"""
Feature engineering for machine learning strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for ML-based AMM strategies."""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.is_fitted = False
        
    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for ML models.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = price_data.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Lagged features
        df = self._add_lagged_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        logger.info(f"Created {len(df.columns)} features for {len(df)} observations")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price levels
        df['price_level'] = df['close']
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        
        # High-Low features
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['hc_ratio'] = df['high'] / df['close']
        df['lc_ratio'] = df['low'] / df['close']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Rolling volatility
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_{period}_annualized'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        # GARCH-like volatility (EWMA)
        df['volatility_ewma'] = df['returns'].ewm(span=20).std()
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(10).std()
        
        # Volatility percentiles
        df['vol_percentile'] = df['volatility_20'].rolling(100).rank(pct=True)
        
        # Volatility regime
        df['vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume volatility
        df['volume_volatility'] = df['volume'].rolling(20).std()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if df.index.dtype == 'object':
            df.index = pd.to_datetime(df.index)
        
        # Time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price impact proxy
        df['price_impact'] = abs(df['returns']) / np.log(1 + df['volume'])
        
        # Order flow imbalance proxy
        df['order_flow_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Tick-by-tick volatility
        df['tick_volatility'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for temporal patterns."""
        lag_features = ['returns', 'volume', 'volatility_20', 'rsi', 'bb_position']
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different indicators."""
        # Volatility-volume interactions
        if 'volatility_20' in df.columns and 'volume_ratio_20' in df.columns:
            df['vol_volume_interaction'] = df['volatility_20'] * df['volume_ratio_20']
        
        # Price-momentum interactions
        if 'returns' in df.columns and 'rsi' in df.columns:
            df['momentum_rsi_interaction'] = df['returns'] * (df['rsi'] - 50) / 50
        
        # Volatility-regime interactions
        if 'volatility_20' in df.columns and 'vol_regime' in df.columns:
            df['vol_regime_interaction'] = df['volatility_20'] * df['vol_regime']
        
        return df
    
    def create_targets(self, df: pd.DataFrame, rebalance_threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target variables for ML models.
        
        Args:
            df: DataFrame with features
            rebalance_threshold: Threshold for rebalancing decision
            
        Returns:
            DataFrame with target variables
        """
        # Future returns (for prediction)
        df['future_return_1'] = df['returns'].shift(-1)
        df['future_return_5'] = df['returns'].rolling(5).sum().shift(-5)
        df['future_return_20'] = df['returns'].rolling(20).sum().shift(-20)
        
        # Rebalancing decision (binary)
        price_deviation = abs(df['close'] / df['close'].rolling(20).mean() - 1)
        df['should_rebalance'] = (price_deviation > rebalance_threshold).astype(int)
        
        # Volatility prediction target
        df['future_volatility'] = df['returns'].rolling(20).std().shift(-20)
        
        # Drawdown prediction
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        df['future_max_drawdown'] = drawdown.rolling(20).min().shift(-20)
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        self.scaler.fit(df[feature_cols])
        self.is_fitted = True
        logger.info(f"Fitted scaler on {len(feature_cols)} features")
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming features")
        
        df_transformed = df.copy()
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        df_transformed[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df_transformed
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr
