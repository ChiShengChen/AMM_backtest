"""
Feature engineering for Steer Intent ML strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

class SteerFeatureEngineer:
    """Feature engineering for Steer Intent ML strategies."""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.scaler = RobustScaler()
        self.is_fitted = False
        
    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for Steer Intent ML models.
        
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
        
        # Intent-specific features
        df = self._add_intent_features(df)
        
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
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
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
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        df['macd_bullish'] = (macd_line > signal_line).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
        
        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14, 3)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
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
        
        # Volatility clustering
        df['vol_clustering'] = (df['volatility_20'] > df['volatility_20'].shift(1)).astype(int)
        
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
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = (df['obv'] > df['obv_sma']).astype(int)
        
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
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session indicators
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
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
        
        # Price gaps
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def _add_intent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intent-specific features for Steer strategies."""
        # Trend strength
        df['trend_strength'] = abs(df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # Mean reversion signals
        df['mean_reversion_signal'] = (df['bb_position'] < 0.2) | (df['bb_position'] > 0.8)
        df['mean_reversion_signal'] = df['mean_reversion_signal'].astype(int)
        
        # Momentum signals
        df['momentum_signal'] = ((df['macd'] > df['macd_signal']) & 
                                (df['rsi'] > 50) & 
                                (df['stoch_k'] > df['stoch_d'])).astype(int)
        
        # Volatility breakout signals
        df['vol_breakout_signal'] = (df['bb_width'] > df['bb_width'].rolling(20).quantile(0.8)).astype(int)
        
        # Support/Resistance levels
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Intent confidence (combination of signals)
        df['intent_confidence'] = (
            df['momentum_signal'] * 0.3 +
            df['mean_reversion_signal'] * 0.3 +
            df['vol_breakout_signal'] * 0.2 +
            (df['rsi_oversold'] | df['rsi_overbought']).astype(int) * 0.2
        )
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for temporal patterns."""
        lag_features = ['returns', 'volume', 'volatility_20', 'rsi', 'bb_position', 'intent_confidence']
        
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
        
        # Intent-confidence interactions
        if 'intent_confidence' in df.columns and 'volatility_20' in df.columns:
            df['intent_vol_interaction'] = df['intent_confidence'] * df['volatility_20']
        
        return df
    
    def create_targets(self, df: pd.DataFrame, strategy_type: str = 'bollinger') -> pd.DataFrame:
        """
        Create target variables for different Steer strategies.
        
        Args:
            df: DataFrame with features
            strategy_type: Type of strategy to create targets for
            
        Returns:
            DataFrame with target variables
        """
        if strategy_type == 'bollinger':
            return self._create_bollinger_targets(df)
        elif strategy_type == 'keltner':
            return self._create_keltner_targets(df)
        elif strategy_type == 'donchian':
            return self._create_donchian_targets(df)
        else:
            return self._create_generic_targets(df)
    
    def _create_bollinger_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets for Bollinger Bands strategy."""
        # Future returns
        df['future_return_1'] = df['returns'].shift(-1)
        df['future_return_5'] = df['returns'].rolling(5).sum().shift(-5)
        
        # Bollinger Band signals
        df['bb_signal'] = 0
        df.loc[df['bb_position'] < 0.2, 'bb_signal'] = 1  # Buy signal (oversold)
        df.loc[df['bb_position'] > 0.8, 'bb_signal'] = -1  # Sell signal (overbought)
        
        # Optimal rebalancing decision
        df['should_rebalance'] = (abs(df['bb_position'] - 0.5) > 0.3).astype(int)
        
        return df
    
    def _create_keltner_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets for Keltner Channels strategy."""
        # Future returns
        df['future_return_1'] = df['returns'].shift(-1)
        df['future_return_5'] = df['returns'].rolling(5).sum().shift(-5)
        
        # Keltner Channel signals (using ATR-based channels)
        keltner_upper = df['ema_20'] + 2 * df['atr']
        keltner_lower = df['ema_20'] - 2 * df['atr']
        keltner_position = (df['close'] - keltner_lower) / (keltner_upper - keltner_lower)
        
        df['keltner_signal'] = 0
        df.loc[keltner_position < 0.2, 'keltner_signal'] = 1
        df.loc[keltner_position > 0.8, 'keltner_signal'] = -1
        
        df['should_rebalance'] = (abs(keltner_position - 0.5) > 0.3).astype(int)
        
        return df
    
    def _create_donchian_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets for Donchian Channels strategy."""
        # Future returns
        df['future_return_1'] = df['returns'].shift(-1)
        df['future_return_5'] = df['returns'].rolling(5).sum().shift(-5)
        
        # Donchian Channel signals
        donchian_high = df['high'].rolling(20).max()
        donchian_low = df['low'].rolling(20).min()
        donchian_position = (df['close'] - donchian_low) / (donchian_high - donchian_low)
        
        df['donchian_signal'] = 0
        df.loc[donchian_position < 0.2, 'donchian_signal'] = 1
        df.loc[donchian_position > 0.8, 'donchian_signal'] = -1
        
        df['should_rebalance'] = (abs(donchian_position - 0.5) > 0.3).astype(int)
        
        return df
    
    def _create_generic_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create generic targets for any strategy."""
        # Future returns
        df['future_return_1'] = df['returns'].shift(-1)
        df['future_return_5'] = df['returns'].rolling(5).sum().shift(-5)
        df['future_return_20'] = df['returns'].rolling(20).sum().shift(-20)
        
        # Generic rebalancing decision
        price_deviation = abs(df['close'] / df['close'].rolling(20).mean() - 1)
        df['should_rebalance'] = (price_deviation > 0.02).astype(int)
        
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
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
