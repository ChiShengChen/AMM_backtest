"""
Data schema definitions using Pydantic for AMM backtester.
"""

from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

class DataSchema(BaseModel):
    """Base data schema class."""
    
    class Config:
        arbitrary_types_allowed = True

class PriceDataSchema(DataSchema):
    """Schema for price data validation."""
    
    timestamp: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if not isinstance(v, pd.DatetimeIndex):
            raise ValueError('timestamp must be pandas DatetimeIndex')
        if v.tz is not None:
            v = v.tz_localize(None)  # Remove timezone info
        return v
    
    @validator('open', 'high', 'low', 'close', 'volume')
    def validate_numeric_arrays(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('Price/volume data must be numpy arrays')
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError('Price/volume data must be numeric')
        return v
    
    @validator('high')
    def validate_high_low(cls, v, values):
        if 'low' in values and 'high' in values:
            if not np.all(values['high'] >= values['low']):
                raise ValueError('High must be >= low')
        return v
    
    @validator('close')
    def validate_close_range(cls, v, values):
        if 'open' in values and 'high' in values and 'low' in values:
            if not np.all((v >= values['low']) & (v <= values['high'])):
                raise ValueError('Close must be between low and high')
        return v

class PoolDataSchema(DataSchema):
    """Schema for pool data validation."""
    
    timestamp: pd.DatetimeIndex
    sqrtPriceX96: Optional[np.ndarray] = None
    tick: Optional[np.ndarray] = None
    liquidity: Optional[np.ndarray] = None
    volume_token0: Optional[np.ndarray] = None
    volume_token1: Optional[np.ndarray] = None
    fee_growth_global0X128: Optional[np.ndarray] = None
    fee_growth_global1X128: Optional[np.ndarray] = None
    fee_tier: Optional[np.ndarray] = None
    tvl_usd: Optional[np.ndarray] = None
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if not isinstance(v, pd.DatetimeIndex):
            raise ValueError('timestamp must be pandas DatetimeIndex')
        if v.tz is not None:
            v = v.tz_localize(None)
        return v
    
    @validator('sqrtPriceX96', 'tick', 'liquidity', 'volume_token0', 'volume_token1', 
               'fee_growth_global0X128', 'fee_growth_global1X128', 'fee_tier', 'tvl_usd')
    def validate_optional_numeric(cls, v):
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError('Pool data must be numpy arrays')
            if not np.issubdtype(v.dtype, np.number):
                raise ValueError('Pool data must be numeric')
        return v

class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    
    total_rows: int
    missing_values: dict
    outliers: dict
    data_quality_score: float
    warnings: List[str]
    errors: List[str]
    
    @validator('data_quality_score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Data quality score must be between 0 and 1')
        return v

class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    
    min_price: float = 0.000001
    max_price: float = 1000000.0
    min_volume: float = 0.0
    max_volume: float = 1e12
    max_price_change_pct: float = 100.0
    min_liquidity: float = 0.0
    max_liquidity: float = 1e15
    min_tick: int = -887272
    max_tick: int = 887272
    outlier_threshold: float = 5.0
    forward_fill_limit: int = 4
    backward_fill_limit: int = 4
    min_data_quality: float = 0.95
