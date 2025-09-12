"""
Binance data fetching module for CLMM backtesting.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Symbol mapping for common pairs
SYMBOL_MAPPING = {
    "USDCUSDT": "USDCUSDT",
    "ETHUSDC": "ETHUSDC", 
    "BTCUSDC": "BTCUSDC",
    "ETHUSDT": "ETHUSDT",
    "BTCUSDT": "BTCUSDT",
}

# Interval mapping
INTERVAL_MAPPING = {
    "1h": "1h",
    "1d": "1d",
    "4h": "4h",
    "15m": "15m",
    "1m": "1m",
}

class BinanceDataFetcher:
    """Fetches historical kline data from Binance REST API."""
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SteerIntentBacktester/1.0"
        })
    
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch kline data from Binance REST API.
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDC')
            interval: Time interval ('1h', '1d', etc.)
            start: Start datetime
            end: End datetime  
            limit: Maximum number of records per request (max 1000)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
        """
        if symbol not in SYMBOL_MAPPING:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        if interval not in INTERVAL_MAPPING:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert to milliseconds for API
        start_ts = int(start.timestamp() * 1000) if start else None
        end_ts = int(end.timestamp() * 1000) if end else None
        
        all_klines = []
        current_start = start_ts
        
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            if current_start:
                params["startTime"] = current_start
            
            if end_ts:
                params["endTime"] = end_ts
            
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v3/klines",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Check if we need to paginate
                if len(klines) < limit or (end_ts and klines[-1][6] >= end_ts):
                    break
                
                # Update start time for next request
                current_start = klines[-1][6] + 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data from Binance: {e}")
                raise
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs."""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/exchangeInfo", timeout=30)
            response.raise_for_status()
            data = response.json()
            return [symbol["symbol"] for symbol in data["symbols"] if symbol["status"] == "TRADING"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching available symbols: {e}")
            return list(SYMBOL_MAPPING.values())
    
    def get_server_time(self) -> datetime:
        """Get Binance server time."""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/time", timeout=30)
            response.raise_for_status()
            data = response.json()
            return datetime.fromtimestamp(data["serverTime"] / 1000)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching server time: {e}")
            return datetime.utcnow()


class BinanceVisionDownloader:
    """Downloads historical data from Binance Vision S3."""
    
    def __init__(self, base_url: str = "https://data.binance.vision"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def download_daily_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Download daily data from Binance Vision S3.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        # Implementation for S3 data download
        # This would involve downloading and stitching daily ZIP files
        # For now, return empty DataFrame
        logger.warning("Binance Vision S3 downloader not yet implemented")
        return pd.DataFrame()


def fetch_klines(
    symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Convenience function to fetch kline data.
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        start: Start datetime
        end: End datetime
        limit: Maximum records per request
        
    Returns:
        DataFrame with kline data
    """
    fetcher = BinanceDataFetcher()
    return fetcher.fetch_klines(symbol, interval, start, end, limit)
