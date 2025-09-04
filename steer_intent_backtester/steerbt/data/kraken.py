"""
Kraken data fetching module for CLMM backtesting (fallback data source).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Symbol mapping for Kraken
SYMBOL_MAPPING = {
    "USDCUSDT": "USDCUSDT",
    "ETHUSDC": "ETH/USDC",
    "BTCUSDC": "XBT/USDC",
    "ETHUSDT": "ETH/USDT", 
    "BTCUSDT": "XBT/USDT",
}

# Interval mapping (Kraken uses minutes)
INTERVAL_MAPPING = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

class KrakenDataFetcher:
    """Fetches historical OHLC data from Kraken REST API."""
    
    def __init__(self, base_url: str = "https://api.kraken.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SteerIntentBacktester/1.0"
        })
    
    def fetch_ohlc(
        self,
        pair: str,
        interval: str,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from Kraken REST API.
        
        Args:
            pair: Trading pair (e.g., 'ETH/USDC')
            interval: Time interval ('1h', '1d', etc.)
            since: Start datetime (Kraken limitation: only since parameter)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, count
        """
        if pair not in SYMBOL_MAPPING.values():
            raise ValueError(f"Unsupported pair: {pair}")
        
        if interval not in INTERVAL_MAPPING:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert interval to minutes
        interval_minutes = INTERVAL_MAPPING[interval]
        
        # Convert since to Unix timestamp
        since_ts = int(since.timestamp()) if since else None
        
        params = {
            "pair": pair,
            "interval": interval_minutes
        }
        
        if since_ts:
            params["since"] = since_ts
        
        try:
            response = self.session.get(
                f"{self.base_url}/0/public/OHLC",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data["error"]:
                raise ValueError(f"Kraken API error: {data['error']}")
            
            # Extract OHLC data
            pair_key = list(data["result"].keys())[0]  # Kraken returns data under pair key
            ohlc_data = data["result"][pair_key]
            
            if not ohlc_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "count"
            ])
            
            # Convert types
            numeric_columns = ["open", "high", "low", "close", "volume", "count"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Kraken: {e}")
            raise
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs."""
        try:
            response = self.session.get(f"{self.base_url}/0/public/AssetPairs", timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data["error"]:
                logger.error(f"Kraken API error: {data['error']}")
                return list(SYMBOL_MAPPING.values())
            
            return list(data["result"].keys())
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching available pairs: {e}")
            return list(SYMBOL_MAPPING.values())
    
    def get_server_time(self) -> datetime:
        """Get Kraken server time."""
        try:
            response = self.session.get(f"{self.base_url}/0/public/Time", timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data["error"]:
                logger.error(f"Kraken API error: {data['error']}")
                return datetime.utcnow()
            
            return datetime.fromtimestamp(data["result"]["unixtime"])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching server time: {e}")
            return datetime.utcnow()


def fetch_ohlc(
    pair: str,
    interval: str,
    since: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch OHLC data from Kraken.
    
    Args:
        pair: Trading pair
        interval: Time interval
        since: Start datetime
        
    Returns:
        DataFrame with OHLC data
    """
    fetcher = KrakenDataFetcher()
    return fetcher.fetch_ohlc(pair, interval, since)
