"""
Data fetching modules for CLMM backtesting.
"""

from .binance import BinanceDataFetcher, fetch_klines
from .kraken import KrakenDataFetcher, fetch_ohlc

__all__ = [
    "BinanceDataFetcher",
    "fetch_klines", 
    "KrakenDataFetcher",
    "fetch_ohlc"
]
