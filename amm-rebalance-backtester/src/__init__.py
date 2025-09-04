"""
AMM Dynamic Rebalancing and Parameter Optimization Backtester

A comprehensive backtesting system for evaluating dynamic rebalancing strategies
vs. traditional fixed approaches in AMM environments.
"""

__version__ = "0.1.0"
__author__ = "Quant Team"

from .core.engine import BacktestEngine
from .core.metrics import MetricsCalculator
from .core.il_lvr import ILLVRCalculator
from .core.frictions import FrictionModel
from .strategies.base import BaseStrategy
from .io.loader import DataLoader
from .opt.search import OptunaOptimizer
from .reporting.plots import PlotGenerator
from .reporting.tables import TableGenerator

__all__ = [
    "BacktestEngine",
    "MetricsCalculator", 
    "ILLVRCalculator",
    "FrictionModel",
    "BaseStrategy",
    "DataLoader",
    "OptunaOptimizer",
    "PlotGenerator",
    "TableGenerator",
]
