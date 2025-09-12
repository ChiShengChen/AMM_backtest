"""
Strategy implementations for CLMM backtesting.
"""

from .base import BaseStrategy
from .classic import ClassicStrategy
from .channel_multiplier import ChannelMultiplierStrategy
from .bollinger import BollingerStrategy
from .keltner import KeltnerStrategy
from .donchian import DonchianStrategy
from .stable import StableStrategy
from .fluid import FluidStrategy
from .imperfect_classic import ImperfectClassicStrategy
from .ml_strategy import MLBollingerStrategy, MLKeltnerStrategy, MLDonchianStrategy, MLHybridStrategy

# Quantum strategies (conditional import)
try:
    from .quantum_strategy import QuantumBollingerStrategy, QuantumKeltnerStrategy, QuantumHybridStrategy
    QUANTUM_STRATEGIES_AVAILABLE = True
except ImportError:
    QUANTUM_STRATEGIES_AVAILABLE = False

__all__ = [
    "BaseStrategy",
    "ClassicStrategy",
    "ChannelMultiplierStrategy",
    "BollingerStrategy",
    "KeltnerStrategy",
    "DonchianStrategy",
    "StableStrategy",
    "FluidStrategy",
    "ImperfectClassicStrategy",
    "MLBollingerStrategy",
    "MLKeltnerStrategy",
    "MLDonchianStrategy",
    "MLHybridStrategy",
]

# Add quantum strategies to __all__ if available
if QUANTUM_STRATEGIES_AVAILABLE:
    __all__.extend([
        "QuantumBollingerStrategy",
        "QuantumKeltnerStrategy",
        "QuantumHybridStrategy",
    ])
