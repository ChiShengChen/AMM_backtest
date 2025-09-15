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

# Strategy registry
STRATEGY_REGISTRY = {
    'classic': ClassicStrategy,
    'channel_multiplier': ChannelMultiplierStrategy,
    'bollinger': BollingerStrategy,
    'keltner': KeltnerStrategy,
    'donchian': DonchianStrategy,
    'stable': StableStrategy,
    'fluid': FluidStrategy,
    'imperfect_classic': ImperfectClassicStrategy,
    'ml_bollinger': MLBollingerStrategy,
    'ml_keltner': MLKeltnerStrategy,
    'ml_donchian': MLDonchianStrategy,
    'ml_hybrid': MLHybridStrategy,
}

# Add quantum strategies to registry if available
if QUANTUM_STRATEGIES_AVAILABLE:
    STRATEGY_REGISTRY.update({
        'quantum_bollinger': QuantumBollingerStrategy,
        'quantum_keltner': QuantumKeltnerStrategy,
        'quantum_hybrid': QuantumHybridStrategy,
    })

def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name]
