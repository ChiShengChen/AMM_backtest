"""
Strategy implementations for AMM backtester.
"""

from .base import BaseStrategy
from .baseline_static import BaselineStaticStrategy
from .baseline_fixed import BaselineFixedStrategy
from .dyn_vol import DynamicVolatilityStrategy
from .dyn_inventory import DynamicInventoryStrategy
from .ml_strategy import MLBasedStrategy, MLVolatilityStrategy, MLHybridStrategy

# Quantum strategies (conditional import)
try:
    from .quantum_strategy import QuantumBasedStrategy, QuantumVolatilityStrategy, QuantumHybridStrategy
    QUANTUM_STRATEGIES_AVAILABLE = True
except ImportError:
    QUANTUM_STRATEGIES_AVAILABLE = False

__all__ = [
    "BaseStrategy",
    "BaselineStaticStrategy",
    "BaselineFixedStrategy", 
    "DynamicVolatilityStrategy",
    "DynamicInventoryStrategy",
    "MLBasedStrategy",
    "MLVolatilityStrategy", 
    "MLHybridStrategy",
]

# Add quantum strategies to __all__ if available
if QUANTUM_STRATEGIES_AVAILABLE:
    __all__.extend([
        "QuantumBasedStrategy",
        "QuantumVolatilityStrategy",
        "QuantumHybridStrategy",
    ])
