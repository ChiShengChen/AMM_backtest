"""
Machine Learning modules for AMM backtesting.
"""

from .feature_engineering import FeatureEngineer
from .models import MLStrategy, RebalancePredictor, VolatilityPredictor
from .training import MLTrainer

# Quantum models (conditional import)
try:
    from .quantum_models import (
        QuantumModelBase,
        QNNRebalancePredictor,
        QSVMVolatilityPredictor,
        PennyLaneQNNPredictor,
        create_quantum_model,
        save_quantum_model,
        load_quantum_model,
        QUANTUM_AVAILABLE,
    )
    QUANTUM_IMPORTED = True
except ImportError:
    QUANTUM_IMPORTED = False
    QUANTUM_AVAILABLE = False

__all__ = [
    'FeatureEngineer',
    'MLStrategy', 
    'RebalancePredictor',
    'VolatilityPredictor',
    'MLTrainer'
]

# Add quantum models to __all__ if available
if QUANTUM_IMPORTED:
    __all__.extend([
        'QuantumModelBase',
        'QNNRebalancePredictor',
        'QSVMVolatilityPredictor',
        'PennyLaneQNNPredictor',
        'create_quantum_model',
        'save_quantum_model',
        'load_quantum_model',
        'QUANTUM_AVAILABLE',
    ])
