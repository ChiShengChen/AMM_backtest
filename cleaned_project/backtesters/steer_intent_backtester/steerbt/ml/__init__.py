"""
Machine Learning modules for Steer Intent Backtester.
"""

from .feature_engineering import SteerFeatureEngineer
from .models import SteerMLStrategy, IntentPredictor, PricePredictor
from .training import SteerMLTrainer

# Quantum models (conditional import)
try:
    from .quantum_models import (
        SteerQuantumModelBase,
        SteerQNNIntentPredictor,
        SteerQSVMPricePredictor,
        SteerPennyLaneQNNPredictor,
        create_steer_quantum_model,
        save_steer_quantum_model,
        load_steer_quantum_model,
        QUANTUM_AVAILABLE,
    )
    QUANTUM_IMPORTED = True
except ImportError:
    QUANTUM_IMPORTED = False
    QUANTUM_AVAILABLE = False

__all__ = [
    'SteerFeatureEngineer',
    'SteerMLStrategy',
    'IntentPredictor', 
    'PricePredictor',
    'SteerMLTrainer'
]

# Add quantum models to __all__ if available
if QUANTUM_IMPORTED:
    __all__.extend([
        'SteerQuantumModelBase',
        'SteerQNNIntentPredictor',
        'SteerQSVMPricePredictor',
        'SteerPennyLaneQNNPredictor',
        'create_steer_quantum_model',
        'save_steer_quantum_model',
        'load_steer_quantum_model',
        'QUANTUM_AVAILABLE',
    ])
