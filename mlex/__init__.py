"""
Money Laundering Expert System (MLEX)

A comprehensive machine learning framework for financial fraud detection and money laundering prevention.
"""

__version__ = "0.0.2"
__author__ = "Diego Pinheiro"
__email__ = "diegompin@gmail.com"

from .evaluation import *
from .features import *
from .models import *
from .utils import *

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Models
    "GRU",
    "LSTM", 
    "RNN",
    
    # Evaluation
    "StandardEvaluator",
    "F1MaxThresholdStrategy",
    "QuantileThresholdStrategy",
    "EvaluationPlotter",
    
    # Utils
    "DataReader",
    "FeatureStratifiedSplit",
    "PreProcessingTransformer",
    "NoiseInjector",
    
    # Features
    "SequenceDataset",
    "SequenceTransformer",
]
