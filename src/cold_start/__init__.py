"""Cold-start recommendation system package.

This package provides solutions for the cold-start problem in recommendation systems,
including content-based filtering, hybrid approaches, and advanced techniques for
handling new users and items with limited interaction data.
"""

__version__ = "1.0.0"
__author__ = "Recommendation Systems Team"

from .data import DataLoader, DataGenerator
from .models import (
    ContentBasedRecommender,
    HybridRecommender,
    ColdStartRecommender,
)
from .evaluation import Evaluator, Metrics
from .utils import set_seed, load_config

__all__ = [
    "DataLoader",
    "DataGenerator", 
    "ContentBasedRecommender",
    "HybridRecommender",
    "ColdStartRecommender",
    "Evaluator",
    "Metrics",
    "set_seed",
    "load_config",
]
