
"""
FPL Tool - Phase 2 Optimization & Tools Package
Production-ready Fantasy Premier League optimization and recommendation system.
"""

__version__ = "2.0.0"
__author__ = "FPL Data Science Team"

from .optimizer import FPLOptimizer
from .recommender import FPLRecommender
from .validator import FPLValidator

__all__ = ["FPLOptimizer", "FPLRecommender", "FPLValidator"]
