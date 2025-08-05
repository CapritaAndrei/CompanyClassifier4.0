"""
Classification module for company classification.
Handles different classification strategies and decision making.
"""

from .similarity import SimilarityClassifier
from .thresholds import TopKSelector

__all__ = ['SimilarityClassifier', 'TopKSelector'] 