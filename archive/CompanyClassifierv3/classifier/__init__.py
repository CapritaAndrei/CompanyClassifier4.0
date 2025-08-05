"""
Insurance Classification System - Core Classifier Module
"""

from .weighted_classifier import WeightedInsuranceClassifier
from .embeddings import EmbeddingManager
from .similarity import SimilarityCalculator

__all__ = [
    'WeightedInsuranceClassifier',
    'EmbeddingManager', 
    'SimilarityCalculator'
] 