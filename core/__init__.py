"""
Core active learning module
"""

from .active_learner import ActiveLearner
from .utils.model import EmbeddingClassifier

__all__ = ["EmbeddingClassifier", "ActiveLearner"]
