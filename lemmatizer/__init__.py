"""Context-aware Russian lemmatizer using PyTorch."""
from .cli import cli
from .inference import LemmatizerInference

__all__ = ['cli', 'LemmatizerInference']
