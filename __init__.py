# HNSCC Pipeline Refactor
# A modular image processing pipeline for HNSCC spatial analysis

from .pipeline import Pipeline
from .config.config import Config

__version__ = "0.1.0"
__all__ = ["Pipeline", "Config"]
