"""Multi-modal microbiome <-> metabolome imputation pipeline (MetaCardis).

Modular, object-oriented PyTorch pipeline with a clean separation between:
  * config      -- typed hyperparameter dataclasses
  * data        -- loading, stratified split, preprocessing, embeddings, loaders
  * models      -- Transformer-encoder imputers (CLS/GAP, configurable head)
  * evaluation  -- Scaling -> PCA -> Euclidean distance -> Mantel test
  * trainer     -- training loop, early stopping, metric-driven checkpointing
  * pipeline    -- end-to-end orchestration façade
"""

from .config import (
    Config,
    DataConfig,
    EmbeddingConfig,
    EvalConfig,
    ModelConfig,
    TrainConfig,
)
from .pipeline import ImputationPipeline

__all__ = [
    "Config",
    "DataConfig",
    "EmbeddingConfig",
    "EvalConfig",
    "ModelConfig",
    "TrainConfig",
    "ImputationPipeline",
]
