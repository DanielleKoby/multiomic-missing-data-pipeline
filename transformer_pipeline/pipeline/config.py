"""Configuration objects for the multi-modal imputation pipeline.

Everything that a user might reasonably want to tune lives here as a typed
dataclass field with an industrial-sensible default. Nothing in the rest of the
codebase hard-codes a hyperparameter; it always reads from these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# --------------------------------------------------------------------------- #
# Data / preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the data lives and how it is split / preprocessed."""

    # Input CSVs (index column = SampleID).
    data_dir: str = "train"
    microbiome_file: str = "microbiome.csv"
    metabolome_file: str = "metabolome.csv"
    metadata_file: str = "metadata.csv"

    # Metadata columns used to build the composite stratification label.
    # NOTE: the real files use ``CENTER_C`` and ``PATGROUPFINAL_C``.
    center_col: str = "CENTER_C"
    patgroup_col: str = "PATGROUPFINAL_C"

    # Train/validation split (applied ONLY to the complete samples).
    val_size: float = 0.20
    random_state: int = 42
    min_stratum_size: int = 2

    # K-fold cross-validation. Used by StratifiedKFoldSplitter (and the CV
    # notebook) to replace the single train/val split with ``n_folds`` rotations
    # so per-feature r / val_mse are reported as mean +/- std instead of one
    # noisy point estimate. The single-split StratifiedSplitter is unaffected and
    # ignores this; only the k-fold splitter reads it. val_size is then implied
    # by 1/n_folds (e.g. 5 folds -> ~20% val per fold, matching val_size=0.20).
    n_folds: int = 5

    # Metabolome preprocessing.
    metabolome_log: Literal["log1p", "log2"] = "log1p"
    standardize_metabolome: bool = True

    # Microbiome preprocessing (Stage 2+3).
    #   microbiome_clr=False -> microbiome stays as raw relative abundance
    #       (ORIGINAL): zeros mean "absent", but every value is tiny (~1/170) so
    #       the model is nearly blind and Model B's MSE on the simplex is flat.
    #   microbiome_clr=True  -> centered log-ratio (DEFAULT). Per row:
    #       clr(x)_i = ln(x_i + eps) - mean_j ln(x_j + eps), then per-feature
    #       z-score (fit on TRAIN). A microbe at the sample geometric mean -> ~0;
    #       an ABSENT microbe -> strongly negative (NOT zero). This is what gives
    #       the value channel real dynamic range and makes Model B's MSE / val_mse
    #       meaningful again. Model B's head switches softmax->identity and its
    #       output is inverse-CLR'd (softmax) back to relative abundance at
    #       imputation time.
    microbiome_clr: bool = True
    clr_pseudocount: float = 1e-6
    standardize_microbiome: bool = True


# --------------------------------------------------------------------------- #
# Node2vec embeddings
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingConfig:
    """Pre-computed node2vec embeddings for microbes and metabolites.

    The pipeline is *strict*: if a file is missing or a feature has no
    embedding, loading raises rather than fabricating data.
    """

    embed_dim: int = 128
    # Aligned, name-indexed node2vec tables produced by pipeline/extract_embeddings.py.
    # Row order matches the microbiome.csv / metabolome.csv feature columns 1:1.
    microbiome_embedding_file: Optional[str] = "embeddings/aligned/microbe_embeddings.csv"
    metabolome_embedding_file: Optional[str] = "embeddings/aligned/metabolite_embeddings.csv"
    # Per-metabolite status (calibrated | island | missing). Rows that are not
    # "calibrated" are random-init. The sidecar is still read for reporting, but
    # whether those rows are learned is controlled by ``unfreeze_random_metabolites``.
    metabolome_status_file: Optional[str] = "embeddings/aligned/metabolite_embedding_status.csv"

    # Embedding freezing policy.
    #   False (default) -> EVERY embedding row stays frozen, including the
    #                      random-init (island/missing) metabolites. The ablation
    #                      notebooks showed unfreezing them adds noise without
    #                      improving per-feature prediction, so all-frozen is the
    #                      default behaviour.
    #   True            -> the island/missing metabolite rows are UNFROZEN in
    #                      Model B (the old behaviour), driven by the status file.
    unfreeze_random_metabolites: bool = False


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Transformer-encoder architecture (shared by both models)."""

    # Token / model dimension. ``d_model`` defaults to ``embed_dim`` when None.
    d_model: Optional[int] = None
    n_heads: int = 8
    n_blocks: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    # Sequence -> vector reduction: learnable CLS token or global average pool.
    pooling: Literal["cls", "gap"] = "cls"

    # How a scalar feature value is fused with its node2vec embedding to form a
    # token.
    #   "scale"  -> token = value * embedding (DEFAULT). Won the CLR grid for
    #               BOTH models (A peak_r ~0.17, B ~0.16). With CLR inputs the
    #               value channel has real dynamic range, so the feared
    #               mean-collapse does not happen and this simple fusion wins.
    #   "add"    -> token = embedding + slope * value. Pure additive: feature
    #               identity (embedding) always present, value scales a SEPARATE
    #               learnable per-dim direction ``slope`` added on top. In the
    #               CLR grid this failed to fit (train_mse stuck ~1.0); kept for
    #               the record.
    #   "affine" -> token = embedding * (1 + slope * value). Keeps the value
    #               entangled with the embedding direction; trailed scale in the
    #               grid, so it is kept only for the record.
    tokenizer: Literal["scale", "affine", "add"] = "scale"

    # MLP regression head.
    head_hidden_dim: int = 256
    head_dropout: float = 0.1

    # Output activation of the head.
    #   Model A (-> standardized metabolome): "identity"
    #   Model B (-> relative abundance):       "softmax" (simplex)
    output_activation: Literal["identity", "softmax", "softplus", "relu"] = "identity"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation / training-loop settings."""

    batch_size: int = 64
    max_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = 1.0

    # Early stopping + checkpoint selection.
    # "val_mse"        -> minimise validation MSE (DEFAULT; honest, cheap)
    # "per_feature_r"  -> maximise mean per-feature Pearson r on val
    # "mantel"         -> maximise the Mantel statistic. NOTE: the ridge probe
    #                     showed Mantel does NOT track imputation quality (a
    #                     strictly better model can score lower), so it is no
    #                     longer the default. It is still computed and printed
    #                     each epoch as a secondary report.
    selection_metric: Literal["mantel", "val_mse", "per_feature_r"] = "val_mse"
    patience: int = 20
    min_epochs: int = 1

    # "mps" (Apple Silicon) -> "cuda" -> "cpu" auto-detection when "auto".
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    num_workers: int = 0
    seed: int = 42

    checkpoint_dir: str = "checkpoints"


# --------------------------------------------------------------------------- #
# Evaluation (Mantel test on PCs)
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    """Scaling -> PCA -> distance -> Mantel evaluation settings."""

    n_components: int = 2
    distance_metric: str = "euclidean"
    # Scaler + PCA are fit INDEPENDENTLY on each dataset (predicted vs truth).
    mantel_method: Literal["spearman", "pearson"] = "spearman"
    mantel_permutations: int = 999


# --------------------------------------------------------------------------- #
# Top-level container
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model_a: ModelConfig = field(
        default_factory=lambda: ModelConfig(output_activation="identity")
    )
    model_b: ModelConfig = field(
        default_factory=lambda: ModelConfig(output_activation="softmax")
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Final inference stage: impute the missing modality for the 348+348
    # structurally-missing samples and write CSVs. Default OFF.
    run_imputation: bool = False
    output_dir: str = "outputs"
