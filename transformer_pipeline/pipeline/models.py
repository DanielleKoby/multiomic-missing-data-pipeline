"""Model architecture: a Transformer-encoder imputer.

Both Model A (microbiome -> standardized metabolome) and Model B
(standardized metabolome -> relative-abundance microbiome) are instances of the
same :class:`ImputationTransformer`; they differ only in the input embedding
matrix, the target dimension and the output activation.

Forward path
------------
    scalar feature vector  (B, N)
        |  x  node2vec embedding  (N, E)            <- ScalarEmbeddingTokenizer
    token sequence         (B, N, E)
        |  optional linear proj E -> d_model
        |  + optional learnable CLS token
    Transformer encoder    (B, N(+1), d_model)      <- self-attn/residual/LN
        |  pool: CLS token OR global average
    pooled vector          (B, d_model)
        |  MLP head + output activation
    predicted targets      (B, target_dim)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# Tokeniser: scalar value x node2vec embedding
# --------------------------------------------------------------------------- #
class ScalarEmbeddingTokenizer(nn.Module):
    """Turns a (B, N) feature vector into a (B, N, E) token sequence.

    The node2vec matrix is split into two parts:

      * ``embedding_base``  -- a non-trainable buffer holding the original
        vectors. It travels with the model (``.to(device)``, checkpoints) and is
        NEVER changed by gradients or weight decay.
      * ``embedding_delta`` -- a learnable parameter, zero-initialised, that is
        multiplied by ``trainable_mask`` (1 for rows we want to learn, 0 for
        frozen rows). Only masked rows can move, and because the delta starts at
        zero, frozen rows stay bit-for-bit equal to ``embedding_base``.

    This is how the random-init metabolite rows (island + missing) become
    trainable while the calibrated node2vec rows remain frozen. With
    ``trainable_mask=None`` every row is frozen -> behaviour is identical to the
    original fixed-lookup tokenizer.

    Tokenization modes (``mode``)
    -----------------------------
      * ``"scale"``  -- token = value * embedding (the ORIGINAL behaviour). A
        feature sitting at its mean (~0 after standardization) produces a ~zero
        token, so the encoder cannot see it -> the model collapses to the column
        mean. Kept for A/B comparison only.
      * ``"add"``    -- token = embedding + slope * value (DEFAULT). Pure
        additive: the embedding (feature identity) is always present, and the
        scalar value scales a SEPARATE learnable per-dimension direction
        ``value_slope`` that is added on top. The value signal is fully
        separable from the embedding pattern.
      * ``"affine"`` -- token = embedding * (1 + slope * value). Keeps the value
        entangled with the embedding direction; in practice it suppressed the
        value signal, so it is kept only for the record.
    """

    def __init__(self, embedding_matrix: np.ndarray, trainable_mask=None,
                 mode: str = "add"):
        super().__init__()
        emb = torch.tensor(np.asarray(embedding_matrix), dtype=torch.float32)
        self.n_features, self.embed_dim = emb.shape
        if mode not in ("scale", "affine", "add"):
            raise ValueError(f"unknown tokenizer mode: {mode}")
        self.mode = mode

        self.register_buffer("embedding_base", emb)  # (N, E) constant
        if trainable_mask is None:
            mask = torch.zeros(self.n_features, 1)
        else:
            m = torch.as_tensor(np.asarray(trainable_mask), dtype=torch.float32)
            mask = m.reshape(self.n_features, 1)
        self.register_buffer("trainable_mask", mask)  # (N, 1) 1=learn 0=frozen
        self.n_trainable = int(mask.sum().item())

        # Zero-init so the effective table starts exactly at embedding_base.
        self.embedding_delta = nn.Parameter(torch.zeros_like(emb))

        # Learnable per-dimension slope for the value->token interaction (used
        # by "add" and "affine"). Init at 1.0.
        self.value_slope = nn.Parameter(torch.ones(self.embed_dim))

    @property
    def embedding(self) -> torch.Tensor:
        """Effective table = frozen base + learned delta on trainable rows only."""
        return self.embedding_base + self.embedding_delta * self.trainable_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding.unsqueeze(0)     # (1, N, E)
        v = x.unsqueeze(-1)                    # (B, N, 1)
        slope = self.value_slope.view(1, 1, -1)   # (1, 1, E)
        if self.mode == "scale":
            # ORIGINAL: (B, N, 1) * (1, N, E) -> (B, N, E)
            return v * emb
        if self.mode == "add":
            # ADDITIVE: embedding + value * slope. The value scales a separate
            # learnable direction, added on top of the always-present embedding.
            return emb + slope * v
        # AFFINE: embedding * (1 + slope * value). Value entangled with the
        # embedding direction; kept for the record.
        return emb + (slope * v) * emb


# --------------------------------------------------------------------------- #
# Transformer encoder imputer
# --------------------------------------------------------------------------- #
class ImputationTransformer(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        target_dim: int,
        cfg: ModelConfig,
        trainable_mask=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ScalarEmbeddingTokenizer(
            embedding_matrix, trainable_mask, mode=cfg.tokenizer
        )

        embed_dim = self.tokenizer.embed_dim
        d_model = cfg.d_model or embed_dim
        self.d_model = d_model

        # Project token dim -> model dim (identity when they already match).
        self.input_proj = (
            nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        )

        # Learnable CLS token (only used when pooling == "cls").
        if cfg.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({cfg.n_heads})."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,  # pre-LN: LayerNorm -> attn -> residual (clean residual stream)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_blocks)
        self.input_norm = nn.LayerNorm(d_model)

        # MLP regression head.
        self.head = nn.Sequential(
            nn.Linear(d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, target_dim),
        )
        self.output_activation = cfg.output_activation

    # ----- pooling helpers -------------------------------------------------- #
    def _pool(self, seq: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return seq[:, 0]            # the CLS position
        return seq.mean(dim=1)          # global average pooling

    def _apply_output_activation(self, y: torch.Tensor) -> torch.Tensor:
        act = self.output_activation
        if act == "identity":
            return y
        if act == "softmax":           # compositional simplex (sums to 1)
            return torch.softmax(y, dim=-1)
        if act == "softplus":
            return torch.nn.functional.softplus(y)
        if act == "relu":
            return torch.relu(y)
        raise ValueError(f"unknown output_activation: {act}")

    # ----- forward ---------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)            # (B, N, E)
        tokens = self.input_proj(tokens)      # (B, N, d_model)

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, d_model)

        tokens = self.input_norm(tokens)
        encoded = self.encoder(tokens)        # (B, N(+1), d_model)
        pooled = self._pool(encoded)          # (B, d_model)
        out = self.head(pooled)               # (B, target_dim)
        return self._apply_output_activation(out)


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def build_model_a(micro_embeddings: np.ndarray, n_metabolites: int, cfg: ModelConfig):
    """Microbiome -> standardized metabolome (identity output)."""
    return ImputationTransformer(micro_embeddings, n_metabolites, cfg)


def build_model_b(
    metab_embeddings: np.ndarray,
    n_microbes: int,
    cfg: ModelConfig,
    trainable_mask=None,
):
    """Standardized metabolome -> relative-abundance microbiome (softmax).

    ``trainable_mask`` (length = n_metabolites, 1 where the embedding row should
    be learned) unfreezes the random-init metabolites while keeping the
    calibrated node2vec rows fixed.
    """
    return ImputationTransformer(
        metab_embeddings, n_microbes, cfg, trainable_mask=trainable_mask
    )
