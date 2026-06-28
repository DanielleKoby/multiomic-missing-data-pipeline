"""Separate Mantel-test evaluation for Model A and Model B.

For each model we build two combined (microbiome | metabolome) datasets on the
validation samples:

    Model A:  Predicted    = [true microbiome | IMPUTED metabolome]
              GroundTruth  = [true microbiome | true   metabolome]

    Model B:  Predicted    = [IMPUTED microbiome | true metabolome]
              GroundTruth  = [true    microbiome | true metabolome]

Each dataset is independently StandardScaler-normalised, PCA-reduced to the
first ``n_components`` PCs, and turned into a Euclidean distance matrix. The two
distance matrices are then compared with a Mantel test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import EvalConfig


def per_feature_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean per-output-column Pearson r between truth and prediction.

    This is the honest "did we predict individual features" metric. Columns with
    no variance in either truth or prediction are skipped (their correlation is
    undefined). Returns NaN if no column is usable.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    rs = []
    for j in range(true.shape[1]):
        t, p = true[:, j], pred[:, j]
        if np.std(t) > 1e-12 and np.std(p) > 1e-12:
            rs.append(np.corrcoef(t, p)[0, 1])
    return float(np.nanmean(rs)) if rs else float("nan")


@dataclass
class MantelResult:
    statistic: float
    p_value: float
    n_permutations: int
    method: str

    def __str__(self) -> str:
        return (
            f"Mantel({self.method}) r={self.statistic:+.4f} "
            f"p={self.p_value:.4f} (perm={self.n_permutations})"
        )


# --------------------------------------------------------------------------- #
# Scaling -> PCA -> distance matrix
# --------------------------------------------------------------------------- #
def embed_and_distance(data: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    """StandardScaler -> PCA(n_components) -> pairwise distance matrix.

    Scaler and PCA are fit on *this* dataset only (independent per dataset),
    so the returned distance matrix reflects the internal geometry of ``data``.
    """
    data = np.asarray(data, dtype=np.float64)
    scaled = StandardScaler().fit_transform(data)
    n_comp = min(cfg.n_components, scaled.shape[1], scaled.shape[0])
    pcs = PCA(n_components=n_comp, random_state=0).fit_transform(scaled)
    return squareform(pdist(pcs, metric=cfg.distance_metric))


# --------------------------------------------------------------------------- #
# Mantel test
# --------------------------------------------------------------------------- #
def mantel_test(dm1: np.ndarray, dm2: np.ndarray, cfg: EvalConfig) -> MantelResult:
    """Mantel test between two distance matrices.

    Uses ``skbio.stats.distance.mantel`` when available; otherwise falls back
    to an equivalent permutation implementation (Pearson/Spearman on the
    condensed upper triangles). Raises rather than silently diverging if the
    inputs are malformed.
    """
    try:
        from skbio.stats.distance import mantel as skbio_mantel

        r, p, n = skbio_mantel(
            dm1,
            dm2,
            method=cfg.mantel_method,
            permutations=cfg.mantel_permutations,
        )
        return MantelResult(float(r), float(p), int(n), cfg.mantel_method)
    except ImportError:
        return _mantel_fallback(dm1, dm2, cfg)


def _mantel_fallback(dm1: np.ndarray, dm2: np.ndarray, cfg: EvalConfig) -> MantelResult:
    from scipy.stats import pearsonr, spearmanr

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    if dm1.shape != dm2.shape or dm1.shape[0] != dm1.shape[1]:
        raise ValueError("Mantel inputs must be two square matrices of equal shape.")

    corr = spearmanr if cfg.mantel_method == "spearman" else pearsonr
    iu = np.triu_indices_from(dm1, k=1)
    v1, v2 = dm1[iu], dm2[iu]
    obs = corr(v1, v2)[0]

    rng = np.random.default_rng(0)
    n = dm1.shape[0]
    count = 0
    for _ in range(cfg.mantel_permutations):
        perm = rng.permutation(n)
        permuted = dm2[np.ix_(perm, perm)][iu]
        if abs(corr(v1, permuted)[0]) >= abs(obs):
            count += 1
    p = (count + 1) / (cfg.mantel_permutations + 1)
    return MantelResult(float(obs), float(p), cfg.mantel_permutations, cfg.mantel_method)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """Builds combined datasets and runs the Mantel evaluation per model."""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def _combined_mantel(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> MantelResult:
        dm_pred = embed_and_distance(predicted, self.cfg)
        dm_true = embed_and_distance(ground_truth, self.cfg)
        return mantel_test(dm_pred, dm_true, self.cfg)

    def evaluate_model_a(
        self,
        true_micro: np.ndarray,
        true_metab: np.ndarray,
        imputed_metab: np.ndarray,
    ) -> MantelResult:
        """Model A: imputed metabolome combined with the true microbiome."""
        predicted = np.hstack([true_micro, imputed_metab])
        ground_truth = np.hstack([true_micro, true_metab])
        return self._combined_mantel(predicted, ground_truth)

    def evaluate_model_b(
        self,
        true_micro: np.ndarray,
        true_metab: np.ndarray,
        imputed_micro: np.ndarray,
    ) -> MantelResult:
        """Model B: imputed microbiome combined with the true metabolome."""
        predicted = np.hstack([imputed_micro, true_metab])
        ground_truth = np.hstack([true_micro, true_metab])
        return self._combined_mantel(predicted, ground_truth)
