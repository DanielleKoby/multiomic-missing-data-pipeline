"""Data loading, splitting, preprocessing and dataset construction.

Pipeline overview
-----------------
1.  ``OmicsData.load``          -- read the three CSVs, align on SampleID,
                                   detect the structural-missingness masks.
2.  ``StratifiedSplitter``      -- 80/20 split of the *complete* samples,
                                   stratified on CENTER_C x PATGROUPFINAL_C
                                   (rare strata merged into ``_rare_`` only as
                                   the split key).
3.  ``MetabolomePreprocessor``  -- log1p -> StandardScaler, fit on the
                                   non-missing TRAIN samples only.
4.  ``EmbeddingProvider``       -- strict loader for node2vec embeddings.
5.  ``OmicsDataset`` / loaders  -- yield raw scalar feature vectors; the
                                   scalar x embedding multiplication happens
                                   inside the model (single source of truth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, EmbeddingConfig


# --------------------------------------------------------------------------- #
# 1. Raw data container
# --------------------------------------------------------------------------- #
@dataclass
class OmicsData:
    """Aligned microbiome / metabolome / metadata tables + missingness masks."""

    microbiome: pd.DataFrame          # (n_samples, 170) relative abundance
    metabolome: pd.DataFrame          # (n_samples, 102) raw serum metabolome
    metadata: pd.DataFrame            # (n_samples, k)
    micro_present: pd.Series          # bool, True where microbiome is observed
    metab_present: pd.Series          # bool, True where metabolome is observed

    @property
    def complete_mask(self) -> pd.Series:
        return self.micro_present & self.metab_present

    @property
    def microbiome_features(self) -> List[str]:
        return list(self.microbiome.columns)

    @property
    def metabolome_features(self) -> List[str]:
        return list(self.metabolome.columns)

    @classmethod
    def load(cls, cfg: DataConfig) -> "OmicsData":
        def _path(name: str) -> str:
            return os.path.join(cfg.data_dir, name)

        micro = pd.read_csv(_path(cfg.microbiome_file), index_col=0)
        metab = pd.read_csv(_path(cfg.metabolome_file), index_col=0)
        meta = pd.read_csv(_path(cfg.metadata_file), index_col=0)

        # Align everything to the metadata index (the canonical sample list).
        index = meta.index
        if not (micro.index.equals(index) and metab.index.equals(index)):
            common = index.intersection(micro.index).intersection(metab.index)
            micro, metab, meta = micro.loc[common], metab.loc[common], meta.loc[common]
            index = common

        # Structural missingness = an entire modality row is NaN.
        micro_present = ~micro.isna().all(axis=1)
        metab_present = ~metab.isna().all(axis=1)

        return cls(
            microbiome=micro,
            metabolome=metab,
            metadata=meta,
            micro_present=micro_present,
            metab_present=metab_present,
        )


# --------------------------------------------------------------------------- #
# 2. Stratified split
# --------------------------------------------------------------------------- #
@dataclass
class SplitIndices:
    train: List[str]
    val: List[str]
    # Structurally-missing blocks (used only for final imputation).
    missing_metabolome: List[str]   # have microbiome, lack metabolome
    missing_microbiome: List[str]   # have metabolome, lack microbiome


class StratifiedSplitter:
    """80/20 stratified split on the complete samples.

    The composite label ``CENTER_C + "_" + PATGROUPFINAL_C`` is used for
    stratification. Strata smaller than ``min_stratum_size`` are merged into a
    single ``_rare_`` bucket **only** to form a valid stratification key -- the
    samples retain their true metadata everywhere else.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.merged_strata_: List[str] = []

    def _composite_label(self, meta: pd.DataFrame) -> pd.Series:
        c = meta[self.cfg.center_col].astype(str)
        g = meta[self.cfg.patgroup_col].astype(str)
        return c.str.cat(g, sep="_")

    def _split_key(self, labels: pd.Series) -> pd.Series:
        counts = labels.value_counts()
        rare = counts[counts < self.cfg.min_stratum_size].index.tolist()
        self.merged_strata_ = list(rare)
        if rare:
            labels = labels.where(~labels.isin(rare), other="_rare_")
        return labels

    def split(self, data: OmicsData) -> SplitIndices:
        complete = data.complete_mask
        complete_ids = complete[complete].index
        meta_complete = data.metadata.loc[complete_ids]

        labels = self._composite_label(meta_complete)
        key = self._split_key(labels)

        if self.merged_strata_:
            print(
                f"[split] merged {len(self.merged_strata_)} rare stratum/strata "
                f"into '_rare_' for the split key: {self.merged_strata_}"
            )

        # Any stratum still below 2 after merging (e.g. a lone '_rare_' member)
        # cannot be stratified at all -- assign those samples deterministically
        # to TRAIN and stratify the remainder.
        key_counts = key.value_counts()
        unstratifiable = key.index[key.isin(key_counts[key_counts < 2].index)]
        forced_train = sorted(unstratifiable.tolist())
        if forced_train:
            print(
                f"[split] {len(forced_train)} sample(s) in singleton strata "
                f"forced into train: {forced_train}"
            )

        strat_ids = [i for i in complete_ids if i not in set(forced_train)]
        strat_key = key.loc[strat_ids]

        train_ids, val_ids = train_test_split(
            strat_ids,
            test_size=self.cfg.val_size,
            random_state=self.cfg.random_state,
            stratify=strat_key.values,
        )
        train_ids = list(train_ids) + forced_train

        missing_metab = data.metadata.index[
            data.micro_present & ~data.metab_present
        ].tolist()
        missing_micro = data.metadata.index[
            data.metab_present & ~data.micro_present
        ].tolist()

        return SplitIndices(
            train=sorted(train_ids),
            val=sorted(val_ids),
            missing_metabolome=missing_metab,
            missing_microbiome=missing_micro,
        )


# --------------------------------------------------------------------------- #
# 2b. Stratified K-fold split
# --------------------------------------------------------------------------- #
class StratifiedKFoldSplitter:
    """K-fold cross-validation version of :class:`StratifiedSplitter`.

    Instead of one 80/20 split it produces ``cfg.n_folds`` :class:`SplitIndices`,
    each rotating a different ~1/K slice of the COMPLETE samples into validation.
    Every fold reuses the exact same composite stratification label
    (``CENTER_C + "_" + PATGROUPFINAL_C``) and the same rare-stratum merging, so
    the class balance per fold matches the single-split behaviour. The
    structurally-missing blocks are identical across folds (they are never part
    of train/val, only of final imputation).

    Small-stratum handling mirrors the single-split logic: any stratum with fewer
    than ``n_folds`` members cannot be spread across all folds, so those samples
    are pinned to TRAIN in every fold (never validated). This keeps
    ``sklearn``'s ``StratifiedKFold`` valid (it requires every class to have at
    least ``n_splits`` members) while losing as few samples to train-only as
    possible.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.merged_strata_: List[str] = []

    # Share the labelling logic with the single-split splitter.
    _composite_label = StratifiedSplitter._composite_label
    _split_key = StratifiedSplitter._split_key

    def split(self, data: OmicsData) -> List[SplitIndices]:
        n_folds = int(self.cfg.n_folds)
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

        complete = data.complete_mask
        complete_ids = list(complete[complete].index)
        meta_complete = data.metadata.loc[complete_ids]

        labels = self._composite_label(meta_complete)
        key = self._split_key(labels)
        if self.merged_strata_:
            print(
                f"[kfold] merged {len(self.merged_strata_)} rare stratum/strata "
                f"into '_rare_' for the split key: {self.merged_strata_}"
            )

        # Strata too small to appear in all K folds -> pinned to train always.
        key_counts = key.value_counts()
        too_small = key_counts[key_counts < n_folds].index
        foldable_mask = ~key.isin(too_small)
        forced_train = sorted(key.index[~foldable_mask].tolist())
        if forced_train:
            print(
                f"[kfold] {len(forced_train)} sample(s) in strata with < {n_folds} "
                f"members pinned to train in every fold."
            )

        foldable_ids = [i for i in complete_ids if foldable_mask.loc[i]]
        foldable_key = key.loc[foldable_ids]

        # Shared structurally-missing blocks (constant across folds).
        missing_metab = data.metadata.index[
            data.micro_present & ~data.metab_present
        ].tolist()
        missing_micro = data.metadata.index[
            data.metab_present & ~data.micro_present
        ].tolist()

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.cfg.random_state
        )
        foldable_ids_arr = np.asarray(foldable_ids, dtype=object)
        folds: List[SplitIndices] = []
        for tr_idx, va_idx in skf.split(foldable_ids_arr, foldable_key.values):
            train_ids = list(foldable_ids_arr[tr_idx]) + forced_train
            val_ids = list(foldable_ids_arr[va_idx])
            folds.append(
                SplitIndices(
                    train=sorted(train_ids),
                    val=sorted(val_ids),
                    missing_metabolome=missing_metab,
                    missing_microbiome=missing_micro,
                )
            )
        return folds


# --------------------------------------------------------------------------- #
# 3. Metabolome preprocessing
# --------------------------------------------------------------------------- #
class MetabolomePreprocessor:
    """log -> StandardScaler. Fit on the non-missing TRAIN samples only.

    Microbiome data is intentionally NOT handled here -- it stays as relative
    abundance.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False

    def _log(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.metabolome_log == "log1p":
            return np.log1p(x)
        if self.cfg.metabolome_log == "log2":
            # Data is strictly positive; +epsilon guards against exact zeros.
            return np.log2(x + 1e-8)
        raise ValueError(f"unknown metabolome_log: {self.cfg.metabolome_log}")

    def fit(self, metab_train: pd.DataFrame) -> "MetabolomePreprocessor":
        if metab_train.isna().any().any():
            raise ValueError(
                "MetabolomePreprocessor.fit received NaNs -- pass only the "
                "non-missing training metabolome rows."
            )
        logged = self._log(metab_train.values.astype(np.float64))
        if self.cfg.standardize_metabolome:
            self.scaler = StandardScaler().fit(logged)
        self._fitted = True
        return self

    def transform(self, metab: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MetabolomePreprocessor must be fit before transform.")
        logged = self._log(metab.values.astype(np.float64))
        out = self.scaler.transform(logged) if self.scaler is not None else logged
        return pd.DataFrame(out, index=metab.index, columns=metab.columns)

    def inverse_transform(self, metab_std: np.ndarray) -> np.ndarray:
        """Map standardized-log space back to the original metabolome scale."""
        if not self._fitted:
            raise RuntimeError("MetabolomePreprocessor must be fit first.")
        logged = (
            self.scaler.inverse_transform(metab_std)
            if self.scaler is not None
            else metab_std
        )
        if self.cfg.metabolome_log == "log1p":
            return np.expm1(logged)
        return np.power(2.0, logged) - 1e-8


# --------------------------------------------------------------------------- #
# 3b. Microbiome preprocessing (CLR)
# --------------------------------------------------------------------------- #
class MicrobiomePreprocessor:
    """Centered log-ratio (CLR) -> optional per-feature StandardScaler.

    Fit on the non-missing TRAIN samples only.

    CLR maps compositional relative abundance to log-ratio space, row-wise::

        clr(x)_i = ln(x_i + eps) - mean_j ln(x_j + eps)

    Semantics of the transformed value:
      * a microbe at the sample's geometric-mean abundance -> CLR ~ 0
      * a microbe that is ABSENT (x_i == 0) -> strongly NEGATIVE (never 0)

    So, unlike raw relative abundance, "0" no longer means "absent" -- it means
    "average". This mirrors the standardized metabolome (0 == mean), gives the
    value channel real dynamic range, and makes MSE in this space meaningful
    (raw simplex targets were ~1/170, so MSE was numerically flat).

    ``inverse_transform`` maps standardized-CLR back to a composition via softmax
    (the analytic inverse of CLR up to closure / sum-to-one).
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False

    def _clr(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # CLR is scale-invariant, so simply adding a small pseudocount (rather
        # than renormalising) is enough to admit exact zeros under the log.
        logx = np.log(x + self.cfg.clr_pseudocount)
        return logx - logx.mean(axis=1, keepdims=True)

    def fit(self, micro_train: pd.DataFrame) -> "MicrobiomePreprocessor":
        if micro_train.isna().any().any():
            raise ValueError(
                "MicrobiomePreprocessor.fit received NaNs -- pass only the "
                "non-missing training microbiome rows."
            )
        clr = self._clr(micro_train.values)
        if self.cfg.standardize_microbiome:
            self.scaler = StandardScaler().fit(clr)
        self._fitted = True
        return self

    def transform(self, micro: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MicrobiomePreprocessor must be fit before transform.")
        clr = self._clr(micro.values)
        out = self.scaler.transform(clr) if self.scaler is not None else clr
        return pd.DataFrame(out, index=micro.index, columns=micro.columns)

    def inverse_transform(self, micro_std: np.ndarray) -> np.ndarray:
        """Map standardized-CLR back to relative abundance (rows sum to 1)."""
        if not self._fitted:
            raise RuntimeError("MicrobiomePreprocessor must be fit first.")
        clr = (
            self.scaler.inverse_transform(micro_std)
            if self.scaler is not None
            else np.asarray(micro_std, dtype=np.float64)
        )
        # CLR -> composition: softmax (numerically stable).
        e = np.exp(clr - clr.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 4. Strict node2vec embedding provider
# --------------------------------------------------------------------------- #
class EmbeddingProvider:
    """Loads pre-computed node2vec embeddings, aligned to a feature order.

    STRICT by design: a missing file raises ``FileNotFoundError`` and a feature
    without an embedding raises ``ValueError``. No random / fabricated vectors
    are ever produced.

    Supported file formats (auto-detected by extension):
      * .csv / .tsv : first column = feature name, remaining columns = vector.
      * .parquet    : same layout (first column = feature name).
      * .npy        : a 2-D array; requires a sibling index file
                      ``<stem>_index.txt`` (one feature name per line) OR a
                      0-D object array holding a ``{name: vector}`` dict.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def _load_table(self, path: Optional[str], kind: str) -> Dict[str, np.ndarray]:
        if not path:
            raise FileNotFoundError(
                f"No {kind} embedding file configured. Set "
                f"EmbeddingConfig.{kind}_embedding_file to the node2vec path."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"{kind} embedding file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, index_col=0)
            return {str(k): v.to_numpy(dtype=np.float32) for k, v in df.iterrows()}
        if ext == ".parquet":
            df = pd.read_parquet(path).set_index(pd.read_parquet(path).columns[0])
            return {str(k): v.to_numpy(dtype=np.float32) for k, v in df.iterrows()}
        if ext == ".npy":
            obj = np.load(path, allow_pickle=True)
            if obj.dtype == object and obj.ndim == 0:  # dict payload
                d = obj.item()
                return {str(k): np.asarray(v, dtype=np.float32) for k, v in d.items()}
            idx_path = os.path.splitext(path)[0] + "_index.txt"
            if not os.path.exists(idx_path):
                raise FileNotFoundError(
                    f"{path} is a raw array; expected a sibling index file "
                    f"{idx_path} (one feature name per line)."
                )
            with open(idx_path) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
            arr = np.asarray(obj, dtype=np.float32)
            if len(names) != arr.shape[0]:
                raise ValueError(
                    f"{idx_path} has {len(names)} names but array has "
                    f"{arr.shape[0]} rows."
                )
            return {n: arr[i] for i, n in enumerate(names)}
        raise ValueError(f"Unsupported embedding format: {ext}")

    def _matrix_for(
        self, table: Dict[str, np.ndarray], features: List[str], kind: str
    ) -> np.ndarray:
        missing = [f for f in features if f not in table]
        if missing:
            raise ValueError(
                f"{len(missing)} {kind} feature(s) have no node2vec embedding, "
                f"e.g. {missing[:5]}. The pipeline is strict and will not "
                f"fabricate embeddings."
            )
        mat = np.stack([table[f] for f in features]).astype(np.float32)
        if mat.shape[1] != self.cfg.embed_dim:
            raise ValueError(
                f"{kind} embeddings have dim {mat.shape[1]} but "
                f"EmbeddingConfig.embed_dim={self.cfg.embed_dim}."
            )
        return mat

    def microbiome_matrix(self, features: List[str]) -> np.ndarray:
        table = self._load_table(self.cfg.microbiome_embedding_file, "microbiome")
        return self._matrix_for(table, features, "microbiome")

    def metabolome_matrix(self, features: List[str]) -> np.ndarray:
        table = self._load_table(self.cfg.metabolome_embedding_file, "metabolome")
        return self._matrix_for(table, features, "metabolome")

    def metabolome_trainable_mask(self, features: List[str]) -> np.ndarray:
        """Bool mask over ``features``: True where the embedding row is random-init.

        Driven by ``metabolome_status_file`` (a name -> status sidecar). Any row
        whose status is not exactly ``"calibrated"`` (i.e. ``island``/``missing``)
        is marked trainable. If the file is not configured / not present, every
        row is frozen (all-False) so behaviour matches the strict fixed lookup.
        """
        path = getattr(self.cfg, "metabolome_status_file", None)
        if not path or not os.path.exists(path):
            return np.zeros(len(features), dtype=bool)
        status = pd.read_csv(path, index_col=0)["status"].to_dict()
        return np.array(
            [status.get(f, "calibrated") != "calibrated" for f in features],
            dtype=bool,
        )


# --------------------------------------------------------------------------- #
# 5. Dataset / DataLoaders
# --------------------------------------------------------------------------- #
class OmicsDataset(Dataset):
    """Yields (input_features, target_features) as float tensors.

    The scalar x node2vec multiplication is deliberately *not* done here; the
    model owns the embedding matrix and performs the tokenisation so there is a
    single source of truth for the embeddings.
    """

    def __init__(self, inputs: pd.DataFrame, targets: pd.DataFrame):
        assert inputs.index.equals(targets.index)
        self.ids = list(inputs.index)
        self.X = torch.tensor(inputs.values, dtype=torch.float32)
        self.Y = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def make_loader(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        OmicsDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
