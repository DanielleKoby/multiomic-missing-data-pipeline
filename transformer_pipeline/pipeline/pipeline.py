"""End-to-end orchestration: data -> models -> training -> evaluation -> (impute).

This is the high-level façade that wires the modular components together. A
typical run is just::

    from pipeline.config import Config
    from pipeline.pipeline import ImputationPipeline

    cfg = Config()
    cfg.embedding.microbiome_embedding_file = "embeddings/microbe_node2vec.csv"
    cfg.embedding.metabolome_embedding_file = "embeddings/metabolite_node2vec.csv"

    pipe = ImputationPipeline(cfg)
    pipe.run()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import (
    EmbeddingProvider,
    MetabolomePreprocessor,
    MicrobiomePreprocessor,
    OmicsData,
    SplitIndices,
    StratifiedSplitter,
    make_loader,
)
from .evaluation import Evaluator, MantelResult, per_feature_pearson
from .models import build_model_a, build_model_b
from .trainer import Trainer


@dataclass
class FittedArtifacts:
    data: OmicsData
    split: SplitIndices
    metab_pre: MetabolomePreprocessor
    micro_emb: np.ndarray
    metab_emb: np.ndarray
    # True where a metabolite embedding row is random-init (island/missing) and
    # should be learned by Model B; calibrated rows are False (frozen).
    metab_trainable_mask: np.ndarray
    # standardized metabolome frames keyed by split
    metab_train_std: pd.DataFrame
    metab_val_std: pd.DataFrame
    # Microbiome frames as fed to / predicted by the models. When
    # cfg.data.microbiome_clr is True these are standardized-CLR; otherwise they
    # are the raw relative-abundance frames. micro_pre is the fitted CLR
    # preprocessor (None when CLR is off).
    micro_pre: Optional[MicrobiomePreprocessor]
    micro_train_proc: pd.DataFrame
    micro_val_proc: pd.DataFrame


class ImputationPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.evaluator = Evaluator(cfg.eval)
        self.artifacts: Optional[FittedArtifacts] = None
        self.trainer_a: Optional[Trainer] = None
        self.trainer_b: Optional[Trainer] = None

    # ------------------------------------------------------------------ #
    # Stage 1 -- data
    # ------------------------------------------------------------------ #
    def prepare_data(self, split: Optional[SplitIndices] = None) -> FittedArtifacts:
        cfg = self.cfg
        data = OmicsData.load(cfg.data)
        # ``split`` lets a caller inject one fold from StratifiedKFoldSplitter
        # (k-fold CV). When None we fall back to the single stratified 80/20
        # split -- the original behaviour, unchanged.
        if split is None:
            split = StratifiedSplitter(cfg.data).split(data)
        print(
            f"[data] complete={data.complete_mask.sum()} "
            f"train={len(split.train)} val={len(split.val)} "
            f"missing_metab={len(split.missing_metabolome)} "
            f"missing_micro={len(split.missing_microbiome)}"
        )

        # Metabolome preprocessing: fit on TRAIN complete rows only.
        metab_pre = MetabolomePreprocessor(cfg.data)
        metab_pre.fit(data.metabolome.loc[split.train])
        metab_train_std = metab_pre.transform(data.metabolome.loc[split.train])
        metab_val_std = metab_pre.transform(data.metabolome.loc[split.val])

        # Microbiome preprocessing: CLR (+ standardize) when enabled, else the
        # raw relative-abundance frames pass straight through.
        micro_pre: Optional[MicrobiomePreprocessor] = None
        if cfg.data.microbiome_clr:
            micro_pre = MicrobiomePreprocessor(cfg.data)
            micro_pre.fit(data.microbiome.loc[split.train])
            micro_train_proc = micro_pre.transform(data.microbiome.loc[split.train])
            micro_val_proc = micro_pre.transform(data.microbiome.loc[split.val])
            print(f"[micro] CLR enabled (pseudocount={cfg.data.clr_pseudocount}, "
                  f"standardize={cfg.data.standardize_microbiome})")
        else:
            micro_train_proc = data.microbiome.loc[split.train]
            micro_val_proc = data.microbiome.loc[split.val]
            print("[micro] CLR disabled -> raw relative abundance")

        # Strict node2vec embeddings (raises if missing).
        provider = EmbeddingProvider(cfg.embedding)
        micro_emb = provider.microbiome_matrix(data.microbiome_features)
        metab_emb = provider.metabolome_matrix(data.metabolome_features)
        metab_trainable_mask = provider.metabolome_trainable_mask(
            data.metabolome_features
        )
        n_random = int(metab_trainable_mask.sum())
        if cfg.embedding.unfreeze_random_metabolites:
            policy = f"UNFROZEN in Model B ({n_random}/{len(metab_trainable_mask)})"
        else:
            policy = f"all FROZEN (random-init rows={n_random}/{len(metab_trainable_mask)})"
        print(
            f"[embed] micro={micro_emb.shape} metab={metab_emb.shape} "
            f"| embeddings: {policy}"
        )

        self.artifacts = FittedArtifacts(
            data=data,
            split=split,
            metab_pre=metab_pre,
            micro_emb=micro_emb,
            metab_emb=metab_emb,
            metab_trainable_mask=metab_trainable_mask,
            metab_train_std=metab_train_std,
            metab_val_std=metab_val_std,
            micro_pre=micro_pre,
            micro_train_proc=micro_train_proc,
            micro_val_proc=micro_val_proc,
        )
        return self.artifacts

    # ------------------------------------------------------------------ #
    # Stage 2 -- Model A: microbiome -> standardized metabolome
    # ------------------------------------------------------------------ #
    def train_model_a(self) -> Trainer:
        a = self._require_artifacts()
        cfg = self.cfg
        # Input microbiome: standardized-CLR when enabled, else raw abundance.
        micro_train = a.micro_train_proc
        micro_val = a.micro_val_proc

        train_loader = make_loader(
            micro_train, a.metab_train_std, cfg.train.batch_size, True,
            cfg.train.num_workers,
        )
        val_loader = make_loader(
            micro_val, a.metab_val_std, cfg.train.batch_size, False,
            cfg.train.num_workers,
        )

        model = build_model_a(a.micro_emb, a.data.metabolome.shape[1], cfg.model_a)

        true_micro_val = micro_val.values
        true_metab_val = a.metab_val_std.values

        def mantel_eval(m) -> MantelResult:
            m.eval()
            import torch
            with torch.no_grad():
                dev = next(m.parameters()).device
                xb = torch.tensor(true_micro_val, dtype=torch.float32, device=dev)
                imputed = m(xb).cpu().numpy()
            return self.evaluator.evaluate_model_a(
                true_micro_val, true_metab_val, imputed
            )

        def per_feature_r(m) -> float:
            m.eval()
            import torch
            with torch.no_grad():
                dev = next(m.parameters()).device
                xb = torch.tensor(true_micro_val, dtype=torch.float32, device=dev)
                imputed = m(xb).cpu().numpy()
            return per_feature_pearson(true_metab_val, imputed)

        self.trainer_a = Trainer(
            model, cfg.train, name="model_a",
            mantel_eval_fn=mantel_eval,
            per_feature_fn=per_feature_r,
        )
        self.trainer_a.fit(train_loader, val_loader)
        self.trainer_a.save()
        return self.trainer_a

    # ------------------------------------------------------------------ #
    # Stage 3 -- Model B: standardized metabolome -> relative abundance
    # ------------------------------------------------------------------ #
    def train_model_b(self) -> Trainer:
        a = self._require_artifacts()
        cfg = self.cfg
        # Target microbiome: standardized-CLR when enabled, else raw abundance.
        micro_train = a.micro_train_proc
        micro_val = a.micro_val_proc

        train_loader = make_loader(
            a.metab_train_std, micro_train, cfg.train.batch_size, True,
            cfg.train.num_workers,
        )
        val_loader = make_loader(
            a.metab_val_std, micro_val, cfg.train.batch_size, False,
            cfg.train.num_workers,
        )

        # Output head: CLR targets live in (-inf, inf) -> identity head + MSE.
        # Raw relative-abundance targets live on the simplex -> softmax head.
        cfg.model_b.output_activation = (
            "identity" if cfg.data.microbiome_clr else "softmax"
        )

        # Freeze every embedding row by default. Only unfreeze the random-init
        # metabolite rows when explicitly requested via the config flag.
        b_mask = (
            a.metab_trainable_mask
            if cfg.embedding.unfreeze_random_metabolites
            else None
        )
        model = build_model_b(
            a.metab_emb,
            a.data.microbiome.shape[1],
            cfg.model_b,
            trainable_mask=b_mask,
        )

        true_micro_val = micro_val.values
        true_metab_val = a.metab_val_std.values

        def mantel_eval(m) -> MantelResult:
            m.eval()
            import torch
            with torch.no_grad():
                dev = next(m.parameters()).device
                xb = torch.tensor(true_metab_val, dtype=torch.float32, device=dev)
                imputed = m(xb).cpu().numpy()
            return self.evaluator.evaluate_model_b(
                true_micro_val, true_metab_val, imputed
            )

        def per_feature_r(m) -> float:
            m.eval()
            import torch
            with torch.no_grad():
                dev = next(m.parameters()).device
                xb = torch.tensor(true_metab_val, dtype=torch.float32, device=dev)
                imputed = m(xb).cpu().numpy()
            return per_feature_pearson(true_micro_val, imputed)

        self.trainer_b = Trainer(
            model, cfg.train, name="model_b",
            mantel_eval_fn=mantel_eval,
            per_feature_fn=per_feature_r,
        )
        self.trainer_b.fit(train_loader, val_loader)
        self.trainer_b.save()
        return self.trainer_b

    # ------------------------------------------------------------------ #
    # Stage 4 -- optional imputation of the structurally-missing samples
    # ------------------------------------------------------------------ #
    def impute_missing(self) -> None:
        a = self._require_artifacts()
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Model A fills metabolome for samples that have microbiome only.
        if a.split.missing_metabolome and self.trainer_a is not None:
            ids = a.split.missing_metabolome
            micro_df = a.data.microbiome.loc[ids]
            # Match Model A's training input space (CLR when enabled).
            micro_in = (
                a.micro_pre.transform(micro_df).values
                if a.micro_pre is not None
                else micro_df.values
            )
            pred_std = self.trainer_a.predict(micro_in)
            pred_raw = a.metab_pre.inverse_transform(pred_std)
            pd.DataFrame(
                pred_raw, index=ids, columns=a.data.metabolome_features
            ).to_csv(os.path.join(cfg.output_dir, "imputed_metabolome.csv"))
            print(f"[impute] wrote imputed_metabolome.csv ({len(ids)} samples)")

        # Model B fills microbiome for samples that have metabolome only.
        if a.split.missing_microbiome and self.trainer_b is not None:
            ids = a.split.missing_microbiome
            metab_in = a.metab_pre.transform(a.data.metabolome.loc[ids]).values
            pred = self.trainer_b.predict(metab_in)
            # Model B predicts in CLR space when enabled -> invert back to
            # relative abundance (rows sum to 1) before writing.
            if a.micro_pre is not None:
                pred = a.micro_pre.inverse_transform(pred)
            pd.DataFrame(
                pred, index=ids, columns=a.data.microbiome_features
            ).to_csv(os.path.join(cfg.output_dir, "imputed_microbiome.csv"))
            print(f"[impute] wrote imputed_microbiome.csv ({len(ids)} samples)")

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.prepare_data()
        self.train_model_a()
        self.train_model_b()
        if self.cfg.run_imputation:
            self.impute_missing()

    # ------------------------------------------------------------------ #
    def _require_artifacts(self) -> FittedArtifacts:
        if self.artifacts is None:
            self.prepare_data()
        assert self.artifacts is not None
        return self.artifacts
