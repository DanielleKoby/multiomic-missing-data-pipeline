"""Entry point for the MetaCardis imputation pipeline.

Usage
-----
    python run.py \
        --data-dir train \
        --micro-emb embeddings/microbe_node2vec.csv \
        --metab-emb embeddings/metabolite_node2vec.csv

All hyperparameters have sensible defaults (see pipeline/config.py). The two
node2vec embedding paths are the only *required* inputs -- the pipeline is
strict and will raise a clean FileNotFoundError if they are not provided.
"""

from __future__ import annotations

import argparse

from pipeline.config import Config
from pipeline.pipeline import ImputationPipeline


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.data.data_dir = args.data_dir
    cfg.embedding.embed_dim = args.embed_dim
    cfg.embedding.microbiome_embedding_file = args.micro_emb
    cfg.embedding.metabolome_embedding_file = args.metab_emb

    cfg.train.batch_size = args.batch_size
    cfg.train.max_epochs = args.max_epochs
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.selection_metric = args.selection_metric

    cfg.eval.mantel_method = args.mantel_method
    cfg.eval.mantel_permutations = args.permutations

    for mc in (cfg.model_a, cfg.model_b):
        mc.n_blocks = args.blocks
        mc.n_heads = args.heads
        mc.pooling = args.pooling
        mc.tokenizer = args.tokenizer

    cfg.run_imputation = args.impute
    cfg.output_dir = args.output_dir
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="MetaCardis multi-omic imputation pipeline")
    p.add_argument("--data-dir", default="train")
    p.add_argument("--micro-emb", required=True, help="node2vec embeddings for microbes")
    p.add_argument("--metab-emb", required=True, help="node2vec embeddings for metabolites")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pooling", choices=["cls", "gap"], default="cls")
    p.add_argument(
        "--tokenizer",
        choices=["scale", "affine", "add"],
        default="scale",
        help="value/embedding fusion: 'scale'=value*emb (default; won the "
             "CLR grid), 'add'=emb+slope*value, 'affine'=emb*(1+slope*value)",
    )
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--selection-metric",
        choices=["mantel", "val_mse", "per_feature_r"],
        default="val_mse",
    )
    p.add_argument("--mantel-method", choices=["spearman", "pearson"], default="spearman")
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--impute", action="store_true", help="run final imputation stage")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    cfg = build_config(args)
    ImputationPipeline(cfg).run()


if __name__ == "__main__":
    main()
