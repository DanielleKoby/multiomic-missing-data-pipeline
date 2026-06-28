"""Extract per-modality node2vec embeddings and align them to the data columns.

What this does
--------------
The node2vec run stored every node (microbe + enzyme + metabolite) in one
matrix (``embeddings/<stub>.npy`` + ``<stub>.vocab.json``). The Transformer only
needs the microbe and metabolite rows, each aligned 1:1 to the feature columns
of ``train/microbiome.csv`` and ``train/metabolome.csv`` -- i.e. "who belongs to
who".

Node ids are type-prefixed (``microbe::<name>`` / ``metabolite::<name>``) and the
suffix is the *verbatim* data-column name, so alignment is an exact name join.

Coverage
--------
* microbiome : all 170 columns have a calibrated vector.
* metabolome : 102 columns total. 77 calibrated, 7 "island" (vector exists but
  sits on a disconnected graph component -> NOT comparable), 18 "missing" (no
  node at all). The 7 island ids are invisible from the embedding files alone,
  so they are hard-coded below (see embeddings/README.md). The 25 island+missing
  rows are filled with a seeded random vector and flagged in the status sidecar
  so the Transformer can keep them trainable.

Outputs (embeddings/aligned/)
-----------------------------
* microbe_embeddings.csv          index=microbiome column, cols dim_0..dim_127
* metabolite_embeddings.csv       index=metabolome column, cols dim_0..dim_127
* metabolite_embedding_status.csv index=metabolome column, col "status"
                                  (calibrated | island | missing)

The two embedding CSVs are pure name+vector tables -> drop-in for
``pipeline.data.EmbeddingProvider`` (which reads first column = name, rest = the
128-dim vector and is strict about every feature being present).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (relative to the project root = the folder holding train/ and embeddings/)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STUB = "embeddings_p1.0_q1.0_d128_l80_nw10"
NPY = os.path.join(ROOT, "embeddings", f"{STUB}.npy")
VOCAB = os.path.join(ROOT, "embeddings", f"{STUB}.vocab.json")
MICRO_CSV = os.path.join(ROOT, "train", "microbiome.csv")
METAB_CSV = os.path.join(ROOT, "train", "metabolome.csv")

OUT_DIR = os.path.join(ROOT, "embeddings", "aligned")

EMBED_DIM = 128
SEED = 42

# Metabolites that own a vector but live on a disconnected satellite component:
# the vector is uncalibrated and not comparable to the main graph. Not derivable
# from the embedding files -> carried by hand (see embeddings/README.md).
ISLAND = [
    "SERUM_ABS_1,5_anhydro_D_sorbitol",
    "SERUM_ABS_D_threitol",
    "SERUM_ABS_O_acetylsalicylic_acid",
    "SERUM_ABS_alpha_tocophereol",
    "SERUM_ABS_hexanoic_acid",
    "SERUM_ABS_phytanic_acid",
    "SERUM_ABS_tartronic_acid",
]


def _load_vocab_matrix():
    vocab = json.load(open(VOCAB))
    mat = np.load(NPY).astype(np.float32)
    assert mat.shape[1] == EMBED_DIM, f"expected dim {EMBED_DIM}, got {mat.shape[1]}"
    return vocab, mat


def _data_columns(csv_path):
    # index_col=0 drops the SampleID column; the rest are the feature columns.
    return list(pd.read_csv(csv_path, index_col=0, nrows=0).columns)


def _dim_cols():
    return [f"dim_{i}" for i in range(EMBED_DIM)]


def build_microbe_table(vocab, mat):
    cols = _data_columns(MICRO_CSV)
    rows = []
    for c in cols:
        node = f"microbe::{c}"
        if node not in vocab:  # should never happen for microbiome
            raise ValueError(f"microbe column without embedding: {c}")
        rows.append(mat[vocab[node]])
    df = pd.DataFrame(np.vstack(rows), index=cols, columns=_dim_cols())
    df.index.name = "feature"
    return df


def build_metabolite_table(vocab, mat):
    cols = _data_columns(METAB_CSV)
    island = set(ISLAND)

    # Reference scale for the seeded random vectors: match the per-dimension std
    # of the calibrated (main-component) metabolite vectors so random rows do not
    # dominate the geometry before fine-tuning.
    calibrated_vecs = [
        mat[vocab[f"metabolite::{c}"]]
        for c in cols
        if f"metabolite::{c}" in vocab and c not in island
    ]
    ref_std = float(np.vstack(calibrated_vecs).std()) if calibrated_vecs else 1.0
    rng = np.random.default_rng(SEED)

    rows, status = [], []
    for c in cols:
        node = f"metabolite::{c}"
        if c in island:
            rows.append(rng.normal(0.0, ref_std, EMBED_DIM).astype(np.float32))
            status.append("island")
        elif node in vocab:
            rows.append(mat[vocab[node]])
            status.append("calibrated")
        else:
            rows.append(rng.normal(0.0, ref_std, EMBED_DIM).astype(np.float32))
            status.append("missing")

    emb = pd.DataFrame(np.vstack(rows), index=cols, columns=_dim_cols())
    emb.index.name = "feature"
    stat = pd.DataFrame({"status": status}, index=cols)
    stat.index.name = "feature"
    return emb, stat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vocab, mat = _load_vocab_matrix()

    micro = build_microbe_table(vocab, mat)
    metab, status = build_metabolite_table(vocab, mat)

    micro_path = os.path.join(OUT_DIR, "microbe_embeddings.csv")
    metab_path = os.path.join(OUT_DIR, "metabolite_embeddings.csv")
    status_path = os.path.join(OUT_DIR, "metabolite_embedding_status.csv")
    micro.to_csv(micro_path)
    metab.to_csv(metab_path)
    status.to_csv(status_path)

    counts = status["status"].value_counts().to_dict()
    print(f"microbe_embeddings.csv      : {micro.shape[0]} features x {micro.shape[1]} dims")
    print(f"metabolite_embeddings.csv   : {metab.shape[0]} features x {metab.shape[1]} dims")
    print(f"  status -> {counts}")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
