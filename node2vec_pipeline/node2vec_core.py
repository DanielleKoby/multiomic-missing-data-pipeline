"""
node2vec_core.py
================
The node2vec algorithm itself (Grover & Leskovec, KDD 2016), kept free of any
experiment-design, file-I/O or CLI concerns so it stays a faithful, testable
mirror of the paper.

Single entry point: ``run_node2vec`` — given a weighted graph and one
hyperparameter set, it runs the canonical node2vec pipeline through the
``node2vec`` package (biased 2nd-order walk precompute + sampling, then a
skip-gram fit) and returns the learned embeddings as a node-indexed lookup table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec


def run_node2vec(
    G: nx.Graph,
    *,
    p: float,
    q: float,
    dimensions: int,
    walk_length: int,
    num_walks: int,
    weight_key: str = "weight",
    workers: int = 1,
    window: int = 10,
    min_count: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run node2vec end-to-end and return a node-indexed embedding table.

    This is the canonical, paper-faithful usage of the ``node2vec`` package
      1. ``Node2Vec(...)`` precomputes the biased transition probabilities from
         the return/in-out parameters ``p`` and ``q`` (§3.2.2) and samples
         ``num_walks`` random walks of length ``walk_length`` per node
         (Algorithm 1, steps a-b). ``weight_key="weight"`` makes the sampler
         honor our enzyme-normalized edge weights.
      2. ``.fit(...)`` learns the embeddings from those walks with the skip-gram
         (Word2Vec) objective (§3.1, Algorithm 1 step c).

    Parameters mirror the paper's knobs: ``p`` (return), ``q`` (in-out),
    ``dimensions`` (d), ``walk_length`` (l), ``num_walks`` (r) and the skip-gram
    context ``window`` (k).

    Reproducibility note: results are deterministic for ``seed`` only at
    ``workers=1``. The package seeds the RNG in the parent process, but walks run
    under joblib parallelism and gensim's Word2Vec is itself non-deterministic
    with multiple workers, so ``workers>1`` trades reproducibility for speed.

    Returns
    -------
    pd.DataFrame
        Index = node id; columns = ``node_type`` then dim_0..dim_{d-1}.
    """
    # Phase 1 (Algorithm 1, steps a-b): precompute biased transitions + sample
    # walks. The package builds the alias tables and the walk corpus here.
    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key=weight_key,
        workers=workers,
        seed=seed,
        quiet=True,
    )

    # Phase 2 (Algorithm 1, step c): skip-gram fit over the walks. ``.fit``
    model = n2v.fit(window=window, min_count=min_count, sg=1, seed=seed)

    # Phase 3: build a DataFrame of learned embeddings, aligned with the graph's node order.
    # Iterate G.nodes() so row order is deterministic and aligned with the graph.
    # Every node has a vector
    node_ids = list(G.nodes())
    matrix = np.vstack([model.wv[str(n)] for n in node_ids])

    cols = [f"dim_{i}" for i in range(dimensions)]
    df = pd.DataFrame(matrix, index=node_ids, columns=cols)
    df.index.name = "node_id"

    # Carry node_type (enzyme/microbe/metabolite) through so the Transformer side can split modalities.
    # Insert it as the first column (right after the node_id index) for readability;
    # downstream code selects/drops it by name, not position.
    node_types = nx.get_node_attributes(G, "node_type")
    df.insert(0, "node_type", [node_types.get(n, "unknown") for n in node_ids])
    return df
