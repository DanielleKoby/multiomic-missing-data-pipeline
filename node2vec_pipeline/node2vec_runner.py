"""
node2vec_runner.py
==================
Experiment driver for node2vec on the multi-omic graph (Grover & Leskovec,
KDD 2016). The algorithm itself lives in ``node2vec_core.py``; this module owns
everything around it: hyperparameter-grid planning, persistence, the grid driver
and the CLI.

Takes the graph built by ``graph_assembly.py`` and runs node2vec under a grid of
hyperparameters. Each configuration is written to its own **self-describing
folder** under ``out_dir``::

    embeddings/
        manifest.csv                          # cross-run index (all versions/configs)
        v1_p1.0_q1.0_d128_l80_nw10/           # one run: v<mapping_version>_<params>
            embeddings_p1.0_q1.0_d128_l80_nw10.csv
            embeddings_p1.0_q1.0_d128_l80_nw10.npy
            embeddings_p1.0_q1.0_d128_l80_nw10.vocab.json
            README.md                         # copied in, so the folder is self-contained

The leading ``v<n>`` is the **metabolite -> KEGG mapping version** (bump it when
the metabolome.csv -> KEGG matching improves); it is independent of the node2vec
hyperparameters. ``manifest.csv`` stays at the top level and is NOT copied into
the per-run folders.

Embedding output format (Transformer-ready)
-------------------------------------------
Each saved table is a **lookup matrix** indexed by node id:

    node_id                     node_type     dim_0     dim_1   ...  dim_{d-1}
    microbe::Agathob...         microbe        0.0123   -0.4567  ...   0.0891
    enzyme::1.1.5.2             enzyme        -0.2210    0.1934  ...  -0.0042
    metabolite::SERUM_ABS_...   metabolite     0.3380    0.0021  ...   0.1120

* The CSV index is the prefixed node id; column ``node_type`` is included so you
  can build per-modality vocabularies (microbe / enzyme / metabolite).
* A companion ``.npy`` (pure float matrix) and ``vocab .json`` (id -> row index)
  are also written for fast tensor loading without pandas.
"""

from __future__ import annotations

import os
import json
import shutil
import itertools
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx

import graph_assembly  # local module
from node2vec_core import run_node2vec


# ===========================================================================
# 1. Hyperparameter grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class N2VParams:
    """A single node2vec configuration."""
    p: float          # return parameter (likelihood of revisiting a node)
    q: float          # in-out parameter (BFS-like q>1 vs DFS-like q<1)
    d: int            # embedding dimensions
    l: int            # walk length
    num_walks: int    # number of walks per node

    def filename_stub(self) -> str:
        """Self-describing stub used for all output files of this run."""
        return f"p{self.p}_q{self.q}_d{self.d}_l{self.l}_nw{self.num_walks}"


def build_param_grid(
    p_values: Iterable[float] = (1.0, 0.5, 2.0),
    q_values: Iterable[float] = (1.0, 0.5, 2.0),
    d_values: Iterable[int] = (128,),
    l_values: Iterable[int] = (80,),
    num_walks_values: Iterable[int] = (10,),
) -> list[N2VParams]:
    """
    Cartesian product of all hyperparameter axes.
    """
    return [
        N2VParams(p=p, q=q, d=d, l=l, num_walks=nw)
        for p, q, d, l, nw in itertools.product(
            p_values, q_values, d_values, l_values, num_walks_values
        )
    ]


# ===========================================================================
# 2. Persistence (CSV + .npy + vocab.json)
# ---------------------------------------------------------------------------
def run_dir_name(params: N2VParams, version: str) -> str:
    """Self-describing folder name for one run: ``v<version>_<param-stub>``.

    The ``v<version>`` prefix records the metabolite -> KEGG mapping version, so
    re-running after the mapping improves lands in a *new* folder rather than
    overwriting the old one.
    """
    return f"v{version}_{params.filename_stub()}"


def save_embeddings(
    df: pd.DataFrame,
    params: N2VParams,
    out_dir: str,
    version: str = "1",
) -> dict:
    """
    Save one configuration's embeddings into its own ``v<version>_<stub>/`` folder
    under ``out_dir``, in three Transformer-friendly forms:

      * ``embeddings_<stub>.csv``  : human-readable lookup table (index = node id)
      * ``embeddings_<stub>.npy``  : float32 [N, d] matrix for fast tensor load
      * ``embeddings_<stub>.vocab.json`` : {node_id -> row index} alignment map

    The .npy row order matches the vocab map exactly, so::

        mat = np.load("embeddings_<stub>.npy")
        vocab = json.load(open("embeddings_<stub>.vocab.json"))
        mat[vocab["microbe::Agathobacter rectalis"]]   # that node's vector

    If a canonical ``README.md`` exists at ``out_dir`` it is copied into the run
    folder so each folder is a self-contained handoff. The cross-run
    ``manifest.csv`` is written by the caller and is NOT placed here.
    """
    run_dir = os.path.join(out_dir, run_dir_name(params, version))
    os.makedirs(run_dir, exist_ok=True)
    stub = params.filename_stub()

    csv_path = os.path.join(run_dir, f"embeddings_{stub}.csv")
    npy_path = os.path.join(run_dir, f"embeddings_{stub}.npy")
    vocab_path = os.path.join(run_dir, f"embeddings_{stub}.vocab.json")

    # CSV (includes node_type column).
    df.to_csv(csv_path)

    # Pure numeric matrix (drop node_type) + vocab for index alignment.
    numeric = df.drop(columns=["node_type"], errors="ignore")
    np.save(npy_path, numeric.values.astype(np.float32))
    vocab = {nid: i for i, nid in enumerate(numeric.index)}
    with open(vocab_path, "w") as fh:
        json.dump(vocab, fh)

    # Copy the handoff README in, so the folder stands alone.
    readme_src = os.path.join(out_dir, "README.md")
    readme_path = os.path.join(run_dir, "README.md") if os.path.exists(readme_src) else None
    if readme_path is not None:
        shutil.copy2(readme_src, readme_path)

    return {"csv": csv_path, "npy": npy_path, "vocab": vocab_path, "readme": readme_path}


# ===========================================================================
# 3. Grid driver
# ---------------------------------------------------------------------------
def run_grid(
    G: nx.Graph,
    grid: list[N2VParams],
    out_dir: str = "embeddings",
    *,
    version: str = "1",
    weight_key: str = "weight",
    workers: int = 1,
    window: int = 10,
    min_count: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run every configuration in ``grid`` and persist results.

    Each config runs node2vec independently via ``run_node2vec`` (biased-walk
    precompute + sampling + skip-gram fit), then its embedding table is saved
    into its own ``v<version>_<stub>/`` folder. ``version`` is the metabolite ->
    KEGG mapping version (folder-name prefix), independent of the hyperparameters.

    Returns a manifest DataFrame (one row per run) recording the version, params
    and output paths — handy for later selecting which embedding to load. The
    manifest is written once at ``out_dir`` (top level), not inside the folders.
    """
    manifest_rows = []
    for params in grid:
        df = run_node2vec(
            G,
            p=params.p,
            q=params.q,
            dimensions=params.d,
            walk_length=params.l,
            num_walks=params.num_walks,
            weight_key=weight_key,
            workers=workers,
            window=window,
            min_count=min_count,
            seed=seed,
        )
        paths = save_embeddings(df, params, out_dir, version=version)
        manifest_rows.append({"v": version, **asdict(params), **paths})
        print(f"{run_dir_name(params, version)} -> {paths['csv']}")

    manifest = pd.DataFrame(manifest_rows)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"manifest -> {manifest_path}")
    return manifest


# ===========================================================================
# 4. CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    processed = os.path.join(data, "processed")

    p = argparse.ArgumentParser(
        description="Run node2vec over a hyperparameter grid on the multi-omic graph."
    )
    # Either load a prebuilt graph (debug convenience) or assemble it from raw data.
    p.add_argument("--graphml", default=None,
                   help="Path to a prebuilt .graphml; if omitted, the graph is built from raw data.")
    p.add_argument("--out_dir", default=os.path.join(processed, "embeddings"))
    p.add_argument("--workers", type=int, default=1)
    # Metabolite -> KEGG mapping version; prefixes each run folder (v<n>_...).
    # Bump it when you improve the metabolome.csv -> KEGG matching.
    p.add_argument("--v", default="1")
    # Hyperparameter axes (comma-separated lists). Defaults = common-practice
    # single values (node2vec paper: p=q=1 unbiased walk, d=128, l=80, nw=10).
    p.add_argument("--p", default="1.0")
    p.add_argument("--q", default="1.0")
    p.add_argument("--d", default="128")
    p.add_argument("--l", default="80")
    p.add_argument("--num_walks", default="10")
    args = p.parse_args()

    if args.graphml:
        G = nx.read_graphml(args.graphml)
    else:
        # Layer 2 — Enzyme(EC) -- Metabolite edges. The Layer 1 microbe/enzyme
        # inputs default to data/raw inside graph_assembly.build_graph.
        edges_enzyme_metabolite_path = os.path.join(processed, "edges_enzyme_metabolite.csv")
        enzyme_metabolite_df = pd.read_csv(edges_enzyme_metabolite_path)
        G = graph_assembly.build_graph(enzyme_metabolite_df=enzyme_metabolite_df)

    def axis(s, cast):  # parse a comma-separated flag into a list, e.g. "64,128" -> [64, 128]
        return [cast(x) for x in s.split(",") if x]

    grid = build_param_grid(
        p_values=axis(args.p, float),
        q_values=axis(args.q, float),
        d_values=axis(args.d, int),
        l_values=axis(args.l, int),
        num_walks_values=axis(args.num_walks, int),
    )
    run_grid(G, grid, out_dir=args.out_dir, version=args.v, workers=args.workers)
