"""
graph_assembly.py
=================
Build the tri-partite biological graph that node2vec embeds:

        Microbe  --(copy_number, enzyme-normalized)--  Enzyme(EC)  --  Metabolite

The graph is undirected: KEGG COMPOUND records give no substrate/product
direction, so we only encode association, not flow. Each node carries a
``node_type`` ("microbe" | "enzyme" | "metabolite") so the downstream
Transformer can slice the embedding matrix per modality.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd
import networkx as nx


# --- Node ids are type-prefixed so the three namespaces never collide; the raw
#     label is kept in a ``label`` attribute so it can be recovered. -----------
MICROBE_PREFIX = "microbe::"
ENZYME_PREFIX = "enzyme::"
METABOLITE_PREFIX = "metabolite::"


# --- Default Layer-1 inputs: the raw files shipped in data/raw (overridable). --
_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_MICROBE_KO_PATH = os.path.join(_RAW_DIR, "ko_microbiome.csv")
DEFAULT_KO_TO_EC_PATH = os.path.join(_RAW_DIR, "ko_enzyme.list")


def _mk_microbe(label: str) -> str:
    return f"{MICROBE_PREFIX}{label}"


def _mk_enzyme(label: str) -> str:
    return f"{ENZYME_PREFIX}{label}"


def _mk_metabolite(label: str) -> str:
    return f"{METABOLITE_PREFIX}{label}"


def load_ko_ec(path: str) -> dict[str, set[str]]:
    """Parse ``ko_enzyme.list`` into {bare KO id: {bare EC, ...}}.

    The shared enzyme node is the EC number, so microbe->KO is translated to
    microbe->EC. KOs absent here are non-enzyme orthologs and get no edge.
    """
    ko2ec: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            ko, ec = line.split()
            ko2ec[ko.replace("ko:", "")].add(ec.replace("ec:", ""))
    return ko2ec


def add_microbe_enzyme_edges(
    G: nx.Graph,
    df: pd.DataFrame,
    ko2ec: dict[str, set[str]],
    microbe_col: str = "taxonomic_label",
    enzyme_col: str = "ko",
    weight_col: str = "copy_number",
) -> nx.Graph:
    """Add Microbe -- Enzyme(EC) edges, normalized *per enzyme*.

    For every enzyme E the incoming weights from all microbes sum to 1.0:

        W(M -> E) = CN(M, E) / sum_k CN(M_k, E)

    so the random walk prefers microbes that contribute most of the community's
    copy-number for that reaction (the major functional contributors). A KO that
    maps to several ECs adds its copy-number to each.
    """
    work = df[[microbe_col, enzyme_col, weight_col]].dropna()

    # Sum raw copy-number per (microbe, EC).
    cn: dict[tuple[str, str], float] = defaultdict(float)
    for microbe, ko, w in zip(work[microbe_col], work[enzyme_col], work[weight_col]):
        for ec in ko2ec.get(str(ko).replace("ko:", ""), ()):
            cn[(str(microbe), ec)] += float(w)

    # Per-enzyme totals = the normalization denominator.
    enzyme_total: dict[str, float] = defaultdict(float)
    for (microbe, ec), c in cn.items():
        enzyme_total[ec] += c

    for (microbe, ec), c in cn.items():
        m_id, e_id = _mk_microbe(microbe), _mk_enzyme(ec)
        G.add_node(m_id, node_type="microbe", label=microbe)
        G.add_node(e_id, node_type="enzyme", label=ec)
        G.add_edge(m_id, e_id, weight=c / enzyme_total[ec], edge_type="microbe_enzyme")
    return G


def add_enzyme_metabolite_edges(
    G: nx.Graph,
    df: pd.DataFrame,
    source_col: str = "ec",
    target_col: str = "metabolome_column",
    compound_col: str = "kegg_compound",
) -> nx.Graph:
    """Add Enzyme(EC) -- Metabolite edges from the KEGG-derived edge table.

    The metabolite node id is the verbatim ``metabolome_column`` (its downstream
    vector name); the KEGG compound id is kept as a node attribute. Edges carry a
    constant weight 1.0.
    """
    work = df.dropna(subset=[source_col, target_col])
    # The KEGG compound id is optional; zip a None stand-in when the column is absent.
    compounds = work[compound_col] if compound_col in work.columns else [None] * len(work)

    for ec, metab, compound in zip(work[source_col], work[target_col], compounds):
        e_id, met_id = _mk_enzyme(str(ec)), _mk_metabolite(str(metab))
        G.add_node(e_id, node_type="enzyme", label=str(ec))
        attrs = {"node_type": "metabolite", "label": str(metab)}
        if compound is not None:
            attrs["kegg_compound"] = str(compound)
        G.add_node(met_id, **attrs)
        G.add_edge(e_id, met_id, weight=1.0, edge_type="enzyme_metabolite")
    return G


def build_graph(
    *,
    # --- Layer 1: Microbe -- Enzyme(EC) (default to the raw data files) -------
    microbe_ko_path: str = DEFAULT_MICROBE_KO_PATH,  # microbe x KO copy-number table
    ko_to_ec_path: str = DEFAULT_KO_TO_EC_PATH,      # KO -> EC translation list
    # --- Layer 2: Enzyme(EC) -- Metabolite (required) ------------------------
    enzyme_metabolite_df: pd.DataFrame,              # EC<->metabolite edges (from metabolite_mapping.py)
    microbe_col: str = "taxonomic_label",
    enzyme_col: str = "ko",
    weight_col: str = "copy_number",
) -> nx.Graph:
    """Assemble the undirected Microbe -- Enzyme(EC) -- Metabolite graph.

    Built one bipartite layer at a time; each layer's edge table also introduces
    its endpoint nodes (so there is no separate "node" file):

      Layer 1  Microbe -- Enzyme(EC)
          ``microbe_ko_path``  : microbe x KO copy-number table -> microbe AND
                                 enzyme nodes + their (per-enzyme-normalized) edges.
                                 (Already restricted to the studied microbes.)
          ``ko_to_ec_path``    : KO -> EC map, so KO copy-numbers land on EC nodes.

      Layer 2  Enzyme(EC) -- Metabolite
          ``enzyme_metabolite_df`` : EC<->metabolite edge table from
                                 ``metabolite_mapping.py`` -> metabolite nodes +
                                 enzyme-metabolite edges.
    """
    G = nx.Graph()

    # Layer 1: Microbe -- Enzyme(EC).
    ko2ec = load_ko_ec(ko_to_ec_path)
    microbe_ko_df = pd.read_csv(microbe_ko_path)
    add_microbe_enzyme_edges(G, microbe_ko_df, ko2ec, microbe_col, enzyme_col, weight_col)

    # Layer 2: Enzyme(EC) -- Metabolite.
    add_enzyme_metabolite_edges(G, enzyme_metabolite_df)
    return G


def save_graph(G: nx.Graph, path: str) -> None:
    """Write the graph to GraphML (preserves node/edge attributes)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nx.write_graphml(G, path)


if __name__ == "__main__":
    import argparse
    from collections import Counter

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    processed = os.path.join(data, "processed")

    p = argparse.ArgumentParser(description="Assemble the microbe-enzyme-metabolite graph.")
    # Layer 1: Microbe -- Enzyme(EC) (default to the raw data files).
    p.add_argument("--microbe_ko_path", default=DEFAULT_MICROBE_KO_PATH)
    p.add_argument("--ko_to_ec_path", default=DEFAULT_KO_TO_EC_PATH)
    # Layer 2: Enzyme(EC) -- Metabolite (required edge table).
    p.add_argument("--edges_enzyme_metabolite_path",
                   default=os.path.join(processed, "edges_enzyme_metabolite.csv"))
    p.add_argument("--out", default=os.path.join(processed, "graph.graphml"))
    args = p.parse_args()

    enzyme_metabolite_df = pd.read_csv(args.edges_enzyme_metabolite_path)
    G = build_graph(
        microbe_ko_path=os.path.abspath(args.microbe_ko_path),
        ko_to_ec_path=os.path.abspath(args.ko_to_ec_path),
        enzyme_metabolite_df=enzyme_metabolite_df,
    )
    save_graph(G, args.out)
    counts = Counter(nx.get_node_attributes(G, "node_type").values())
    print(f"graph: {dict(counts)} | {G.number_of_edges():,} edges -> {args.out}")
