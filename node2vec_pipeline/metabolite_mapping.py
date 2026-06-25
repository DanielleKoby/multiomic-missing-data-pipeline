"""
metabolite_mapping.py
=====================
Build the KEGG-derived mapping that connects the three node types of the
tri-partite graph (microbe -- enzyme -- metabolite), with the **EC number** as
the shared enzyme node:

        taxonomic_label --(KO->EC)--> EC --(KEGG COMPOUND)--> metabolite

This module produces three reviewable CSV artifacts under ``data/``:

1. ``metabolite_kegg_map.csv``     metabolome column  -> KEGG compound id
2. ``edges_microbe_enzyme.csv``    taxonomic_label    -> EC   (weighted by copy_number)
3. ``edges_enzyme_metabolite.csv`` EC                 -> metabolite (KEGG compound)

Design notes
------------
* Metabolome columns are free-text lab names (``SERUM_ABS_<name>_<N>``) with no
  KEGG ids. We resolve them with a 3-stage cascade:
    (1) offline match against the KEGG COMPOUND synonym index (parsed from
        ``compound.txt`` with Biopython),
    (2) online ``Bio.KEGG.REST.kegg_find`` for residuals (cached to disk so
        rebuilds are reproducible offline),
    (3) a hand-curated ``metabolite_kegg_overrides.csv`` (also lets the user
        *correct* a wrong auto-match -- overrides take top priority).
  Every metabolome column is written to the map CSV, matched or not, so the
  misses are explicit and easy to review.
* Edge direction is NOT derivable from ``compound.txt`` (the ENZYME field is an
  undirected association), so the graph stays undirected downstream.
* ``edges_enzyme_metabolite`` is restricted to ECs actually carried by the
  cohort microbes -- an EC that no microbe carries would dangle off a metabolite
  with no microbe behind it, adding a dead-end stub and no microbe<->metabolite
  path.

Author: multi-omic-missing-data-pipeline
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from collections import defaultdict
from typing import Optional

import pandas as pd
from Bio.KEGG import REST
from Bio.KEGG.Compound import parse as parse_compound


# ===========================================================================
# Default file locations (relative to this file's ../data directory)
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "data"))
RAW_DIR = os.path.join(_DATA, "raw")              # inputs received from advisors
PROCESSED_DIR = os.path.join(_DATA, "processed")  # artifacts this pipeline produces

# --- raw inputs ---
COMPOUND_TXT = os.path.join(RAW_DIR, "compound.txt")
KO_ENZYME_LIST = os.path.join(RAW_DIR, "ko_enzyme.list")
KO_MICROBIOME_CSV = os.path.join(RAW_DIR, "ko_microbiome.csv")
MICROBIOME_CSV = os.path.join(RAW_DIR, "microbiome.csv")
METABOLOME_CSV = os.path.join(RAW_DIR, "metabolome.csv")

# --- processed outputs / curated config ---
MAP_CSV = os.path.join(PROCESSED_DIR, "metabolite_kegg_map.csv")
OVERRIDES_CSV = os.path.join(PROCESSED_DIR, "metabolite_kegg_overrides.csv")
REST_CACHE_CSV = os.path.join(PROCESSED_DIR, "kegg_find_cache.csv")
EDGES_ME_CSV = os.path.join(PROCESSED_DIR, "edges_microbe_enzyme.csv")
EDGES_EM_CSV = os.path.join(PROCESSED_DIR, "edges_enzyme_metabolite.csv")


# ===========================================================================
# 1. Name normalization
# ---------------------------------------------------------------------------
# A single canonical key collapses cosmetic differences (case, separators) so
# that a metabolome label and a KEGG synonym describing the same molecule hash
# to the same string. e.g.  "L-Glutamic acid"  ==  "L_glutamic_acid"  -> "l glutamic acid".
# ===========================================================================
_SERUM_PREFIX = re.compile(r"^SERUM_ABS_", re.IGNORECASE)
_TRAILING_IDX = re.compile(r"_\d+$")          # replicate-disambiguation suffix (_1, _2, ...)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def canon(name: str) -> str:
    """Lower-case, drop non-alphanumeric runs -> single space. Order preserved."""
    return _NON_ALNUM.sub(" ", name.lower()).strip()


def clean_metabolome_name(col: str) -> str:
    """Strip the ``SERUM_ABS_`` prefix and trailing ``_<digit>`` replicate index.

    Used for matching ONLY -- the original column string is preserved verbatim
    everywhere it is stored.
    """
    s = _SERUM_PREFIX.sub("", col)
    s = _TRAILING_IDX.sub("", s)
    return s


# ===========================================================================
# 2. Parse KEGG COMPOUND  (Biopython)
# ---------------------------------------------------------------------------
def load_kegg_compounds(compound_txt: str = COMPOUND_TXT):
    """Parse ``compound.txt`` into lookup tables.

    Returns
    -------
    name_index : dict[str, str]
        canonical-name -> compound id (every synonym points back to its cid).
        First writer wins on collisions (KEGG lists the canonical name first).
    cid_primary : dict[str, str]
        compound id -> KEGG primary (first-listed) name.
    cid_synonyms : dict[str, list[str]]
        compound id -> original synonym list (for the audit trail).
    cid_enzymes : dict[str, set[str]]
        compound id -> set of EC numbers (bare, e.g. "1.1.1.1").
    """
    name_index: dict[str, str] = {}
    cid_primary: dict[str, str] = {}
    cid_synonyms: dict[str, list[str]] = {}
    cid_enzymes: dict[str, set[str]] = {}

    with open(compound_txt) as handle:
        for rec in parse_compound(handle):
            cid = rec.entry
            names = list(rec.name)
            cid_primary[cid] = names[0] if names else cid
            cid_synonyms[cid] = names
            cid_enzymes[cid] = set(rec.enzyme)
            for nm in names:
                key = canon(nm)
                if key and key not in name_index:
                    name_index[key] = cid

    print(f"[load_kegg_compounds] {len(cid_primary):,} compounds | "
          f"{len(name_index):,} unique name keys")
    return name_index, cid_primary, cid_synonyms, cid_enzymes


# ===========================================================================
# 3. REST fallback (online) with on-disk cache
# ---------------------------------------------------------------------------
def _load_rest_cache(path: str = REST_CACHE_CSV) -> dict[str, tuple[str, str]]:
    cache: dict[str, tuple[str, str]] = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cache[row["query"]] = (row["kegg_compound"], row.get("matched_name", ""))
    return cache


def _save_rest_cache(cache: dict[str, tuple[str, str]], path: str = REST_CACHE_CSV) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "kegg_compound", "matched_name"])
        for q, (cid, nm) in sorted(cache.items()):
            w.writerow([q, cid, nm])


def kegg_find_compound(query: str) -> tuple[str, str]:
    """Query the KEGG REST API for a compound by name.

    Returns ``(compound_id, matched_name)``, or ``("", "")`` on a miss.

    IMPORTANT: KEGG's ``find`` is a keyword AND-search that returns *any*
    compound whose name contains the query words (lowest C-number first), which
    yields wrong hits like ``gluconic acid -> D-Glucono-1,5-lactone`` or
    ``2,3-butanediol -> Dithiothreitol``. To stay biologically accurate we accept
    a result ONLY when it is unambiguous:
      * a returned synonym is an EXACT (canonicalized) match to the query, OR
      * there is exactly ONE hit (a unique keyword-complete compound).
    Ambiguous multi-hit responses are left for a manual override, not guessed.
    Empirically this cleanly separates real matches (capric acid, aspirin,
    indole-3-acetate -> 1 hit each) from false positives (gluconic acid,
    2,3-butanediol, xylulose -> 7-20 hits each).
    """
    # KEGG rejects commas/underscores (HTTP 400); multi-keyword queries join with
    # '+' (its AND separator) -- spaces are not URL-encoded by Biopython.
    safe = re.sub(r"[^0-9a-zA-Z -]+", " ", query)
    safe = "+".join(safe.split())
    if not safe:
        return "", ""
    try:
        text = REST.kegg_find("compound", safe).read()
    except Exception as exc:  # network / API hiccup -> treat as miss
        print(f"[kegg_find_compound] REST error for {query!r}: {exc}", file=sys.stderr)
        return "", ""

    hits = []  # [(cid, [names])]
    for line in text.splitlines():
        if not line.strip():
            continue
        ident, _, names = line.partition("\t")
        cid = ident.replace("cpd:", "").strip()
        if cid:
            hits.append((cid, [n.strip() for n in names.split(";")]))

    qk = canon(query)
    # 1) exact synonym match anywhere in the results
    for cid, names in hits:
        for nm in names:
            if canon(nm) == qk:
                return cid, nm
    # 2) a single unambiguous hit
    if len(hits) == 1:
        cid, names = hits[0]
        return cid, (names[0] if names else "")
    return "", ""


# ===========================================================================
# 4. Build the metabolite -> KEGG map
# ---------------------------------------------------------------------------
def _load_overrides(path: str = OVERRIDES_CSV) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                col = row.get("metabolome_column", "").strip()
                cid = row.get("kegg_compound", "").strip()
                if col and cid:
                    overrides[col] = cid
    return overrides


def map_metabolites(
    metabolome_csv: str = METABOLOME_CSV,
    *,
    use_rest: bool = True,
    rest_sleep: float = 0.34,
) -> pd.DataFrame:
    """Resolve every metabolome column to a KEGG compound id (or leave unmatched).

    Cascade: manual override > offline synonym match > REST fallback.

    Returns a DataFrame with columns:
        metabolome_column, kegg_compound, kegg_primary_name, matched_synonym, match_method
    """
    name_index, cid_primary, cid_synonyms, _ = load_kegg_compounds()
    overrides = _load_overrides()
    cache = _load_rest_cache()

    # metabolome column names (verbatim) -- header row, drop the index column.
    with open(metabolome_csv, newline="") as f:
        columns = next(csv.reader(f))[1:]

    rows = []
    cache_dirty = False
    for col in columns:
        cid = ""
        method = "unmatched"
        matched_syn = ""        # KEGG synonym our name matched on (audit trail)

        # 1) manual override (top priority -- also fixes wrong auto-matches)
        if col in overrides:
            cid, method = overrides[col], "manual_override"
        else:
            cleaned = clean_metabolome_name(col)
            key = canon(cleaned)
            # 2) offline synonym-index match
            if key in name_index:
                cid, method = name_index[key], "offline_name"
            # 3) REST fallback (cached, exact-match only)
            elif use_rest:
                if key in cache:
                    cid, rest_name = cache[key]
                else:
                    cid, rest_name = kegg_find_compound(cleaned)
                    cache[key] = (cid, rest_name)
                    cache_dirty = True
                    time.sleep(rest_sleep)
                if cid:
                    method, matched_syn = "kegg_rest", rest_name

        # for offline / override hits, recover the matched synonym from our parse
        if cid and not matched_syn:
            ck = canon(clean_metabolome_name(col))
            for syn in cid_synonyms.get(cid, []):
                if canon(syn) == ck:
                    matched_syn = syn
                    break

        rows.append({
            "metabolome_column": col,                     # verbatim, never altered
            "kegg_compound": cid,
            "kegg_primary_name": cid_primary.get(cid, "") if cid else "",
            "matched_synonym": matched_syn,
            "match_method": method,
        })

    if cache_dirty:
        _save_rest_cache(cache)

    df = pd.DataFrame(rows)
    n_matched = (df["kegg_compound"] != "").sum()
    print(f"[map_metabolites] matched {n_matched}/{len(df)} metabolome columns")
    by_method = df["match_method"].value_counts().to_dict()
    print(f"[map_metabolites] by method: {by_method}")
    return df


# ===========================================================================
# 5. KO -> EC and the two edge tables
# ---------------------------------------------------------------------------
def load_ko_ec(path: str = KO_ENZYME_LIST) -> dict[str, set[str]]:
    """Parse ``ko_enzyme.list`` -> {ko_id: {ec, ...}} with bare ids.

    Keys are bare KO ids ("K00004"), values bare EC numbers ("1.1.1.4") to
    match both ``ko_microbiome.csv`` (after stripping "ko:") and the KEGG
    COMPOUND ENZYME field.
    """
    ko2ec: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            ko, ec = line.split()
            ko2ec[ko.replace("ko:", "")].add(ec.replace("ec:", ""))
    return ko2ec


def build_microbe_enzyme_edges(
    ko_microbiome_csv: str = KO_MICROBIOME_CSV,
    microbiome_csv: str = MICROBIOME_CSV,
    ko_enzyme_list: str = KO_ENZYME_LIST,
) -> pd.DataFrame:
    """taxonomic_label -> EC edges (one row per assembly/KO/EC), weighted by copy_number.

    * Taxa are restricted to species present as columns in ``microbiome.csv``.
    * KOs with no EC mapping are dropped (non-enzyme orthologs).
    * A KO with several ECs expands to one row per EC.
    """
    ko2ec = load_ko_ec(ko_enzyme_list)

    with open(microbiome_csv, newline="") as f:
        keep_taxa = set(c.strip() for c in next(csv.reader(f))[1:])

    df = pd.read_csv(ko_microbiome_csv)
    df = df[df["taxonomic_label"].isin(keep_taxa)].copy()
    df["ko_bare"] = df["ko"].str.replace("ko:", "", regex=False)

    out = []
    for tax, ko, cn in zip(df["taxonomic_label"], df["ko_bare"], df["copy_number"]):
        for ec in ko2ec.get(ko, ()):                      # no EC -> contributes nothing
            out.append((tax, ec, cn))

    edges = pd.DataFrame(out, columns=["taxonomic_label", "ec", "copy_number"])
    print(f"[build_microbe_enzyme_edges] {len(edges):,} edges | "
          f"{edges['taxonomic_label'].nunique()} taxa | {edges['ec'].nunique()} ECs")
    return edges


def build_enzyme_metabolite_edges(
    metabolite_map: pd.DataFrame,
    cohort_ecs: set[str],
    compound_txt: str = COMPOUND_TXT,
    *,
    restrict_to_cohort: bool = True,
) -> pd.DataFrame:
    """EC -> metabolite edges for every compound in the map.

    ``restrict_to_cohort=True`` keeps only ECs that some cohort microbe carries
    (avoids dead-end enzyme stubs; tightest microbe<->metabolite graph).

    ``restrict_to_cohort=False`` keeps ALL KEGG enzymes of each compound, even
    those no microbe carries. Those extra enzyme nodes link only metabolites, so
    metabolites that share an enzyme become connected (metabolite<->metabolite
    via a shared EC) -- this gives an embedding to metabolites that have KEGG
    enzymes but none in the cohort, at the cost of a larger, looser graph.
    """
    _, _, _, cid_enzymes = load_kegg_compounds(compound_txt)

    mapped = metabolite_map[metabolite_map["kegg_compound"] != ""]
    out = []
    for col, cid in zip(mapped["metabolome_column"], mapped["kegg_compound"]):
        for ec in cid_enzymes.get(cid, ()):
            if restrict_to_cohort and ec not in cohort_ecs:
                continue
            out.append((ec, cid, col))

    edges = pd.DataFrame(out, columns=["ec", "kegg_compound", "metabolome_column"])
    print(f"[build_enzyme_metabolite_edges] {len(edges):,} edges | "
          f"{edges['ec'].nunique()} ECs | {edges['metabolome_column'].nunique()} metabolites "
          f"({'cohort-only' if restrict_to_cohort else 'ALL enzymes'})")
    return edges


# ===========================================================================
# 5b. Curation helper -- overrides template with KEGG candidate suggestions
# ---------------------------------------------------------------------------
def kegg_find_candidates(query: str, max_candidates: int = 8) -> list[tuple[str, str]]:
    """Return up to ``max_candidates`` (compound_id, primary_name) REST hits.

    Unlike :func:`kegg_find_compound` this does NOT decide -- it lists options so
    a human can pick the right one in the overrides file.
    """
    safe = "+".join(re.sub(r"[^0-9a-zA-Z -]+", " ", query).split())
    if not safe:
        return []
    try:
        text = REST.kegg_find("compound", safe).read()
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        ident, _, names = line.partition("\t")
        cid = ident.replace("cpd:", "").strip()
        primary = names.split(";")[0].strip() if names else ""
        if cid:
            out.append((cid, primary))
        if len(out) >= max_candidates:
            break
    return out


def write_overrides_template(
    metabolite_map: pd.DataFrame,
    path: str = OVERRIDES_CSV + ".template",
    use_rest: bool = True,
) -> None:
    """Write a curation worksheet for the rows that need a human decision.

    Includes every UNMATCHED column plus the low-confidence ``kegg_rest`` auto
    -matches (which can still be wrong, e.g. a single-hit false positive). For
    each, lists candidate KEGG compounds from REST so the user only has to copy
    the right id into ``kegg_compound`` and save the file as
    ``metabolite_kegg_overrides.csv`` (read on the next pipeline run).
    """
    review = metabolite_map[
        (metabolite_map["kegg_compound"] == "")
        | (metabolite_map["match_method"] == "kegg_rest")
    ]
    rows = []
    for _, r in review.iterrows():
        cands = (kegg_find_candidates(clean_metabolome_name(r["metabolome_column"]))
                 if use_rest else [])
        if use_rest:
            time.sleep(0.34)
        rows.append({
            "metabolome_column": r["metabolome_column"],          # keep verbatim
            "kegg_compound": r["kegg_compound"],                  # prefilled if kegg_rest
            "auto_match_method": r["match_method"],
            "auto_primary_name": r["kegg_primary_name"],
            "kegg_candidates": " | ".join(f"{c}:{n}" for c, n in cands),
        })
    out = pd.DataFrame(rows, columns=[
        "metabolome_column", "kegg_compound",
        "auto_match_method", "auto_primary_name", "kegg_candidates",
    ])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[write_overrides_template] wrote {path} ({len(out)} rows to review)")
    print(f"  -> fill in 'kegg_compound', then save as {os.path.basename(OVERRIDES_CSV)}")


# ===========================================================================
# 6. Orchestration
# ===========================================================================
def main(use_rest: bool = True, restrict_to_cohort: bool = True) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=== Step 1: metabolite -> KEGG compound map ===")
    met_map = map_metabolites(use_rest=use_rest)
    met_map.to_csv(MAP_CSV, index=False)
    print(f"  wrote {MAP_CSV}")

    print("\n=== Step 2a: microbe -> EC edges ===")
    me = build_microbe_enzyme_edges()
    me.to_csv(EDGES_ME_CSV, index=False)
    print(f"  wrote {EDGES_ME_CSV}")

    print("\n=== Step 2b: EC -> metabolite edges ===")
    cohort_ecs = set(me["ec"].unique())
    em = build_enzyme_metabolite_edges(met_map, cohort_ecs, restrict_to_cohort=restrict_to_cohort)
    em.to_csv(EDGES_EM_CSV, index=False)
    print(f"  wrote {EDGES_EM_CSV}")

    # Surface the unmatched metabolites for review / overrides.
    unmatched = met_map.loc[met_map["kegg_compound"] == "", "metabolome_column"].tolist()
    print(f"\n[summary] {len(unmatched)} unmatched metabolome columns "
          f"(add them to {os.path.basename(OVERRIDES_CSV)} to fix):")
    for u in unmatched:
        print("   ", u)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build KEGG metabolite map + EC-bridge edge tables.")
    ap.add_argument("--no-rest", action="store_true",
                    help="Skip the online KEGG REST fallback (offline-only run).")
    ap.add_argument("--all-enzymes", action="store_true",
                    help="Keep ALL KEGG enzymes per compound (not just in-cohort), so "
                         "metabolites connect to each other via shared enzymes.")
    ap.add_argument("--make-overrides-template", action="store_true",
                    help="Also write a curation worksheet (unmatched + low-confidence "
                         "matches with KEGG candidate suggestions) for manual review.")
    args = ap.parse_args()
    main(use_rest=not args.no_rest, restrict_to_cohort=not args.all_enzymes)
    if args.make_overrides_template:
        print("\n=== Curation worksheet ===")
        met_map = pd.read_csv(MAP_CSV).fillna("")
        write_overrides_template(met_map, use_rest=not args.no_rest)
