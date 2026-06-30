"""radlex_support.py — RadLex ontology loading + domain filtering for vocab building.

Parses the RadLex CSV into its is-a DAG and filters preferred labels to a target
imaging domain (chest X-ray by default) via hierarchical traversal:

  keep a term if one of its RIDs
    - is NOT obsolete, AND
    - has no EXCLUDE keyword in any ancestor / anatomical-site label, AND
    - descends from a TARGET root (thoracic anatomy), possibly via its anatomical
      site, OR descends from a FINDING root (clinical findings: pneumothorax,
      consolidation, ...) OR a DEVICE root (tubes/catheters).

The domain-specific knobs (roots + exclude keywords) live in :class:`RadLexFilterConfig`
so a different dataset / imaging domain can supply its own without editing this
module. :func:`load_and_filter_radlex` is the high-level entry point used by
``build_vocabulary.py`` (keeps its import surface to a single name).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


# ============================================================================
# Domain configuration (chest X-ray default; override for other datasets)
# ============================================================================

@dataclass(frozen=True)
class RadLexFilterConfig:
    """Domain-specific knobs for RadLex hierarchical filtering.

    Defaults are calibrated for chest X-ray (IU X-Ray) against the committed
    ``data/radlex.csv``. To support a different imaging domain (e.g. abdominal
    CT), build a custom instance with that domain's anatomy/finding/device roots
    and pass it to :func:`load_and_filter_radlex` / :func:`filter_terms`.
    """

    # Human label for logs/reports.
    domain: str = "chest X-ray"

    # Thoracic anatomy roots
    target_roots: frozenset[str] = frozenset({
        "RID1243",   # thorax / chest
        "RID1301",   # lung / Lunge
        "RID1385",   # heart / Herz
        "RID1362",   # Pleura
        "RID1363",   # Pleuraraum
        "RID1384",   # mediastinum
        "RID1524",   # diaphragm
        "RID15103",  # bony thorax
        "RID1463",   # lymph node of thorax
        "RID49962",  # chest vessels
        "RID49961",  # chest veins
        "RID50696",  # lung imaging observation
    })

    # Clinical-finding branch (pneumothorax, consolidation, granuloma, mass,
    # nodule, ...) NOT under thoracic anatomy in RadLex.
    finding_roots: frozenset[str] = frozenset({"RID34785"})  # klinischer Befund

    # Support-device branch (endotracheal tube, central venous catheter, ...).
    device_roots: frozenset[str] = frozenset({"RID5554"})    # tube or catheter

    # Substring excludes i.e. organs / systems / diseases of other body regions.
    exclude_keywords: frozenset[str] = frozenset({
        "abdomen", "abdominal", "pelvis", "pelvic", "pelvi", "brain", "cerebral",
        "cerebro", "cranial", "prostate", "prostatic", "breast", "mammo", "ovary",
        "ovarian", "uterus", "uterine", "renal", "kidney", "nephro", "pancrea",
        "pancreas", "pancreatic", "liver", "hepatic", "hepato", "hepatobiliary",
        "biliary", "cholangio", "spleen", "splenic", "gastric", "stomach", "colon",
        "duodenum", "bowel", "intestinal", "skull", "mandible", "lower extremity",
        "upper extremity", "leg", "foot", "ankle", "femur", "tibia", "fibula",
        "arm", "hand", "elbow", "hip", "bladder", "thyroid",
        "gliom", "rhabdomyo", "ureter", "myelom", "leuk",
        "cervical spine", "lumbar spine", "lumbosacral", "coccyx", "sacrum",
        "bi-rads", "birads", "pi-rads", "pirads", "li-rads", "lirads", "c-rads", "crads",
    })


# Default config for the current project (IU X-Ray chest radiographs).
CXR_FILTER_CONFIG = RadLexFilterConfig()


# ============================================================================
# RadLex DAG loading
# ============================================================================

# RadLex CSV column holding the anatomical-site annotation (URI-style header).
_ANATOMICAL_SITE_COL = "http://www.radlex.org/RID/Anatomical_Site"


@dataclass
class RadLexGraph:
    """Parsed RadLex DAG + label index.

    Built once by the function `load_radlex_graph` and reused to classify many terms.
    """

    # RID -> list of parent RIDs (the is-a / superclass edges).
    child_to_parents: dict[str, list[str]] = field(default_factory=dict)
    # RID -> preferred label.
    rid_to_label: dict[str, str] = field(default_factory=dict)
    # lowercased preferred label -> RIDs that share it (1-to-N possible).
    label_to_rids: dict[str, list[str]] = field(default_factory=dict)
    # RID -> anatomical-site RIDs (from the Anatomical_Site column).
    rid_to_sites: dict[str, list[str]] = field(default_factory=dict)
    # RIDs flagged Obsolete=TRUE.
    obsolete_rids: set[str] = field(default_factory=set)


def extract_rid(uri: str) -> str:
    """Coerce a RadLex Class ID / parent / site URI to a bare ``RIDxxxx`` token.

    Handles both ``http://www.radlex.org/RID/RID1234`` and ``RID:1234`` styles,
    plus already-bare tokens.
    """
    if not uri:
        return ""
    if "/" in uri:
        return uri.split("/")[-1].strip()
    if ":" in uri:
        return uri.split(":")[-1].strip()
    return uri.strip()


def load_radlex_graph(csv_path) -> RadLexGraph:
    """Parse the RadLex CSV into an instance of  Class `RadLexGraph` (is-a DAG + label index).

    Args:
        csv_path: Path to the RadLex CSV (Class ID / Preferred Label / Obsolete /
            Parents / Anatomical_Site columns).

    Returns:
        A populated instance of `RadLexGraph` Class.
    """
    graph = RadLexGraph()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = extract_rid(row.get("Class ID", ""))
            if not rid:
                continue
            if (row.get("Obsolete") or "").strip().upper() == "TRUE":
                graph.obsolete_rids.add(rid)
            label = (row.get("Preferred Label") or "").strip()
            graph.rid_to_label[rid] = label
            if label:
                graph.label_to_rids.setdefault(label.lower(), []).append(rid)
            parents = [extract_rid(p) for p in (row.get("Parents") or "").split("|") if p.strip()]
            if parents:
                graph.child_to_parents[rid] = parents
            sites = [extract_rid(s) for s in (row.get(_ANATOMICAL_SITE_COL) or "").split("|") if s.strip()]
            if sites:
                graph.rid_to_sites[rid] = sites
    return graph


# ============================================================================
# Classification (memoized)
# ============================================================================

class _RadLexClassifier:
    """Classify RIDs against a `RadLexFilterConfig` over a RadLexGraph`"""

    def __init__(self, graph: RadLexGraph, config: RadLexFilterConfig):
        self.graph = graph
        self.config = config
        self._ancestor_memo: dict[str, set[str]] = {}
        # One descendant memo per root-set (keyed by the frozenset identity).
        self._descendant_memos: dict[frozenset, dict[str, bool]] = {
            config.target_roots: {},
            config.finding_roots: {},
            config.device_roots: {},
        }

    def ancestor_labels(self, rid: str, seen: set[str] | None = None) -> set[str]:
        """Lowercased labels of ``rid`` and all its ancestors (cycle-safe)."""
        if rid in self._ancestor_memo:
            return self._ancestor_memo[rid]
        if seen is None:
            seen = set()
        if rid in seen:          # cycle guard (RadLex should be acyclic; defensive)
            return set()
        seen.add(rid)
        labels: set[str] = set()
        label = self.graph.rid_to_label.get(rid, "")
        if label:
            labels.add(label.lower())
        for parent in self.graph.child_to_parents.get(rid, []):
            labels |= self.ancestor_labels(parent, seen)
        self._ancestor_memo[rid] = labels
        return labels

    def _descends_from(self, rid: str, roots: frozenset[str], memo: dict[str, bool]) -> bool:
        """True if ``rid`` reaches any of ``roots`` by walking parents upward."""
        if rid in roots:
            return True
        if rid not in self.graph.child_to_parents:
            return False
        if rid in memo:
            return memo[rid]
        memo[rid] = False  # provisional (also breaks accidental cycles)
        for parent in self.graph.child_to_parents[rid]:
            if self._descends_from(parent, roots, memo):
                memo[rid] = True
                return True
        return False

    def keep(self, rid: str) -> bool:
        """Should ``rid`` be kept for the target domain?"""
        if rid in self.graph.obsolete_rids:
            return False

        # Exclude if any keyword is a substring of any ancestor / site label.
        related = self.ancestor_labels(rid)
        for site in self.graph.rid_to_sites.get(rid, []):
            related |= self.ancestor_labels(site)
        for kw in self.config.exclude_keywords:
            if any(kw in lbl for lbl in related):
                return False

        cfg = self.config
        # Thoracic anatomy, via self or anatomical site.
        if self._descends_from(rid, cfg.target_roots, self._descendant_memos[cfg.target_roots]):
            return True
        for site in self.graph.rid_to_sites.get(rid, []):
            if self._descends_from(site, cfg.target_roots, self._descendant_memos[cfg.target_roots]):
                return True
        # Non-anatomy CXR branches: clinical findings + support devices.
        if self._descends_from(rid, cfg.finding_roots, self._descendant_memos[cfg.finding_roots]):
            return True
        if self._descends_from(rid, cfg.device_roots, self._descendant_memos[cfg.device_roots]):
            return True
        return False


# ============================================================================
# High-level API
# ============================================================================

def preferred_labels(graph: RadLexGraph) -> list[str]:
    """Deduplicated, non-obsolete RadLex preferred labels (order-preserving).

    Iterates RIDs in CSV row order, skipping obsolete RIDs and empty / single-
    character labels, and deduplicates case-insensitively. Equivalent to the
    former ``utils.load_radlex_terms`` but derived from an already-loaded graph
    (no second CSV read).

    Args:
        graph: A RadLex graph from :func:`load_radlex_graph`.

    Returns:
        Ordered list of original-case preferred labels.
    """
    seen: set[str] = set()
    labels: list[str] = []
    for rid, label in graph.rid_to_label.items():  # insertion order == CSV row order
        if not label or rid in graph.obsolete_rids:
            continue
        low = label.lower()
        if low in seen or len(label) <= 1:
            continue
        seen.add(low)
        labels.append(label)
    return labels


def filter_terms(
    graph: RadLexGraph,
    terms: Sequence[str],
    config: RadLexFilterConfig = CXR_FILTER_CONFIG,
) -> list[str]:
    """Filter ``terms`` to the target domain, preserving input order.

    Terms not found in the RadLex graph (e.g. manually-injected NIH seeds) are
    kept by default (safety-net), matching the prior behavior.

    Args:
        graph: A RadLex graph.
        terms: Preferred labels to classify (order is preserved).
        config: Domain config (defaults to chest X-ray).

    Returns:
        Subset of ``terms`` deemed relevant to the target domain.
    """
    clf = _RadLexClassifier(graph, config)
    filtered: list[str] = []
    for term in terms:
        rids = graph.label_to_rids.get(term.lower())
        if not rids:
            filtered.append(term)  # not a RadLex preferred label -> keep (safety-net)
            continue
        if any(clf.keep(rid) for rid in rids):
            filtered.append(term)
    return filtered


def load_and_filter_radlex(
    csv_path,
    *,
    config: RadLexFilterConfig = CXR_FILTER_CONFIG,
) -> list[str]:
    """Load the RadLex CSV, derive its preferred labels, and filter them to the target domain.

    One-call wrapper over `load_radlex_graph` + `preferred_labels` + `filter_terms`;
    this is the single function ``build_vocabulary.py`` imports. Reads the CSV exactly
    once and filters all of its (non-obsolete, deduplicated) preferred labels.

    Args:
        csv_path: Path to the RadLex CSV.
        config: Domain config (keyword-only; defaults to chest X-ray).

    Returns:
        Subset of the CSV's preferred labels deemed relevant to the target domain.
    """
    graph = load_radlex_graph(csv_path)
    terms = preferred_labels(graph)
    filtered = filter_terms(graph, terms, config)
    print(
        f"RadLex hierarchical filtering: {len(terms)} terms -> "
        f"{len(filtered)} {config.domain}-relevant terms."
    )
    return filtered
