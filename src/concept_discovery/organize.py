"""Concept organization extension (brief §3).

Method-agnostic post-processing that clusters discovered concepts (from SPLiCE
or SAE explanations) by RadLex text-embedding cosine similarity, annotates each
cluster with a best-effort RadLex anatomical ancestor, and re-expresses
per-image explanations as concept families with a redundancy metric.

Three layers:
  - Adapters (from_spliece_explanations / from_sae_explanations) -> ConceptSet
  - Core (cluster_concepts / annotate_radlex / build_structured_explanations /
    compute_metrics / run) operates only on ConceptSet
  - scripts/run_concept_organization.py drives end-to-end

Standalone output: does NOT feed the LLM judge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src/")

import config  # noqa: E402
import torch  # noqa: E402


@dataclass
class ImageConcepts:
    """Per-image concept activations (normalized)."""
    image_id: str
    activations: dict[str, float]  # name -> activation (>0); subset of ConceptSet.names


@dataclass
class ConceptSet:
    """Method-agnostic concept collection produced by adapters.

    Invariants:
      - embeddings.shape[0] == len(names) == len(name_to_idx)
      - name_to_idx[name] == index of name in names
      - every key in every per_image[].activations is in names
    """
    names: list[str]
    embeddings: torch.Tensor               # (M, 512)
    name_to_idx: dict[str, int]
    per_image: list[ImageConcepts]


@dataclass
class Cluster:
    """Result of clustering concept names."""
    cluster_id: int
    members: list[str]                     # sorted for stable ids
    medoid: str                             # member with max mean cosine to others


@dataclass
class AnnotatedCluster(Cluster):
    """Cluster + best-effort RadLex ancestor."""
    radlex_label: str | None = None
    radlex_rid: str | None = None
    n_resolved: int = 0                    # members that resolved to a RadLex RID
    n_members: int = 0

    @property
    def display_label(self) -> str:
        return self.radlex_label if self.radlex_label else self.medoid


def _build_term_to_idx(vocab_terms: list[dict], vocab_emb: torch.Tensor) -> dict[str, int]:
    """term -> vocab row index, with a count guard (mirrors spliece F-007)."""
    if len(vocab_terms) != vocab_emb.shape[0]:
        raise ValueError(
            f"vocab_terms ({len(vocab_terms)}) != vocab_emb rows ({vocab_emb.shape[0]}); "
            "coefficient index would not map to the correct term — re-embed the vocabulary."
        )
    return {t["term"]: i for i, t in enumerate(vocab_terms) if isinstance(t, dict) and "term" in t}


def _finalize_concept_set(
    per_image: list[ImageConcepts], term_to_idx: dict[str, int], vocab_emb: torch.Tensor
) -> ConceptSet:
    """Collect active names, slice their embeddings, build the ConceptSet.

    Names are sorted for deterministic cluster ids. Names absent from the
    vocabulary have already been skipped by the caller (per-image activations
    only contain resolvable names).
    """
    active = sorted({name for img in per_image for name in img.activations})
    name_to_idx = {n: i for i, n in enumerate(active)}
    if active:
        rows = torch.stack([vocab_emb[term_to_idx[n]] for n in active])
    else:
        rows = torch.empty((0, vocab_emb.shape[1]), dtype=vocab_emb.dtype)
    return ConceptSet(names=active, embeddings=rows, name_to_idx=name_to_idx, per_image=per_image)


def from_spliece_explanations(
    explanations: list[dict],
    vocab_terms: list[dict],
    vocab_emb: torch.Tensor,
) -> ConceptSet:
    """Normalize SPLiCE sample_explanations.json into a ConceptSet.

    SPLiCE feature_id == vocab index; name == vocab term directly. Drops
    non-positive activations and names absent from the vocabulary (graceful,
    no crash). Missing image_id falls back to f"img_{i}".
    """
    term_to_idx = _build_term_to_idx(vocab_terms, vocab_emb)
    per_image: list[ImageConcepts] = []
    for i, img in enumerate(explanations):
        activations: dict[str, float] = {}
        for c in img.get("top_k_concepts", []):
            name = c.get("name")
            act = float(c.get("activation", 0.0))
            if act > 0 and name in term_to_idx:
                # keep the max activation if a name repeats
                activations[name] = max(activations.get(name, 0.0), act)
        image_id = img.get("image_id") or f"img_{i}"
        per_image.append(ImageConcepts(image_id=image_id, activations=activations))
    return _finalize_concept_set(per_image, term_to_idx, vocab_emb)


def from_sae_explanations(
    explanations: list[dict],
    concept_names: dict,
    vocab_terms: list[dict],
    vocab_emb: torch.Tensor,
) -> ConceptSet:
    """Normalize SAE sample_explanations.json (+ concept_names.json) into a ConceptSet.

    SAE feature_id == SAE feature index; name is the assigned vocab term. Excludes
    features flagged is_dead (or named 'DEAD_FEATURE') in concept_names. concept_names
    keys are strings; explanations feature_id are ints -> coerce to str. Drops names
    absent from the vocabulary (graceful).
    """
    term_to_idx = _build_term_to_idx(vocab_terms, vocab_emb)
    dead_fids: set[str] = set()
    for fid, info in (concept_names or {}).items():
        if not isinstance(info, dict):
            continue
        if info.get("is_dead") or info.get("name") == "DEAD_FEATURE":
            dead_fids.add(str(fid))

    per_image: list[ImageConcepts] = []
    for i, img in enumerate(explanations):
        activations: dict[str, float] = {}
        for c in img.get("top_k_concepts", []):
            fid = str(c.get("feature_id"))
            if fid in dead_fids:
                continue
            name = c.get("name")
            act = float(c.get("activation", 0.0))
            if act > 0 and name in term_to_idx:
                activations[name] = max(activations.get(name, 0.0), act)
        image_id = img.get("image_id") or f"img_{i}"
        per_image.append(ImageConcepts(image_id=image_id, activations=activations))
    return _finalize_concept_set(per_image, term_to_idx, vocab_emb)


def _medoid(members: list[str], embeddings: torch.Tensor, name_to_idx: dict[str, int]) -> str:
    """Member whose embedding has max mean cosine similarity to the others."""
    if len(members) == 1:
        return members[0]
    idx = [name_to_idx[m] for m in members]
    rows = embeddings[idx]                              # (k, D)
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = rows @ rows.T                                 # (k, k)
    mean_sim = sim.sum(dim=1) / (len(members) - 1)      # diagonal (self) is constant across rows -> argmax unaffected
    return members[int(mean_sim.argmax())]


def cluster_concepts(
    concept_set: ConceptSet,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    linkage: str = "average",
) -> list[Cluster]:
    """Agglomerative cosine clustering of concept-name embeddings.

    Deterministic: cluster ids are assigned after sorting clusters by their
    sorted member tuples. Stopping rule: n_clusters OR distance_threshold
    (compute_full_tree=True); if neither given, n_clusters = max(2, round(sqrt(M))).
    No post-hoc merging.
    """
    from sklearn.cluster import AgglomerativeClustering

    names = concept_set.names
    M = len(names)
    if M == 0:
        return []
    if M == 1:
        return [Cluster(cluster_id=0, members=list(names), medoid=names[0])]

    if n_clusters is None and distance_threshold is None:
        n_clusters = max(2, round(M ** 0.5))

    kwargs: dict = dict(metric="cosine", linkage=linkage)
    if distance_threshold is not None:
        kwargs.update(n_clusters=None, distance_threshold=distance_threshold, compute_full_tree=True)
    else:
        kwargs.update(n_clusters=min(n_clusters, M))

    labels = AgglomerativeClustering(**kwargs).fit_predict(concept_set.embeddings.numpy())

    groups: dict[int, list[str]] = {}
    for name, lbl in zip(names, labels):
        groups.setdefault(int(lbl), []).append(name)
    ordered = sorted(groups.values(), key=lambda members: sorted(members))
    clusters: list[Cluster] = []
    for cid, members in enumerate(ordered):
        members_sorted = sorted(members)
        clusters.append(Cluster(
            cluster_id=cid,
            members=members_sorted,
            medoid=_medoid(members_sorted, concept_set.embeddings, concept_set.name_to_idx),
        ))
    return clusters


def ancestor_rids(graph, rid: str, seen: set[str] | None = None) -> set[str]:
    """Cycle-safe set of {rid} ∪ all ancestor RIDs via graph.child_to_parents."""
    if seen is None:
        seen = set()
    if rid in seen:
        return set()
    seen.add(rid)
    acc = {rid}
    for parent in graph.child_to_parents.get(rid, []):
        acc |= ancestor_rids(graph, parent, seen)
    return acc


def _resolve_member_rid(graph, name: str) -> str | None:
    """Deterministically resolve a vocab term to one RadLex RID (or None).

    graph.label_to_rids maps lowercased preferred label -> [RID, ...]; pick the
    lexicographically smallest RID for determinism.
    """
    rids = graph.label_to_rids.get(name.lower())
    if not rids:
        return None
    return min(rids)


def _build_children_map(graph) -> dict[str, list[str]]:
    """Invert graph.child_to_parents (child->parents) into parent->children."""
    children: dict[str, list[str]] = {}
    for child, parents in graph.child_to_parents.items():
        for p in parents:
            children.setdefault(p, []).append(child)
    return children


def _descendant_count(rid: str, children: dict[str, list[str]], memo: dict[str, int]) -> int:
    """Number of RIDs reachable from ``rid`` via child edges (inclusive of rid).

    Cycle-safe BFS (``seen`` set) over the RadLex DAG, memoized per annotate_radlex call.
    """
    if rid in memo:
        return memo[rid]
    from collections import deque
    seen = {rid}
    dq = deque([rid])
    n = 0
    while dq:
        x = dq.popleft()
        n += 1
        for c in children.get(x, []):
            if c not in seen:
                seen.add(c)
                dq.append(c)
    memo[rid] = n
    return n


def annotate_radlex(
    clusters: list[Cluster],
    graph,
) -> list[AnnotatedCluster]:
    """Annotate each cluster with a best-effort RadLex ancestor.

    Two degeneracy guards (spec §6.2, amended):
      1. Leaf-root rejection: candidate RIDs with no parents are rejected.
      2. Canopy rejection (ontology-intrinsic): candidate RIDs whose descendant
         count exceeds 1% of the graph are rejected as trivially generic (e.g.
         'RadLex entity', 'anatomical entity', 'pathophysiologischer Befund').
         Only active when the graph has >= 1000 RIDs (so tiny synthetic graphs
         used in unit tests are unaffected — their canopy would be all-or-nothing
         under a 1% threshold).

    Candidate selection: an ancestor must be supported by >= ceil(0.5 * n_resolved)
    members (and >= 2 when n_resolved >= 2, so a member's own RID alone never
    qualifies unless shared). Among surviving candidates pick the most specific
    (deepest in the candidate sub-DAG); tie-break by support desc, then shortest
    label. None -> cluster label falls back to the medoid.
    Never raises on unresolved members or empty clusters.
    """
    import math

    total_rids = len(graph.rid_to_label)
    use_canopy = total_rids >= 1000
    canopy_max = 0.01 * total_rids if use_canopy else float("inf")
    children = _build_children_map(graph) if use_canopy else {}
    desc_memo: dict[str, int] = {}
    _anc_memo: dict[str, set[str]] = {}

    def _anc(rid: str) -> set[str]:
        if rid not in _anc_memo:
            _anc_memo[rid] = ancestor_rids(graph, rid)
        return _anc_memo[rid]

    out: list[AnnotatedCluster] = []
    for c in clusters:
        resolved: list[tuple[str, set[str]]] = []
        for m in c.members:
            rid = _resolve_member_rid(graph, m)
            if rid is not None:
                resolved.append((rid, _anc(rid)))
        n_resolved = len(resolved)
        n_members = len(c.members)

        radlex_rid: str | None = None
        radlex_label: str | None = None
        if n_resolved > 0:
            threshold = max(2, math.ceil(0.5 * n_resolved)) if n_resolved >= 2 else 1
            support: dict[str, int] = {}
            for _, anc_set in resolved:
                for r in anc_set:
                    support[r] = support.get(r, 0) + 1
            candidates = [
                r for r, cnt in support.items()
                if cnt >= threshold
                and r in graph.child_to_parents                     # not a leaf-root
                and (not use_canopy or _descendant_count(r, children, desc_memo) <= canopy_max)
            ]
            if candidates:
                cand_set = set(candidates)
                anc_of = {r: _anc(r) for r in candidates}

                def sort_key(r: str):
                    specificity = len(anc_of[r] & cand_set)
                    label = graph.rid_to_label.get(r, r)
                    return (specificity, support[r], -len(label))

                best = max(candidates, key=sort_key)
                radlex_rid = best
                radlex_label = graph.rid_to_label.get(best)

        out.append(AnnotatedCluster(
            cluster_id=c.cluster_id,
            members=c.members,
            medoid=c.medoid,
            radlex_label=radlex_label,
            radlex_rid=radlex_rid,
            n_resolved=n_resolved,
            n_members=n_members,
        ))
    return out


def build_structured_explanations(
    concept_set: ConceptSet, clusters: list[AnnotatedCluster]
) -> list[dict]:
    """Re-express per-image explanations as concept families + redundancy score.

    Each family = the subset of an image's active concepts that fall in one cluster.
    redundancy_score (image) = n_raw_concepts / n_distinct_families (>=1, or 0 if empty).
    """
    name_to_cluster: dict[str, AnnotatedCluster] = {}
    for c in clusters:
        for m in c.members:
            name_to_cluster[m] = c

    out: list[dict] = []
    for img in concept_set.per_image:
        fam_acc: dict[int, dict] = {}
        for name, act in img.activations.items():
            c = name_to_cluster.get(name)
            if c is None:
                continue  # name not in any cluster (shouldn't happen post-adapter)
            fam = fam_acc.setdefault(c.cluster_id, {
                "cluster_id": c.cluster_id,
                "label": c.display_label,
                "radlex_label": c.radlex_label,
                "concepts": [],
                "aggregate_activation": 0.0,
            })
            fam["concepts"].append({"name": name, "activation": act})
            fam["aggregate_activation"] += act

        families = []
        for fam in fam_acc.values():
            fam["intra_redundancy"] = len(fam["concepts"])
            families.append(fam)
        families.sort(key=lambda f: f["cluster_id"])

        n_raw = len(img.activations)
        n_fam = len(families)
        redundancy = (n_raw / n_fam) if n_fam > 0 else 0
        out.append({
            "image_id": img.image_id,
            "families": families,
            "redundancy_score": redundancy,
        })
    return out


def compute_metrics(
    concept_set: ConceptSet,
    clusters: list[AnnotatedCluster],
    structured: list[dict],
) -> dict:
    """Quantify organization: cluster sizes, silhouette, redundancy reduction,
    RadLex coverage, empty-image count. Silhouette is None when < 2 clusters or
    a single non-empty cluster (sklearn requirement).
    """
    import numpy as np
    from sklearn.metrics import silhouette_score

    n_active = len(concept_set.names)
    n_clusters = len(clusters)
    sizes = [len(c.members) for c in clusters]
    mean_size = (sum(sizes) / n_clusters) if n_clusters else 0.0

    silhouette = None
    if n_active >= 2 and n_clusters >= 2:
        labels = [-1] * n_active
        for c in clusters:
            for m in c.members:
                labels[concept_set.name_to_idx[m]] = c.cluster_id
        if len(set(labels)) >= 2:
            try:
                silhouette = float(silhouette_score(
                    concept_set.embeddings.numpy(), np.array(labels), metric="cosine"
                ))
            except Exception:
                silhouette = None

    raw_counts = [len(img.activations) for img in concept_set.per_image]
    fam_counts = [len(ex["families"]) for ex in structured]
    mean_raw = (sum(raw_counts) / len(raw_counts)) if raw_counts else 0.0
    mean_fam = (sum(fam_counts) / len(fam_counts)) if fam_counts else 0.0
    redundancy_reduction = (mean_raw / mean_fam) if mean_fam > 0 else 0.0

    resolved = sum(c.n_resolved for c in clusters)
    members_total = sum(c.n_members for c in clusters)
    coverage = (resolved / members_total * 100.0) if members_total else 0.0

    n_empty = sum(1 for img in concept_set.per_image if not img.activations)

    return {
        "n_concepts_active": n_active,
        "n_clusters": n_clusters,
        "mean_cluster_size": mean_size,
        "silhouette_cosine": silhouette,
        "redundancy_reduction": redundancy_reduction,
        "mean_raw_concepts_per_image": mean_raw,
        "mean_families_per_image": mean_fam,
        "radlex_coverage_pct": coverage,
        "n_empty_images": n_empty,
    }


def _clusters_to_dicts(clusters: list[AnnotatedCluster]) -> list[dict]:
    return [{
        "cluster_id": c.cluster_id,
        "label": c.display_label,
        "radlex_label": c.radlex_label,
        "radlex_rid": c.radlex_rid,
        "medoid": c.medoid,
        "members": c.members,
        "size": len(c.members),
        "n_resolved": c.n_resolved,
    } for c in clusters]


def run(
    cfg,
    concept_set: ConceptSet,
    graph=None,
) -> dict:
    """Orchestrate cluster -> annotate -> structured -> metrics; write outputs.

    If graph is None, RadLex annotation is skipped (all radlex_label None).
    Returns the metrics dict.
    """
    import json

    raw_clusters = cluster_concepts(
        concept_set,
        n_clusters=cfg.n_clusters,
        distance_threshold=cfg.distance_threshold,
        linkage=cfg.linkage,
    )
    if graph is not None:
        annotated = annotate_radlex(raw_clusters, graph)
    else:
        annotated = [AnnotatedCluster(
            cluster_id=c.cluster_id, members=c.members, medoid=c.medoid,
            radlex_label=None, radlex_rid=None, n_resolved=0, n_members=len(c.members),
        ) for c in raw_clusters]

    structured = build_structured_explanations(concept_set, annotated)
    metrics = compute_metrics(concept_set, annotated, structured)

    # I-1: surface active concept names that did not resolve to a RadLex RID
    # (diagnostic for vocab/RadLex drift). Only meaningful when a graph is present.
    if graph is not None:
        unresolved = sorted(
            n for n in concept_set.names if _resolve_member_rid(graph, n) is None
        )
        metrics["unresolved_terms"] = unresolved[:20]
        metrics["n_unresolved_terms"] = len(unresolved)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "concept_clusters.json").write_text(
        json.dumps(_clusters_to_dicts(annotated), indent=2)
    )
    (cfg.output_dir / "structured_explanations.json").write_text(
        json.dumps(structured, indent=2)
    )
    (cfg.output_dir / "organization_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


if __name__ == "__main__":
    """Self-check on synthetic data: never clobbers production output (writes /tmp)."""
    import json
    from dataclasses import replace

    print("🔍 organize self-check running...")
    V = 30
    torch.manual_seed(0)
    vocab_emb = torch.eye(V, 512)
    vocab_terms = [{"term": f"t{i}"} for i in range(V)]
    explanations = [{
        "image_id": f"img_{i}",
        "top_k_concepts": [
            {"feature_id": (2 * i) % V, "name": f"t{(2*i)%V}", "activation": 1.0},
            {"feature_id": (2 * i + 1) % V, "name": f"t{(2*i+1)%V}", "activation": 0.5},
        ],
    } for i in range(8)]

    cs = from_spliece_explanations(explanations, vocab_terms, vocab_emb)
    print(f"   active concepts: {len(cs.names)} across {len(cs.per_image)} images")
    assert len(cs.names) > 0

    selfcheck_cfg = replace(
        config.organize, n_clusters=3,
        output_dir=Path("/tmp/organize_selfcheck"),
        radlex_csv_path=Path("/tmp/nonexistent_radlex.csv"),
    )
    metrics = run(selfcheck_cfg, cs, graph=None)
    print(f"   clusters: {metrics['n_clusters']}  redundancy_reduction: {metrics['redundancy_reduction']:.2f}")
    assert metrics["n_clusters"] >= 1
    print(f"✅ Self-check passed. Output: /tmp/organize_selfcheck/")
