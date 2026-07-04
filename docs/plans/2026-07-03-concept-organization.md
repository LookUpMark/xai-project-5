# Concept Organization Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a method-agnostic, dataset-portable post-processing stage that clusters discovered concepts (SPLiCE + SAE) by RadLex text-embedding similarity, annotates each cluster with a best-effort RadLex ancestor, and re-expresses per-image explanations as concept families with redundancy metrics.

**Architecture:** Three layers — (1) adapters normalize SPLiCE/SAE outputs into a unified `ConceptSet`, (2) a method-agnostic core (`cluster_concepts`, `annotate_radlex`, `build_structured_explanations`, `compute_metrics`, `run`) operates only on `ConceptSet`, (3) a `scripts/run_concept_organization.py` driver mirrors the existing `run_spliece.py` pattern.

**Tech Stack:** Python 3.12, PyTorch, scikit-learn (`AgglomerativeClustering`, `silhouette_score`), existing `src/vocabulary_building/radlex_support.py` (`RadLexGraph`, `load_radlex_graph`). Tests via pytest. Run with `PYTHONPATH=src:.`.

**Spec:** `docs/design/proposals/2026-07-03-concept-organization.md`
**Branch:** `feat/concept-organization` (already created)

**Conventions (from `spliece.py` / `test_spliece.py`):**
- Module does `sys.path.insert(0, "src/")` then bare imports (`import config`, `from utils import load_tensor`).
- Tests do `sys.path.insert(0, "src/")` then `from concept_discovery.organize import …`.
- Frozen-config overrides via `dataclasses.replace(config.organize, …)`.
- SAFE tensor load: `utils.load_tensor(path)` (`weights_only=True`).
- All commands assume CWD = repo root and run via `.venv/bin/python` / `.venv/bin/pytest` with `PYTHONPATH=src:.`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/config.py` | Add `OrganizeConfig` frozen dataclass + `organize = OrganizeConfig()` singleton. |
| `src/concept_discovery/organize.py` | Core + adapters + `run()` + `__main__` self-check. Single module, focused units. |
| `scripts/run_concept_organization.py` | CLI driver (`--source`, `--tag`, …), writes `REPORT_organization.md`. |
| `tests/unit/test_organize.py` | Unit tests, one per core unit + adapters (TDD). |
| `tests/integration/test_organize_pipeline.py` | End-to-end `run()` on synthetic data. |
| `results/concept_organization[_{tag}]/` | Output dir (gitignored; produced by run). |

---

## Task 1: Add `OrganizeConfig` to `config.py`

**Files:**
- Modify: `src/config.py` (insert class after `SpliCEConfig`, before the "Instantiate configs" block ~line 471; add singleton in the block ~line 484).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_organize_config.py`:

```python
"""Config smoke test for OrganizeConfig."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, "src/")

import config
from config import OrganizeConfig


def test_organize_singleton_exists():
    assert isinstance(config.organize, OrganizeConfig)


def test_defaults_anchored_to_paths():
    assert config.organize.vocab_path == config.paths.vocab_labels_path
    assert config.organize.vocab_emb_path == config.paths.vocab_embeddings_path
    assert config.organize.radlex_csv_path == config.paths.data_dir / "radlex.csv"
    assert config.organize.output_dir == config.paths.results_dir / "concept_organization"


def test_n_clusters_and_distance_mutually_exclusive():
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        OrganizeConfig(n_clusters=5, distance_threshold=0.5)


def test_invalid_linkage_rejected():
    import pytest
    with pytest.raises(ValueError, match="linkage"):
        OrganizeConfig(linkage="ward")  # ward incompatible with cosine


def test_invalid_metric_rejected():
    import pytest
    with pytest.raises(ValueError, match="metric"):
        OrganizeConfig(metric="euclidean")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'OrganizeConfig'`.

- [ ] **Step 3: Write minimal implementation**

In `src/config.py`, insert immediately after the `SpliCEConfig` class definition (after its closing line, before the `# ── Instantiate configs ──` comment):

```python
@dataclass(frozen=True)
class OrganizeConfig:
    """Configuration for the concept-organization extension (brief §3).

    Clusters discovered concepts (SPLiCE or SAE) by cosine similarity of their
    RadLex text embeddings, annotates each cluster with a best-effort RadLex
    anatomical ancestor, and re-expresses per-image explanations as concept
    families with a redundancy metric. Method-agnostic and dataset-portable.

    Args:
        n_clusters: Target number of clusters. If None and distance_threshold is
            None, defaults to max(2, round(sqrt(M))) at runtime.
        distance_threshold: Agglomerative linkage-distance cut. Mutually exclusive
            with n_clusters (enforced in __post_init__).
        linkage: Agglomerative linkage criterion ('average' | 'complete' | 'single').
        metric: Distance metric; 'cosine' is the only supported value for this design.
        radlex_csv_path: Path to the RadLex ontology CSV (for ancestor annotation).
        vocab_path: Path to vocabulary.json (list of {"term": str, ...} dicts).
        vocab_emb_path: Path to text_vocab_embeddings.pt (V, 512).
        output_dir: Directory for output artifacts.
    """

    n_clusters: Optional[int] = None
    distance_threshold: Optional[float] = None
    linkage: str = "average"
    metric: str = "cosine"
    radlex_csv_path: Path = field(default_factory=lambda: paths.data_dir / "radlex.csv")
    vocab_path: Path = field(default_factory=lambda: paths.vocab_labels_path)
    vocab_emb_path: Path = field(default_factory=lambda: paths.vocab_embeddings_path)
    output_dir: Path = field(default_factory=lambda: paths.results_dir / "concept_organization")

    def __post_init__(self):
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise ValueError(
                "n_clusters and distance_threshold are mutually exclusive in OrganizeConfig"
            )
        if self.linkage not in {"average", "complete", "single"}:
            raise ValueError(
                f"OrganizeConfig.linkage must be average|complete|single, got {self.linkage}"
            )
        if self.metric != "cosine":
            raise ValueError(
                f"OrganizeConfig.metric must be 'cosine', got {self.metric}"
            )
```

Then in the singleton instantiation block (after `spliece = SpliCEConfig()`), add:

```python
organize = OrganizeConfig()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize_config.py -q`
Expected: PASS (5 tests).

Also run the full unit suite to confirm no regression:
Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit -q`
Expected: PASS (no new failures vs baseline — baseline has 1 pre-existing unrelated failure in `test_extract_embeddings`).

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/unit/test_organize_config.py
git commit -m "feat(config): add OrganizeConfig + singleton for concept-organization extension"
```

---

## Task 2: Scaffold `organize.py` dataclasses

**Files:**
- Create: `src/concept_discovery/organize.py`
- Test: `tests/unit/test_organize.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_organize.py`:

```python
"""Unit tests for concept_discovery.organize."""
from __future__ import annotations

import sys
sys.path.insert(0, "src/")

import torch
from concept_discovery.organize import ConceptSet, ImageConcepts


class TestDataclasses:
    def test_concept_set_constructs(self):
        cs = ConceptSet(
            names=["a", "b"],
            embeddings=torch.randn(2, 512),
            name_to_idx={"a": 0, "b": 1},
            per_image=[ImageConcepts(image_id="x", activations={"a": 1.0})],
        )
        assert cs.names == ["a", "b"]
        assert cs.embeddings.shape == (2, 512)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'concept_discovery.organize'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/concept_discovery/organize.py`:

```python
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
from dataclasses import dataclass, field

sys.path.insert(0, "src/")

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): scaffold ConceptSet + Cluster dataclasses"
```

---

## Task 3: SPLiCE adapter `from_spliece_explanations`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append function)
- Test: `tests/unit/test_organize.py` (append class)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import from_spliece_explanations


class TestSpliCEAdapter:
    def _vocab(self):
        terms = [{"term": f"term_{i}"} for i in range(10)]
        emb = torch.eye(10, 512)  # 10 distinct atoms
        return terms, emb

    def test_builds_concept_set_from_explanations(self):
        terms, emb = self._vocab()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 0.5},
                {"feature_id": 7, "name": "term_7", "activation": 0.2},
            ],
            "pseudo_report": "x",
        }]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert set(cs.names) == {"term_3", "term_7"}
        assert cs.embeddings.shape == (2, 512)
        # rows match the vocab rows of the named terms
        assert torch.equal(cs.embeddings[cs.name_to_idx["term_3"]], emb[3])
        assert cs.per_image[0].image_id == "img0"
        assert cs.per_image[0].activations == {"term_3": 0.5, "term_7": 0.2}

    def test_drops_nonpositive_and_unresolved_names(self):
        terms, emb = self._vocab()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 1, "name": "term_1", "activation": 0.0},   # dropped (<=0)
                {"feature_id": 2, "name": "term_2", "activation": -1.0},  # dropped (<0)
                {"feature_id": 9, "name": "ghost", "activation": 0.9},    # dropped (not in vocab)
                {"feature_id": 4, "name": "term_4", "activation": 0.4},
            ],
        }]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert cs.names == ["term_4"]
        assert cs.per_image[0].activations == {"term_4": 0.4}

    def test_vocab_emb_count_mismatch_raises(self):
        terms = [{"term": f"t{i}"} for i in range(5)]
        emb = torch.randn(3, 512)  # mismatch
        import pytest
        with pytest.raises(ValueError, match="vocab_emb"):
            from_spliece_explanations([], terms, emb)

    def test_missing_image_id_falls_back_to_index(self):
        terms, emb = self._vocab()
        explanations = [{"top_k_concepts": [{"feature_id": 0, "name": "term_0", "activation": 1.0}]}]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert cs.per_image[0].image_id == "img_0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestSpliCEAdapter -q`
Expected: FAIL — `ImportError: cannot import name 'from_spliece_explanations'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestSpliCEAdapter -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): SPLiCE adapter with count guard + unresolved-name skip"
```

---

## Task 4: SAE adapter `from_sae_explanations`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import from_sae_explanations


class TestSAEAdapter:
    def _inputs(self):
        terms = [{"term": f"term_{i}"} for i in range(10)]
        emb = torch.eye(10, 512)
        concept_names = {
            "3": {"name": "term_3", "score": 0.4, "is_dead": False},
            "7": {"name": "term_7", "score": 0.3, "is_dead": False},
            "9": {"name": "DEAD_FEATURE", "score": 0.0, "is_dead": True},
        }
        return terms, emb, concept_names

    def test_excludes_dead_features(self):
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.5},   # live
                {"feature_id": 9, "name": "DEAD_FEATURE", "activation": 9.9},  # dead -> dropped
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]
        assert cs.per_image[0].activations == {"term_3": 1.5}

    def test_fid_key_type_coercion(self):
        """concept_names keys are str; explanations feature_id is int -> coerce."""
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 7, "name": "term_7", "activation": 2.0},
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_7"]

    def test_unresolved_name_skipped(self):
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.0},
                {"feature_id": 5, "name": "ghost", "activation": 5.0},  # not in vocab
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]

    def test_missing_concept_names_keeps_all_named(self):
        """If concept_names is empty, no feature is known-dead -> keep all resolvable."""
        terms, emb, _ = self._inputs()
        concept_names = {}
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.0},
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestSAEAdapter -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestSAEAdapter -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): SAE adapter with dead-feature exclusion + fid coercion"
```

---

## Task 5: `cluster_concepts`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import cluster_concepts


def _cs(names_embs, per_image=None):
    names, embs = names_embs
    return ConceptSet(
        names=names,
        embeddings=torch.tensor(embs, dtype=torch.float32),
        name_to_idx={n: i for i, n in enumerate(names)},
        per_image=per_image or [],
    )


class TestCluster:
    def test_two_well_separated_groups(self):
        # two clusters in 2D (padded to 512): {a,b} near (0,0), {c,d} near (10,10)
        import math
        z = [0.0] * 512
        far = [10.0] + [0.0] * 511
        cs = _cs((["a", "b", "c", "d"], [z, z, far, far]))
        clusters = cluster_concepts(cs, n_clusters=2)
        assert len(clusters) == 2
        ids = {frozenset(c.members) for c in clusters}
        assert frozenset({"a", "b"}) in ids
        assert frozenset({"c", "d"}) in ids
        # medoid is a real member
        for c in clusters:
            assert c.medoid in c.members

    def test_deterministic_same_input_same_ids(self):
        import random
        rows = [[float(x) for x in [random.random()]] + [0.0] * 511 for _ in range(8)]
        names = [f"n{i}" for i in range(8)]
        cs = _cs((names, rows))
        c1 = cluster_concepts(cs, n_clusters=3)
        c2 = cluster_concepts(cs, n_clusters=3)
        assert [c.members for c in c1] == [c.members for c in c2]
        assert [c.cluster_id for c in c1] == [0, 1, 2]

    def test_singleton_input(self):
        cs = _cs((["only"], [[0.0] * 512]))
        clusters = cluster_concepts(cs, n_clusters=1)
        assert len(clusters) == 1
        assert clusters[0].members == ["only"]
        assert clusters[0].medoid == "only"

    def test_empty_input(self):
        cs = _cs(([], [[]]))
        # fix: empty embeddings must be handled
        cs = ConceptSet(names=[], embeddings=torch.empty((0, 512)), name_to_idx={}, per_image=[])
        clusters = cluster_concepts(cs)
        assert clusters == []

    def test_distance_threshold_mode(self):
        z = [0.0] * 512
        far = [5.0] + [0.0] * 511
        cs = _cs((["a", "b", "c"], [z, z, far]))
        clusters = cluster_concepts(cs, distance_threshold=0.5, linkage="average")
        # a,b together; c alone
        member_sets = {frozenset(c.members) for c in clusters}
        assert frozenset({"a", "b"}) in member_sets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestCluster -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
def _medoid(members: list[str], embeddings: torch.Tensor, name_to_idx: dict[str, int]) -> str:
    """Member whose embedding has max mean cosine similarity to the others."""
    if len(members) == 1:
        return members[0]
    idx = [name_to_idx[m] for m in members]
    rows = embeddings[idx]                              # (k, D)
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = rows @ rows.T                                 # (k, k)
    mean_sim = sim.sum(dim=1) / (len(members) - 1)      # exclude self via off-diagonal avg
    # subtract self-sim (==1) contribution: sum includes diagonal
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

    # group members, sort each group + sort groups by member tuple for stable ids
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestCluster -q`
Expected: PASS (5 tests). If the "two well-separated groups" test is flaky due to cosine on zero-norm rows (`a` and `b` are identical zero vectors → norm 0 → undefined cosine), that is expected to surface here. Fix by making the test rows unit-norm (see Step 1 note below).

> **Note on the zero-vector test fixture:** rows of all-zeros have undefined cosine. If `test_two_well_separated_groups` fails numerically, replace `z = [0.0]*512` with a fixed unit vector `z = [1.0] + [0.0]*511` (so `a==b` are identical unit vectors, cosine 0 distance → merged; `c==d` at `[1.0]+...`? no — they must differ from a/b). Use:
> ```python
> z1 = [1.0] + [0.0] * 511      # a, b
> z2 = [0.0, 1.0] + [0.0] * 510  # c, d
> ```
> Apply this fix in the test if needed before declaring green.

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): deterministic agglomerative cosine clustering"
```

---

## Task 6: `ancestor_rids` helper + `annotate_radlex`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import ancestor_rids, annotate_radlex
from vocabulary_building.radlex_support import RadLexGraph


def _graph():
    """Tiny DAG: root <- mid <- leaf ; root <- sibling."""
    g = RadLexGraph()
    g.rid_to_label = {"RID0": "root", "RID1": "mid", "RID2": "leaf", "RID3": "sibling"}
    g.label_to_rids = {"root": ["RID0"], "mid": ["RID1"], "leaf": ["RID2"], "sibling": ["RID3"]}
    g.child_to_parents = {"RID2": ["RID1"], "RID1": ["RID0"], "RID3": ["RID0"]}
    return g


class TestAnnotate:
    def test_ancestor_rids_walks_parents(self):
        g = _graph()
        assert ancestor_rids(g, "RID2") == {"RID2", "RID1", "RID0"}
        assert ancestor_rids(g, "RID0") == {"RID0"}  # root has no parents

    def test_cluster_gets_specific_common_ancestor(self):
        g = _graph()
        # two members resolving to RID2 (leaf) and RID1 (mid) share RID1 + RID0;
        # most specific common = RID1 (not the root RID0).
        clusters = [Cluster(cluster_id=0, members=["leaf", "mid"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].radlex_label == "mid"
        assert annotated[0].radlex_rid == "RID1"
        assert annotated[0].n_resolved == 2
        assert annotated[0].n_members == 2

    def test_root_only_common_falls_back_to_none(self):
        g = _graph()
        # leaf (under mid under root) and sibling (under root) share ONLY root.
        # root is rejected as trivially uninformative -> radlex_label None.
        clusters = [Cluster(cluster_id=0, members=["leaf", "sibling"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].radlex_label is None
        assert annotated[0].radlex_rid is None

    def test_unresolved_members_skipped(self):
        g = _graph()
        clusters = [Cluster(cluster_id=0, members=["leaf", "ghost"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].n_resolved == 1
        assert annotated[0].n_members == 2
        # single resolved member -> its own label is most specific common ancestor
        assert annotated[0].radlex_label == "leaf"

    def test_empty_cluster_no_crash(self):
        g = _graph()
        clusters: list[Cluster] = []
        assert annotate_radlex(clusters, g) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestAnnotate -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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


def annotate_radlex(clusters: list[Cluster], graph) -> list[AnnotatedCluster]:
    """Annotate each cluster with a best-effort RadLex ancestor (root-degeneracy guard).

    Selection rule (spec §6.2):
      1. resolve each member to a RID; collect ancestor_rids per resolved member.
      2. candidate = ancestor RID supported by >= ceil(0.5 * n_resolved) members.
      3. reject roots (RIDs with no parents) as trivially uninformative.
      4. pick the most specific candidate = the one with the most candidate-ancestors
         (deepest); tie-break by support desc, then shortest label.
      5. no surviving candidate -> radlex_label None (cluster falls back to medoid).
    Never raises on unresolved members or empty clusters.
    """
    import math

    out: list[AnnotatedCluster] = []
    for c in clusters:
        member_anc: list[set[str]] = []
        for m in c.members:
            rid = _resolve_member_rid(graph, m)
            if rid is not None:
                member_anc.append(ancestor_rids(graph, rid))
        n_resolved = len(member_anc)
        n_members = len(c.members)

        radlex_rid: str | None = None
        radlex_label: str | None = None
        if n_resolved > 0:
            threshold = math.ceil(0.5 * n_resolved)
            support: dict[str, int] = {}
            for anc_set in member_anc:
                for r in anc_set:
                    support[r] = support.get(r, 0) + 1
            # candidates: majority-supported, non-root (must have parents)
            candidates = [
                r for r, cnt in support.items()
                if cnt >= threshold and r in graph.child_to_parents
            ]
            if candidates:
                # specificity = how many OTHER candidates are ancestors of cand
                cand_set = set(candidates)
                anc_of = {r: ancestor_rids(graph, r) for r in candidates}

                def sort_key(r: str):
                    specificity = len(anc_of[r] & cand_set)  # includes self -> +1 ok for ranking
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestAnnotate -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): RadLex ancestor annotation with root-degeneracy guard"
```

---

## Task 7: `build_structured_explanations`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import build_structured_explanations, AnnotatedCluster


class TestStructured:
    def test_family_aggregation_and_redundancy(self):
        clusters = [
            AnnotatedCluster(cluster_id=0, members=["a", "b"], medoid="a", radlex_label="famA"),
            AnnotatedCluster(cluster_id=1, members=["c"], medoid="c", radlex_label=None),
        ]
        cs = ConceptSet(
            names=["a", "b", "c"],
            embeddings=torch.zeros((3, 512)),
            name_to_idx={"a": 0, "b": 1, "c": 2},
            per_image=[ImageConcepts(image_id="img0", activations={"a": 1.0, "b": 2.0, "c": 4.0})],
        )
        out = build_structured_explanations(cs, clusters)
        assert len(out) == 1
        ex = out[0]
        assert ex["image_id"] == "img0"
        # two families: {a,b} and {c}
        fam_by_label = {f["label"]: f for f in ex["families"]}
        assert set(fam_by_label) == {"famA", "c"}  # cluster1 radlex None -> medoid "c"
        assert fam_by_label["famA"]["aggregate_activation"] == 3.0
        assert fam_by_label["famA"]["intra_redundancy"] == 2
        assert len(fam_by_label["famA"]["concepts"]) == 2
        # 3 raw concepts / 2 families = 1.5
        assert ex["redundancy_score"] == 1.5

    def test_image_with_no_active_concepts(self):
        clusters = [AnnotatedCluster(cluster_id=0, members=["a"], medoid="a")]
        cs = ConceptSet(
            names=["a"],
            embeddings=torch.zeros((1, 512)),
            name_to_idx={"a": 0},
            per_image=[ImageConcepts(image_id="empty", activations={})],
        )
        out = build_structured_explanations(cs, clusters)
        assert out[0]["families"] == []
        assert out[0]["redundancy_score"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestStructured -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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
        # deterministic order by cluster_id
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestStructured -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): per-sample structured family explanations + redundancy"
```

---

## Task 8: `compute_metrics`

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from concept_discovery.organize import compute_metrics


class TestMetrics:
    def test_basic_metrics_and_redundancy_reduction(self):
        clusters = [
            AnnotatedCluster(cluster_id=0, members=["a", "b"], medoid="a",
                             radlex_label="famA", n_resolved=2, n_members=2),
            AnnotatedCluster(cluster_id=1, members=["c"], medoid="c",
                             radlex_label=None, n_resolved=0, n_members=1),
        ]
        cs = ConceptSet(
            names=["a", "b", "c"],
            embeddings=torch.tensor([[1.0] + [0.0] * 511,
                                      [1.0] + [0.0] * 511,
                                      [0.0, 1.0] + [0.0] * 510], dtype=torch.float32),
            name_to_idx={"a": 0, "b": 1, "c": 2},
            per_image=[
                ImageConcepts(image_id="i0", activations={"a": 1.0, "b": 1.0, "c": 1.0}),
                ImageConcepts(image_id="i1", activations={"a": 1.0, "c": 1.0}),
            ],
        )
        structured = [
            {"image_id": "i0", "families": [{}, {}], "redundancy_score": 1.5},
            {"image_id": "i1", "families": [{}], "redundancy_score": 2.0},
        ]
        m = compute_metrics(cs, clusters, structured)
        assert m["n_concepts_active"] == 3
        assert m["n_clusters"] == 2
        assert m["mean_cluster_size"] == 1.5
        assert m["radlex_coverage_pct"] == (2 / 3) * 100
        assert m["n_empty_images"] == 0
        # mean raw = (3+2)/2 = 2.5 ; mean families = (2+1)/2 = 1.5 -> 2.5/1.5
        assert abs(m["redundancy_reduction"] - (2.5 / 1.5)) < 1e-9

    def test_silhouette_none_below_two_clusters(self):
        clusters = [AnnotatedCluster(cluster_id=0, members=["a"], medoid="a")]
        cs = ConceptSet(
            names=["a"], embeddings=torch.zeros((1, 512)),
            name_to_idx={"a": 0}, per_image=[],
        )
        m = compute_metrics(cs, clusters, [])
        assert m["silhouette_cosine"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestMetrics -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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

    # silhouette: need >= 2 clusters AND >= 2 samples AND >1 distinct label
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

    # redundancy reduction = mean(raw per image) / mean(families per image)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestMetrics -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): organization metrics (silhouette, redundancy, coverage)"
```

---

## Task 9: `run()` orchestrator + output files

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)
- Test: `tests/unit/test_organize.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_organize.py`:

```python
from dataclasses import replace
import config
from concept_discovery.organize import run as organize_run


class TestRun:
    def test_run_writes_three_output_files(self, tmp_path):
        terms = [{"term": f"t{i}"} for i in range(6)]
        emb = torch.eye(6, 512)
        # 3 images, each a positive combo of 2 vocab atoms
        explanations = [
            {"image_id": "i0", "top_k_concepts": [
                {"feature_id": 0, "name": "t0", "activation": 0.5},
                {"feature_id": 1, "name": "t1", "activation": 0.5}]},
            {"image_id": "i1", "top_k_concepts": [
                {"feature_id": 2, "name": "t2", "activation": 0.5},
                {"feature_id": 3, "name": "t3", "activation": 0.5}]},
            {"image_id": "i2", "top_k_concepts": [
                {"feature_id": 4, "name": "t4", "activation": 0.5},
                {"feature_id": 5, "name": "t5", "activation": 0.5}]},
        ]
        cs = from_spliece_explanations(explanations, terms, emb)
        cfg = replace(config.organize, n_clusters=2, output_dir=tmp_path, radlex_csv_path=tmp_path / "no.csv")
        metrics = organize_run(cfg, cs, graph=None)
        assert (tmp_path / "concept_clusters.json").exists()
        assert (tmp_path / "structured_explanations.json").exists()
        assert (tmp_path / "organization_metrics.json").exists()
        assert "n_clusters" in metrics
        import json
        clusters_json = json.loads((tmp_path / "concept_clusters.json").read_text())
        assert all("display_label" in c or "label" in c for c in clusters_json)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestRun -q`
Expected: FAIL — `ImportError: cannot import name 'run'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/concept_discovery/organize.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py::TestRun -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/concept_discovery/organize.py tests/unit/test_organize.py
git commit -m "feat(organize): run() orchestrator writes 3 output JSON files"
```

---

## Task 10: `__main__` self-check (throwaway dir)

**Files:**
- Modify: `src/concept_discovery/organize.py` (append)

- [ ] **Step 1: Append self-check**

Append to `src/concept_discovery/organize.py`:

```python
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
```

Also add `from pathlib import Path` to the imports at the top of the file (after `import torch`), since the self-check uses `Path`.

- [ ] **Step 2: Run the self-check**

Run: `PYTHONPATH=src:. .venv/bin/python src/concept_discovery/organize.py`
Expected: prints "✅ Self-check passed" and writes `/tmp/organize_selfcheck/`.

- [ ] **Step 3: Commit**

```bash
git add src/concept_discovery/organize.py
git commit -m "feat(organize): __main__ self-check on throwaway dir"
```

---

## Task 11: Driver `scripts/run_concept_organization.py`

**Files:**
- Create: `scripts/run_concept_organization.py`

- [ ] **Step 1: Write the driver**

Create `scripts/run_concept_organization.py`:

```python
"""run_concept_organization.py — orchestrate the concept-organization extension.

Thin driver over ``src.concept_discovery.organize``. Clusters discovered concepts
(SPLiCE or SAE) by RadLex text-embedding cosine, annotates clusters with a
best-effort RadLex ancestor, and emits structured per-image explanations.

Usage:
    python scripts/run_concept_organization.py --source spliece
    python scripts/run_concept_organization.py --source sae-hidden --tag run2
    python scripts/run_concept_organization.py --source spliece --no-radlex
    python scripts/run_concept_organization.py --source spliece --n-clusters 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/ -> config, utils, concept_discovery
sys.path.insert(0, str(_HERE.parent))          # repo root

import config
from concept_discovery.organize import (
    from_spliece_explanations,
    from_sae_explanations,
    run as organize_run,
)
from utils import load_tensor
from vocabulary_building.radlex_support import load_radlex_graph


_SOURCE_DEFAULTS = {
    "spliece": {
        "explanations": lambda: config.paths.results_dir / "spliece" / "sample_explanations.json",
        "concept_names": None,
    },
    "sae-baseline": {
        "explanations": lambda: config.paths.results_dir / "baseline" / "sample_explanations.json",
        "concept_names": lambda: config.paths.results_dir / "baseline" / "concept_names.json",
    },
    "sae-hidden": {
        "explanations": lambda: config.paths.results_dir / "sae_hidden" / "sample_explanations.json",
        "concept_names": lambda: config.paths.results_dir / "sae_hidden" / "concept_names.json",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the concept-organization extension end-to-end."
    )
    p.add_argument("--source", required=True, choices=list(_SOURCE_DEFAULTS),
                   help="which method's explanations to organize")
    p.add_argument("--tag", default=None, help="suffix: results/concept_organization_{tag}/")
    p.add_argument("--n-clusters", type=int, default=None, help="override OrganizeConfig.n_clusters")
    p.add_argument("--distance", type=float, default=None, help="linkage distance threshold (mutually exclusive with --n-clusters)")
    p.add_argument("--no-radlex", action="store_true", help="skip RadLex ancestor annotation")
    # input overrides (dataset portability)
    p.add_argument("--explanations", type=Path, default=None)
    p.add_argument("--concept-names", type=Path, default=None)
    p.add_argument("--vocab", type=Path, default=None)
    p.add_argument("--vocab-emb", type=Path, default=None)
    p.add_argument("--radlex", type=Path, default=None)
    return p.parse_args()


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (source={ '--source' }); regenerate it first.")
    with open(path) as f:
        return json.load(f)


def _repro_info(inputs: dict) -> list[str]:
    import hashlib
    import subprocess
    lines: list[str] = []
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, cwd=config.paths.project_root).stdout.strip()
        lines.append(f"- git commit: `{sha or 'unknown'}`")
    except Exception:
        lines.append("- git commit: unknown")
    try:
        import sklearn, torch, numpy
        lines.append(f"- versions: scikit-learn {sklearn.__version__} | torch {torch.__version__} | numpy {numpy.__version__}")
    except Exception:
        pass
    for label, path in inputs.items():
        if path and Path(path).exists():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
            lines.append(f"- sha256({label}) [{Path(path).name}]: `{digest}`")
        elif path:
            lines.append(f"- sha256({label}): <missing>")
    return lines


def _write_report(output_dir: Path, args, cfg, metrics, inputs, total) -> None:
    sections = [
        ("Run config",
         f"| param | value |\n|-------|-------|\n"
         f"| source | {args.source} |\n| tag | {args.tag or '—'} |\n"
         f"| output dir | {output_dir} |\n| n_clusters | {cfg.n_clusters} |\n"
         f"| distance_threshold | {cfg.distance_threshold} |\n| linkage | {cfg.linkage} |\n"
         f"| radlex annotation | {'disabled' if args.no_radlex else 'enabled'} |"),
        ("Metrics",
         f"| metric | value |\n|-------|-------|\n"
         f"| n_concepts_active | {metrics['n_concepts_active']} |\n"
         f"| n_clusters | {metrics['n_clusters']} |\n"
         f"| mean_cluster_size | {metrics['mean_cluster_size']:.2f} |\n"
         f"| silhouette_cosine | {metrics['silhouette_cosine']} |\n"
         f"| redundancy_reduction | {metrics['redundancy_reduction']:.3f} |\n"
         f"| radlex_coverage_pct | {metrics['radlex_coverage_pct']:.1f} |\n"
         f"| n_empty_images | {metrics['n_empty_images']} |"),
        ("Output files",
         "- `concept_clusters.json` — clusters with RadLex ancestor labels\n"
         "- `structured_explanations.json` — per-image concept families + redundancy\n"
         "- `organization_metrics.json` — metrics snapshot"),
        ("Reproducibility", "\n".join(_repro_info(inputs))),
        ("References",
         "- Spec: `docs/design/proposals/2026-07-03-concept-organization.md`\n"
         "- Plan: `docs/plans/2026-07-03-concept-organization.md`"),
    ]
    body = f"# Concept Organization — Pipeline Run\n\n"
    body += f"**Status**: Complete ✅\n**Total time**: {total:.1f}s\n**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for title, content in sections:
        body += f"## {title}\n\n{content}\n\n"
    (output_dir / "REPORT_organization.md").write_text(body)


def main() -> None:
    args = parse_args()
    root = config.paths.project_root
    output_dir = root / "results" / f"concept_organization_{args.tag}" if args.tag \
        else root / "results" / "concept_organization"

    defaults = _SOURCE_DEFAULTS[args.source]
    explanations_path = args.explanations or defaults["explanations"]()
    concept_names_path = args.concept_names or (defaults["concept_names"]() if defaults["concept_names"] else None)
    vocab_path = args.vocab or config.organize.vocab_path
    vocab_emb_path = args.vocab_emb or config.organize.vocab_emb_path
    radlex_path = None if args.no_radlex else (args.radlex or config.organize.radlex_csv_path)

    overrides = {"output_dir": output_dir}
    if args.n_clusters is not None:
        overrides["n_clusters"] = args.n_clusters
    if args.distance is not None:
        overrides["distance_threshold"] = args.distance
    cfg = replace(config.organize, **overrides)

    print("=" * 64)
    print(f"  Concept organization  (source={args.source}" + (f", tag={args.tag}" if args.tag else "") + ")")
    print(f"  output: {output_dir}")
    print("=" * 64)

    t0 = time.time()
    vocab_terms = _load_json(vocab_path)
    vocab_emb = load_tensor(vocab_emb_path)
    explanations = _load_json(explanations_path)

    if args.source == "spliece":
        cs = from_spliece_explanations(explanations, vocab_terms, vocab_emb)
    else:
        concept_names = _load_json(concept_names_path) if concept_names_path else {}
        cs = from_sae_explanations(explanations, concept_names, vocab_terms, vocab_emb)

    graph = None
    if radlex_path and Path(radlex_path).exists():
        print(f"  loading RadLex graph: {radlex_path}")
        graph = load_radlex_graph(radlex_path)
    elif radlex_path:
        print(f"  ⚠ RadLex CSV not found at {radlex_path}; skipping annotation.")

    metrics = organize_run(cfg, cs, graph=graph)
    total = time.time() - t0

    inputs = {
        "explanations": explanations_path, "vocab": vocab_path,
        "vocab_emb": vocab_emb_path, "radlex": radlex_path,
        "concept_names": concept_names_path,
    }
    _write_report(output_dir, args, cfg, metrics, inputs, total)
    print(f"\nDone in {total:.1f}s. Report: {output_dir / 'REPORT_organization.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on synthetic (no real data needed yet)**

Verify the driver imports cleanly:
Run: `PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --help`
Expected: prints usage with `--source` required, no import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_concept_organization.py
git commit -m "feat(driver): run_concept_organization.py CLI (mirrors run_spliece.py)"
```

---

## Task 12: Integration test (end-to-end on synthetic)

**Files:**
- Create: `tests/integration/test_organize_pipeline.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_organize_pipeline.py`:

```python
"""Integration test: full organize pipeline on synthetic data."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src/")

import torch
import config
from concept_discovery.organize import (
    from_spliece_explanations, from_sae_explanations, run as organize_run,
)


def _write_inputs(tmp_path, source):
    V = 12
    torch.manual_seed(0)
    emb = torch.eye(V, 512)
    vocab_path = tmp_path / "vocabulary.json"
    vocab_path.write_text(json.dumps([{"term": f"t{i}"} for i in range(V)]))
    emb_path = tmp_path / "vocab_emb.pt"
    torch.save(emb, emb_path)
    explanations = [{
        "image_id": f"img_{i}",
        "top_k_concepts": [
            {"feature_id": (i % V), "name": f"t{i%V}", "activation": 1.0},
            {"feature_id": (i + 1) % V, "name": f"t{(i+1)%V}", "activation": 0.5},
        ],
    } for i in range(8)]
    expl_path = tmp_path / "sample_explanations.json"
    expl_path.write_text(json.dumps(explanations))
    cn_path = None
    if source.startswith("sae"):
        cn = {str(i): {"name": f"t{i}", "score": 0.5, "is_dead": False} for i in range(V)}
        cn_path = tmp_path / "concept_names.json"
        cn_path.write_text(json.dumps(cn))
    return vocab_path, emb_path, expl_path, cn_path


class TestPipeline:
    def test_spliece_end_to_end(self, tmp_path):
        vocab_path, emb_path, expl_path, _ = _write_inputs(tmp_path, "spliece")
        vocab_terms = json.loads(vocab_path.read_text())
        emb = torch.load(emb_path, weights_only=True)
        expl = json.loads(expl_path.read_text())
        cs = from_spliece_explanations(expl, vocab_terms, emb)
        cfg = replace(config.organize, n_clusters=3, output_dir=tmp_path,
                      radlex_csv_path=tmp_path / "no.csv")
        metrics = organize_run(cfg, cs, graph=None)
        assert (tmp_path / "concept_clusters.json").exists()
        assert (tmp_path / "structured_explanations.json").exists()
        assert (tmp_path / "organization_metrics.json").exists()
        assert metrics["n_clusters"] == 3
        # determinism: a second run with identical inputs yields identical cluster members
        import json as _json
        first = _json.loads((tmp_path / "concept_clusters.json").read_text())
        cs2 = from_spliece_explanations(expl, vocab_terms, emb)
        organize_run(cfg, cs2, graph=None)
        second = _json.loads((tmp_path / "concept_clusters.json").read_text())
        assert [c["members"] for c in first] == [c["members"] for c in second]

    def test_sae_end_to_end(self, tmp_path):
        vocab_path, emb_path, expl_path, cn_path = _write_inputs(tmp_path, "sae-hidden")
        vocab_terms = json.loads(vocab_path.read_text())
        emb = torch.load(emb_path, weights_only=True)
        expl = json.loads(expl_path.read_text())
        cn = json.loads(cn_path.read_text())
        cs = from_sae_explanations(expl, cn, vocab_terms, emb)
        cfg = replace(config.organize, n_clusters=3, output_dir=tmp_path,
                      radlex_csv_path=tmp_path / "no.csv")
        organize_run(cfg, cs, graph=None)
        assert (tmp_path / "structured_explanations.json").exists()
```

- [ ] **Step 2: Run the integration test**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/integration/test_organize_pipeline.py -q`
Expected: PASS (2 tests).

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py tests/unit/test_organize_config.py tests/integration/test_organize_pipeline.py -q`
Expected: PASS (all new tests).

Also run the whole suite to confirm no regression:
Run: `PYTHONPATH=src:. .venv/bin/pytest -q`
Expected: only the 1 pre-existing `test_extract_embeddings` failure remains; everything else passes.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_organize_pipeline.py
git commit -m "test(organize): end-to-end integration on synthetic SPLiCE + SAE inputs"
```

---

## Task 13: Run on real data + verify

**Files:** none (verification run only; outputs land in gitignored `results/`).

- [ ] **Step 1: Run on real SPLiCE output (with RadLex)**

Run: `PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source spliece`
Expected: completes, writes `results/concept_organization/{concept_clusters,structured_explanations,organization_metrics}.json` + `REPORT_organization.md`. Console prints `redundancy_reduction` and `radlex_coverage_pct`.

- [ ] **Step 2: Run on SAE baseline + SAE hidden**

Run: `PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source sae-baseline --tag baseline`
Run: `PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source sae-hidden --tag hidden`
Expected: both complete; outputs in `results/concept_organization_baseline/` and `results/concept_organization_hidden/`.

- [ ] **Step 3: Sanity-check the outputs**

Inspect:
- `results/concept_organization/concept_clusters.json` — confirm clusters have `radlex_label` populated for at least some clusters (coverage > 0); confirm NOT all clusters share the same `radlex_label` (root-degeneracy guard works).
- `results/concept_organization/organization_metrics.json` — confirm `redundancy_reduction > 1.0` (organization reduces redundancy) and `radlex_coverage_pct` is a plausible non-zero value.
- `results/concept_organization/REPORT_organization.md` — confirm reproducibility block lists git SHA + input hashes.

If `radlex_coverage_pct` is 0, the vocab terms do not match RadLex preferred labels by exact string — this is a known limitation (best-effort), not a bug; report it honestly in the recap. If all clusters share one `radlex_label`, the root guard failed — revisit Task 6.

- [ ] **Step 4: Commit the run reports (results/ is tracked for reports)**

> **Note:** `results/` is partially gitignored. Per repo convention (`git log` shows commits like "chore(results): track results/ in git"), commit the `REPORT_organization.md` files only if the team tracks reports. Otherwise skip this step.

```bash
git add results/concept_organization*/REPORT_organization.md
git commit -m "chore(results): concept-organization run reports (spliece + sae)"
```

---

## Self-Review (completed)

**1. Spec coverage:** every spec section maps to a task — OrganizeConfig (T1), ConceptSet/Cluster dataclasses (T2), SPLiCE adapter §5.3 (T3), SAE adapter §5.3 (T4), cluster_concepts §6.1 (T5), ancestor_rids + annotate_radlex §6.2 (T6), build_structured_explanations §6.3 (T7), compute_metrics §6.4 (T8), run + output files §6.5/§7 (T9), self-check §10 (T10), driver §9 (T11), integration §11 (T12), real-run verification (T13). No spec gap.

**2. Placeholder scan:** no TBD/TODO/"add error handling" in code steps; every code step shows complete code. One conditional note in T5 (zero-vector test fixture) gives the exact fix to apply if the assertion is numerically unstable — not a placeholder, a deterministic fix.

**3. Type/name consistency:** `ConceptSet`, `ImageConcepts`, `Cluster`, `AnnotatedCluster`, `display_label`, `cluster_concepts`, `annotate_radlex`, `ancestor_rids`, `build_structured_explanations`, `compute_metrics`, `run`, `from_spliece_explanations`, `from_sae_explanations` — names match across all tasks and the spec. `OrganizeConfig` fields (`n_clusters`, `distance_threshold`, `linkage`, `metric`, `radlex_csv_path`, `vocab_path`, `vocab_emb_path`, `output_dir`) match between T1, T9, T11.
