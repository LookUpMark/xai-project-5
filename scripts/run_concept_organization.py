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
        raise FileNotFoundError(f"{path} missing; regenerate it first.")
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
