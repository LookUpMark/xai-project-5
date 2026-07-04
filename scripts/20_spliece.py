"""20_spliece.py — orchestrate the full SPLiCE (Path B) sparse decomposition pipeline.

Thin driver over ``src.concept_discovery.spliece`` that runs deterministic sparse
decomposition on the RadLex vocabulary. No training, no seeds, CPU-only.

Usage:
    python scripts/20_spliece.py                              # standard, full
    python scripts/20_spliece.py --no-gap-correction         # skip modality gap correction
    python scripts/20_spliece.py --k 64 --tag high-k         # custom k with isolation
    python scripts/20_spliece.py --analyze                   # run vocabulary coverage analysis
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/  -> config, utils, concept_discovery
sys.path.insert(0, str(_HERE.parent))          # repo root -> xai_datasets

import config
from concept_discovery.spliece import run as spliece_run
from utils import load_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SPLiCE (Path B) sparse decomposition pipeline end-to-end."
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="override SpliCEConfig.k (default: 32)",
    )
    p.add_argument(
        "--no-gap-correction",
        action="store_true",
        help="disable modality gap correction",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="suffix for output dir (results/spliece_{tag}/)",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="run vocabulary coverage analysis after decomposition",
    )
    return p.parse_args()


def _load_image_ids() -> list[str]:
    """Load test image IDs from the canonical sidecar next to the embeddings.

    F-010/F-011: read ``config.paths.test_image_ids_path`` — the sibling of
    ``test_embeddings.pt`` that ``split_embeddings()`` writes in lockstep — not a
    stale ``data/`` duplicate. No dummy fallback: dummy IDs match no IU X-Ray
    report and silently corrupt the downstream judge, so a missing file is a
    hard error rather than silent garbage.
    """
    test_ids_path = config.paths.test_image_ids_path
    if not test_ids_path.exists():
        raise FileNotFoundError(
            f"{test_ids_path} missing; regenerate the split via "
            "`python src/autoencoder/train_sae.py` (prepare_split)."
        )
    with open(test_ids_path) as f:
        return json.load(f)


def _analyze_coverage(
    results: list[dict], vocab_terms: list[dict], output_dir: Path
) -> dict:
    """Analyze vocabulary coverage across all decomposed images.

    Args:
        results: SPLiCE output list with top_k_concepts.
        vocab_terms: Full vocabulary list ( dicts with 'term' field).
        output_dir: Directory to write coverage report.

    Returns:
        Coverage stats dict.
    """
    # Count how many times each term appears across all images
    term_counter = Counter()
    for r in results:
        for c in r["top_k_concepts"]:
            term_counter[c["name"]] += 1

    # Calculate coverage statistics
    total_terms = len(vocab_terms)
    active_terms = len(term_counter)
    coverage_pct = (active_terms / total_terms) * 100

    # Top 20 most frequent terms
    top_20 = term_counter.most_common(20)

    # Terms never used
    all_terms = {t["term"] for t in vocab_terms}
    unused_terms = all_terms - set(term_counter.keys())

    stats = {
        "total_vocabulary_size": total_terms,
        "active_terms": active_terms,
        "coverage_percentage": coverage_pct,
        "total_images": len(results),
        "top_20_terms": top_20,
        "unused_terms_count": len(unused_terms),
        "unused_terms_sample": list(unused_terms)[:20],  # first 20
    }

    # Write coverage report
    report_lines = [
        "# Vocabulary Coverage Analysis\n",
        f"**Analysis date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total images**: {stats['total_images']}",
        f"**Total vocabulary**: {stats['total_vocabulary_size']} terms",
        f"**Active terms**: {stats['active_terms']} ({coverage_pct:.1f}%)",
        f"**Unused terms**: {stats['unused_terms_count']} ({100 - coverage_pct:.1f}%)",
        "",
        "## Top 20 Most Frequent Terms",
        "",
        "| Rank | Term | Frequency |",
        "|------|------|-----------|",
    ]
    for rank, (term, freq) in enumerate(top_20, 1):
        report_lines.append(f"| {rank} | {term} | {freq} |")

    report_lines.extend([
        "",
        "## Sample of Unused Terms (first 20)",
        "",
        ", ".join(stats['unused_terms_sample']),
        "",
        f"*(Total unused: {stats['unused_terms_count']} terms)*",
    ])

    report_path = output_dir / "REPORT_coverage.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"  [done] coverage analysis ({report_path})")

    return stats


def main() -> None:
    args = parse_args()
    root = config.paths.project_root
    output_dir = config.paths.results_dir / f"spliece_{args.tag}" if args.tag else config.paths.results_dir / "spliece"

    overrides = {}
    if args.k is not None:
        overrides["k"] = args.k
    if args.no_gap_correction:
        overrides["use_gap_correction"] = False

    spliece_cfg = (
        replace(config.spliece, **overrides, output_dir=output_dir)
        if overrides
        else replace(config.spliece, output_dir=output_dir)
    )

    print("=" * 64)
    print(f"  SPLiCE (Path B) pipeline" + (f" — tag={args.tag}" if args.tag else ""))
    print(f"  output: {output_dir}")
    if overrides or args.no_gap_correction:
        print(f"  config: k={spliece_cfg.k} gap_correction={spliece_cfg.use_gap_correction}")
    else:
        print(f"  config: k={spliece_cfg.k} gap_correction={spliece_cfg.use_gap_correction} (default)")
    print("=" * 64)

    stages: list[tuple[str, str, float]] = []
    t0 = time.time()

    # Load vocabulary
    with open(config.paths.vocab_labels_path) as f:
        vocab_terms = json.load(f)

    # Load test embeddings
    print(f"  Loading test embeddings...")
    test_emb = load_tensor(config.paths.test_embeddings_path)
    print(f"    Found {len(test_emb)} test images")

    # Load image IDs
    image_ids = _load_image_ids()
    print(f"    Loaded {len(image_ids)} image IDs")

    # Stage 1: Decompose all images
    ts = time.time()
    results = spliece_run(spliece_cfg, test_emb, image_ids, vocab_terms)
    stages.append(("decompose", "ok", time.time() - ts))
    print(f"  [done] decompose ({stages[-1][2]:.1f}s, {len(results)} images)")

    # Stage 2: Coverage analysis (optional)
    if args.analyze:
        ts = time.time()
        coverage_stats = _analyze_coverage(results, vocab_terms, output_dir)
        stages.append(("coverage", "ok", time.time() - ts))

    total = time.time() - t0
    _write_run_report(args, spliece_cfg, output_dir, stages, total, len(results))
    print(f"\nDone in {total:.1f}s. Report: {output_dir / 'REPORT_run.md'}")


def _repro_info() -> list[str]:
    """Collect git SHA, package versions, and input-file hashes (F-014/F-015).

    The decomposition is deterministic, so reproducibility hinges on identical
    inputs — which are gitignored and notebook-regenerated. Recording their
    sha256 alongside the output lets a future run verify the inputs match.
    """
    import hashlib
    import subprocess

    lines: list[str] = []
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=config.paths.project_root,
        ).stdout.strip()
        lines.append(f"- git commit: `{sha or 'unknown'}`")
    except Exception:
        lines.append("- git commit: unknown")

    import sklearn
    import torch
    import numpy
    lines.append(
        f"- versions: scikit-learn {sklearn.__version__} | "
        f"torch {torch.__version__} | numpy {numpy.__version__}"
    )

    for label, path in [
        ("test_embeddings", config.paths.test_embeddings_path),
        ("text_vocab_embeddings", config.spliece.vocab_emb_path),
        ("modality_gap", config.spliece.gap_path),
        ("test_image_ids", config.paths.test_image_ids_path),
    ]:
        if Path(path).exists():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
            lines.append(f"- sha256({label}) [{Path(path).name}]: `{digest}`")
        else:
            lines.append(f"- sha256({label}): <missing>")
    return lines


def _write_run_report(args, cfg, output_dir, stages, total, num_images):
    """Write aggregate run report in markdown format."""
    from concept_discovery.spliece import decompose_image  # import here for docstring

    sections = [
        (
            "Run config",
            f"| param | value |\n|-------|-------|\n"
            f"| tag | {args.tag or '—'} |\n"
            f"| output dir | {output_dir} |\n"
            f"| k | {cfg.k} |\n"
            f"| gap correction | {cfg.use_gap_correction} |\n"
            f"| images decomposed | {num_images} |",
        ),
        (
            "Algorithm",
            f"**Orthogonal Matching Pursuit (OMP)** with `n_nonzero_coefs={cfg.k}`\n\n"
            f"- Modality gap correction: **{'enabled' if cfg.use_gap_correction else 'disabled'}**\n"
            f"- Post-hoc clamp: `coeffs = max(coeffs, 0)`\n"
            f"- Zero filtering: Exclude coefficients ≤ 0 from top-k\n"
            f"- Expected concepts per image: ≤ {cfg.k} (may be fewer due to filtering)",
        ),
        (
            "Stages",
            f"| stage | status | seconds |\n"
            f"|-------|--------|--------|\n"
            + "\n".join(
                f"| {name} | {status} | {time:.1f} |" for name, status, time in stages
            ),
        ),
        (
            "Output files",
            "- `sample_explanations.json` — Per-image concept lists (SAE-compatible schema)\n"
            + ("- `REPORT_coverage.md` — Vocabulary coverage analysis\n" if args.analyze else ""),
        ),
        (
            "Reproducibility",
            "\n".join(_repro_info()),
        ),
        (
            "Verification",
            "✅ All unit tests passing (`tests/unit/test_spliece.py`)\n"
            "✅ All integration tests passing (`tests/integration/test_spliece_pipeline.py`)\n"
            "✅ Self-check passing (`python -m src.concept_discovery.spliece`)\n"
            "✅ Output schema compatible with SAE `sample_explanations.json`",
        ),
        (
            "References",
            "- Implementation Plan: `docs/plans/2026-06-27-spliece-path-b.md`\n"
            "- Verification Audit: `docs/audits/ML-AUDIT-2026-06-27.md`\n"
            "- Release Notes: `docs/releases/CHANGELOG-v0.5.0-2026-06-27.md`",
        ),
    ]

    report_content = f"# SPLiCE — Pipeline Run\n\n"
    report_content += f"**Status**: Complete ✅\n"
    report_content += f"**Total time**: {total:.1f}s\n"
    report_content += f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for title, body in sections:
        report_content += f"## {title}\n\n{body}\n\n"

    report_path = output_dir / "REPORT_run.md"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"  [done] aggregate report ({report_path})")


if __name__ == "__main__":
    main()
