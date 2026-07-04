"""08_baseline_ablation.py — baseline (512-d) hyperparameter sweep (dict_size x k).

Runs ``train -> naming -> stability`` for each preset under ``baseline_variant``,
**reusing the cached ``embeddings/standard/`` tensors** (dict_size/k don't change
the input). Harvests dead%, reconstruction cosine, L0, naming mean, and
cross-seed mean Jaccard into one comparison report.

Usage:
    python scripts/08_baseline_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/  -> config, utils, autoencoder, sae_hidden
sys.path.insert(0, str(_HERE.parent))          # repo root

import config
from autoencoder import concept_naming, stability_analysis, train_sae
from autoencoder.variant import baseline_variant
from sae_hidden.reports import md_table, write_report

# Steps held fixed at the baseline default; the sweep varies dict_size and k.
STEPS = config.sae.steps
PRESETS: dict[str, dict[str, int]] = {
    "conservative": dict(dict_size=1024, k=16),
    "default": dict(dict_size=2048, k=32),  # the current SAEConfig
    "aggressive": dict(dict_size=4096, k=64),
}


def run_preset(tag: str, overrides: dict[str, int], abl_dir: Path) -> Path:
    root = config.paths.project_root
    models_dir = root / "models" / f"sae_baseline_abl_{tag}"
    results_dir = abl_dir / tag
    sae = replace(config.sae, steps=STEPS, **overrides)
    print(
        f"\n=== preset {tag}: dict_size={overrides['dict_size']} "
        f"k={overrides['k']} steps={STEPS} ==="
    )
    t0 = time.time()
    # No embeddings override -> baseline always reads embeddings/standard/.
    with baseline_variant(sae=sae, models_dir=models_dir, results_dir=results_dir):
        train_sae.main()
        concept_naming.run()
        stability_analysis.run()
    print(f"=== {tag} done in {time.time() - t0:.0f}s ===")
    return results_dir


def _f(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def harvest(results_dir: Path) -> dict:
    """dead%/recon/L0 from per-seed stability; match/null from matched stability."""
    m: dict = {}
    sm = results_dir / "stability_analysis.json"
    if sm.exists():
        d = json.load(open(sm))
        psm = list(d.get("per_seed_metrics", {}).values())
        if psm:
            m["dead_pct"] = float(np.mean([r["dead_features_pct"] for r in psm]))
            m["recon_cos"] = float(np.mean([r["cosine"] for r in psm]))
            m["l0"] = float(np.mean([r["l0_mean"] for r in psm]))
    sml = results_dir / "stability_matched.json"
    if sml.exists():
        d = json.load(open(sml))
        m["match_cos"] = d.get("mean_best_match_cosine")
        m["null_cos"] = d.get("null_mean")
        m["frac_0.9"] = d.get("mean_frac_matched_0.9")
    cn = results_dir / "concept_names.json"
    if cn.exists():
        d = json.load(open(cn))
        live = [v for v in d.values() if not v.get("is_dead")]
        m["naming_mean"] = float(np.mean([v["score"] for v in live])) if live else 0.0
    return m


def main() -> None:
    abl_dir = config.paths.project_root / "results" / "sae_baseline_ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)

    per: dict[str, dict] = {}
    for tag, ov in PRESETS.items():
        per[tag] = harvest(run_preset(tag, ov, abl_dir))

    _write_comparison(abl_dir, per)
    print(f"\nComparison: {abl_dir / 'REPORT_ablation.md'}")


def _write_comparison(abl_dir: Path, per: dict[str, dict]) -> None:
    rows = []
    for tag, ov in PRESETS.items():
        m = per.get(tag, {})
        mc, nc = m.get("match_cos"), m.get("null_cos")
        ratio = (
            f"{mc / nc:.2f}x"
            if (isinstance(mc, (int, float)) and isinstance(nc, (int, float)) and nc)
            else "—"
        )
        rows.append([
            tag, ov["dict_size"], ov["k"],
            _f(m.get("dead_pct"), 1), _f(m.get("recon_cos")), _f(m.get("l0"), 1),
            _f(m.get("naming_mean"), 4), _f(mc), _f(nc), ratio,
        ])
    summary = (
        f"Baseline (512-d) dict_size x k sweep (steps={STEPS}, seeds={list(config.training.seeds)}), "
        f"reusing cached standard embeddings. dead%/recon/L0 are per-seed means over the test set; "
        f"match/null is the permutation-invariant decoder best-match cosine over the isotropic null "
        f"(>1x = shared subspace above chance). Slot-wise Jaccard is omitted — degenerate by construction "
        f"(ML-AUDIT-2026-06-26 F-001)."
    )
    sections = [
        (
            "Presets",
            md_table(
                ["preset", "dict_size", "k"],
                [[t, ov["dict_size"], ov["k"]] for t, ov in PRESETS.items()],
            ),
        ),
        (
            "Results",
            md_table(
                ["preset", "dict_size", "k", "dead%", "recon cos", "L0",
                 "naming mean", "match cos", "null cos", "match/null"],
                rows,
            ),
        ),
        (
            "Notes",
            "- **dead%**: activation-based, per-seed mean over the test set.\n"
            "- **naming mean**: mean top-1 cosine of live decoder features vs RadLex "
            "(random ~0.372).\n"
            "- **match cos / null cos**: permutation-invariant decoder best-match cosine vs "
            "isotropic null (>1x = shared subspace above chance). Slot-wise Jaccard is omitted — "
            "~0 by construction for SAEs with no canonical feature ordering "
            "(ML-AUDIT-2026-06-26 F-001).\n"
            "- Per-preset artifacts under `results/sae_baseline_ablation/{preset}/`.",
        ),
    ]
    write_report(abl_dir / "REPORT_ablation.md", "Baseline (512-d) — dict_size/k Ablation", sections, summary)


if __name__ == "__main__":
    main()
