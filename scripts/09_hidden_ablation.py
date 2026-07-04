"""09_hidden_ablation.py — Path A hyperparameter sweep (dict_size x k).

Runs the existing ``train -> naming -> stability`` stages under ``hidden_variant``
for each preset, **reusing the cached ``embeddings/standard_hidden/`` tensors**
(dict_size/k don't change the input). Harvests dead%, reconstruction, L0, naming,
and matched-stability into one comparison report.

Usage:
    python scripts/09_hidden_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/  -> config, utils, sae_hidden
sys.path.insert(0, str(_HERE.parent))          # repo root

import config
from sae_hidden import naming_hidden, stability_hidden, train_hidden
from sae_hidden.reports import md_table, write_report
from sae_hidden.variant import hidden_variant

# Steps held fixed at the Path A default; the sweep varies dict_size and k.
STEPS = config.sae_hidden.steps
PRESETS: dict[str, dict[str, int]] = {
    "conservative": dict(dict_size=1024, k=16),
    "default": dict(dict_size=2048, k=32),  # the current SAEHiddenConfig
    "aggressive": dict(dict_size=4096, k=64),
}


def run_preset(tag: str, overrides: dict[str, int], abl_dir: Path) -> Path:
    root = config.paths.project_root
    models_dir = root / "models" / f"sae_hidden_abl_{tag}"
    results_dir = abl_dir / tag
    sae_hidden = replace(config.sae_hidden, steps=STEPS, **overrides)
    print(
        f"\n=== preset {tag}: dict_size={overrides['dict_size']} "
        f"k={overrides['k']} steps={STEPS} ==="
    )
    t0 = time.time()
    # No embeddings_dir override -> reuses cached standard_hidden train/test tensors.
    with hidden_variant(sae_hidden=sae_hidden, models_dir=models_dir, results_dir=results_dir):
        train_hidden.run()
        naming_hidden.run()
        stability_hidden.main()
    print(f"=== {tag} done in {time.time() - t0:.0f}s ===")
    return results_dir


def _f(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def harvest(results_dir: Path) -> dict:
    m: dict = {}
    tm = results_dir / "train_metrics.json"
    if tm.exists():
        rows = json.load(open(tm))
        m["dead_pct"] = float(np.mean([r["dead_features_pct"] for r in rows]))
        m["recon_cos"] = float(np.mean([r["test_cosine"] for r in rows]))
        m["l0"] = float(np.mean([r["l0_mean"] for r in rows]))
    cn = results_dir / "concept_names.json"
    if cn.exists():
        d = json.load(open(cn))
        live = [v for v in d.values() if not v.get("is_dead")]
        m["naming_mean"] = float(np.mean([v["score"] for v in live])) if live else 0.0
    sm = results_dir / "stability_matched.json"
    if sm.exists():
        d = json.load(open(sm))
        m["match_cos"] = d.get("mean_best_match_cosine")
        m["null_cos"] = d.get("null_mean")
        m["frac_0.7"] = d.get("mean_frac_matched_0.7")
    return m


def main() -> None:
    abl_dir = config.paths.project_root / "results" / "sae_hidden_ablation"
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
        ratio = f"{mc / nc:.2f}x" if (isinstance(mc, (int, float)) and isinstance(nc, (int, float)) and nc) else "—"
        rows.append([
            tag, ov["dict_size"], ov["k"],
            _f(m.get("dead_pct"), 1), _f(m.get("recon_cos")), _f(m.get("l0"), 1),
            _f(m.get("naming_mean"), 4), _f(mc), _f(nc), ratio,
        ])
    summary = (
        f"Path A dict_size x k sweep (steps={STEPS}, seeds={list(config.training.seeds)}), "
        f"reusing cached standard_hidden embeddings. dead% is activation-based (train); "
        f"match/null is the cross-seed best-match cosine over the isotropic null."
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
            "- **dead%**: activation-based (train_hidden); baseline 512-d was 40-60%.\n"
            "- **naming mean**: mean top-1 cosine of live decoder features vs RadLex "
            "(random ~0.372).\n"
            "- **match cos / null cos**: cross-seed decoder-cosine best-match vs isotropic "
            "null (>1x = shared subspace above chance).\n"
            "- Per-preset stage REPORTs under `results/sae_hidden_ablation/{preset}/`.",
        ),
    ]
    write_report(abl_dir / "REPORT_ablation.md", "Path A — dict_size/k Ablation", sections, summary)


if __name__ == "__main__":
    main()
