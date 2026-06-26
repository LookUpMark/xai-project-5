"""stability_hidden.py — Path A cross-seed stability on the 768-d hidden state.

The headline diagnostic: do Path A SAEs discover the *same* features across seeds?
Reports mean pairwise cross-seed Jaccard (computed per-sample on the held-out test
set, one model loaded at a time) against the baseline's analytical-null result
(0.0038) and the closed-form chance floor k/(2D-k).

Run:
    python src/sae_hidden/stability_hidden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import utils
from autoencoder.sae_module import SAEManager
from sae_hidden.reports import md_table, write_report
from sae_hidden.train_hidden import hidden_sae_config

log = utils.setup_logging(__name__)

BASELINE_JACCARD = 0.0038  # ML-AUDIT-2026-06-25: 512-d projected SAE at chance floor


def run() -> Path:
    test_emb = utils.load_tensor(config.paths.hidden_test_embeddings_path)
    model_dirs = [
        config.paths.hidden_models_dir / f"sae_seed{s}" for s in config.training.seeds
    ]
    missing = [d for d in model_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing SAE model dirs: {missing}. Run train_hidden.py first."
        )

    log.info(
        f"Computing cross-seed Jaccard over {len(model_dirs)} seeds on "
        f"{test_emb.shape[0]} held-out samples..."
    )
    result = SAEManager.compute_stability(
        model_dirs=model_dirs,
        embeddings=test_emb,
        config=hidden_sae_config(),
        n=config.sae_hidden.k,
    )

    jmat = result["jaccard_matrix"]
    mean_j = result["mean_jaccard"]
    std_j = result["std_jaccard"]
    k = config.sae_hidden.k
    D = config.sae_hidden.dict_size
    chance_floor = k / (2 * D - k)  # expected Jaccard of two independent k-subsets of D

    out_path = config.paths.hidden_results_dir / "stability_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "seeds": list(config.training.seeds),
                "k": k,
                "dict_size": D,
                "mean_jaccard": mean_j,
                "std_jaccard": std_j,
                "analytical_chance_floor": chance_floor,
                "baseline_512d_jaccard": BASELINE_JACCARD,
                "jaccard_matrix": jmat.tolist(),
            },
            f,
            indent=2,
        )

    # Per-pair table (upper triangle)
    n = len(model_dirs)
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_rows.append(
                [f"{config.training.seeds[i]}-{config.training.seeds[j]}",
                 f"{jmat[i, j]:.4f}"]
            )

    lift_vs_baseline = mean_j / BASELINE_JACCARD if BASELINE_JACCARD else float("inf")
    verdict = (
        "ABOVE chance floor — features are reproducible across seeds."
        if mean_j > 2 * chance_floor
        else "near/at chance floor — features still seed-dependent (non-identifiable)."
    )
    summary = (
        f"Mean cross-seed Jaccard = {mean_j:.4f} (chance floor {chance_floor:.4f}, "
        f"baseline 512-d {BASELINE_JACCARD:.4f}). {verdict} "
        f"Lift over baseline: {lift_vs_baseline:.1f}x."
    )
    sections = [
        (
            "Stability vs references",
            md_table(
                ["metric", "value"],
                [
                    ["mean Jaccard", f"{mean_j:.4f}"],
                    ["std Jaccard", f"{std_j:.4f}"],
                    ["analytical chance floor (k/(2D-k))", f"{chance_floor:.4f}"],
                    ["baseline 512-d Jaccard", f"{BASELINE_JACCARD:.4f}"],
                    ["lift over baseline", f"{lift_vs_baseline:.1f}x"],
                    ["k / dict_size", f"{k} / {D}"],
                ],
            ),
        ),
        (
            "Per-seed-pair Jaccard",
            md_table(["seed pair", "Jaccard"], pair_rows),
        ),
        (
            "Interpretation",
            f"Mean Jaccard {mean_j:.4f} vs chance floor {chance_floor:.4f}. "
            f"{verdict} If mean Jaccard sits near the floor, the 768-d SAE is still "
            f"non-identifiable at this data scale (cf. M-002) — consider a smaller "
            f"dict_size or more data, not more seeds.",
        ),
    ]
    report_path = config.paths.hidden_results_dir / "REPORT_stability.md"
    write_report(report_path, "Path A — Cross-seed Stability (768-d)", sections, summary)
    log.info(f"Report: {report_path}")
    return out_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
