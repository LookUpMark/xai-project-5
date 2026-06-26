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
import math
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
        else "near/at chance floor — features still seed-dependent. "
        "NOTE: this slot-wise index Jaccard is NOT permutation-invariant and "
        "cannot by itself establish non-identifiability (see REPORT_stability_matched.md)."
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


def run_matched() -> Path:
    """Literature-aligned (permutation-invariant) cross-seed feature agreement.

    Pairs each feature with its most cosine-similar decoder direction across seeds
    (decoder-cosine matching), with a random-pairing permutation null. Fixes the
    slot-wise Jaccard's inability to show identifiability (F-001). Result is framed
    as weak vs strong universality (Lan et al. 2024; Leask et al. 2025).
    """
    model_dirs = [
        config.paths.hidden_models_dir / f"sae_seed{s}" for s in config.training.seeds
    ]
    missing = [d for d in model_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing SAE model dirs: {missing}. Run train_hidden.py first."
        )

    cfg = config.sae_hidden
    log.info(
        f"Decoder-cosine matched stability over {len(model_dirs)} seeds "
        f"(isotropic null n_perm={cfg.n_perm})..."
    )
    result = SAEManager.compute_stability_matched(
        model_dirs=model_dirs,
        config=hidden_sae_config(),
        n_perm=cfg.n_perm,
        thresholds=cfg.match_thresholds,
        seed=0,
    )

    out_path = config.paths.hidden_results_dir / "stability_matched.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    obs = result["mean_best_match_cosine"]
    null = result["null_mean"]
    p = result["p_value"]
    ratio = obs / null if null > 0 else float("inf")
    frac09 = result["mean_frac_matched_0.9"]
    frac07 = result["mean_frac_matched_0.7"]
    # Analytical sanity anchor: E[max cosine] of D random unit vectors in d dims.
    d_dim, D = cfg.activation_dim, cfg.dict_size
    anchor = (1.0 / math.sqrt(d_dim)) * math.sqrt(2.0 * math.log(D))
    if frac09 >= 0.10:
        verdict = (
            f"STRONG universality: {frac09:.1%} of features match at ≥0.9 cosine "
            f"across seeds — features reproduce near-identically (up to permutation)."
        )
    elif p < 0.05 and obs > 1.5 * null:
        verdict = (
            f"WEAK universality (p={p:.3f}, obs/null={ratio:.1f}x): decoder subspaces "
            f"share structure well above chance, but only {frac09:.1%} match ≥0.9 "
            f"({frac07:.1%} ≥0.7) — no strong feature-level reproducibility."
        )
    else:
        verdict = (
            f"AT null (p={p:.3f}, obs/null={ratio:.1f}x) — no significant feature "
            f"agreement; non-identifiable at this data scale (M-002)."
        )

    summary = (
        f"Mean best-match cosine = {obs:.4f} (isotropic null {null:.4f}, "
        f"p={p:.3f}). {verdict} Analytical random-anchor ≈ {anchor:.3f}. "
        f" Cf. slot-wise Jaccard ({BASELINE_JACCARD}-class) which cannot show this."
    )
    pair_rows = [
        [pr["pair"], f"{pr['mean_best_match_cosine']:.4f}", f"{pr['null_mean']:.4f}",
         f"{pr['p_value']:.3f}",
         f"{pr['frac_matched_0.7']:.3f}", f"{pr['frac_matched_0.9']:.3f}",
         f"{pr['frac_mutual_1to1']:.3f}"]
        for pr in result["pairs"]
    ]
    thr_rows = [
        [f"mean frac matched ≥{t}", f"{result[f'mean_frac_matched_{t}']:.4f}"]
        for t in cfg.match_thresholds
    ]
    sections = [
        (
            "Headline metrics",
            md_table(
                ["metric", "value"],
                [
                    ["mean best-match cosine", f"{obs:.4f}"],
                    ["permutation null mean", f"{null:.4f}"],
                    ["observed / null", f"{ratio:.2f}x"],
                    ["p-value (P(null≥obs))", f"{p:.4f}"],
                    ["min p-value across pairs", f"{result['min_p_value']:.4f}"],
                    ["mean frac mutual 1-to-1", f"{result['mean_frac_mutual_1to1']:.4f}"],
                    ["analytical random anchor", f"{anchor:.4f}"],
                    ["dict_size / activation_dim", f"{D} / {d_dim}"],
                ],
            ),
        ),
        ("Matched-fraction thresholds", md_table(["metric", "value"], thr_rows)),
        ("Per-pair results", md_table(
            ["pair", "best-match", "null", "p", "frac≥0.7", "frac≥0.9", "mutual1-1"],
            pair_rows,
        )),
        (
            "Interpretation",
            f"{summary}\n\n"
            f"**Metric**: decoder-cosine matching (each feature paired with its "
            f"most similar decoder direction across seeds) + isotropic random-vector "
            f"null. Unlike slot-wise index Jaccard, this is permutation-invariant in "
            f"the matching step — the property the F-001 metric lacked. NB: a row-"
            f"shuffle null is degenerate for max-cosine (max-over-columns is "
            f"permutation-invariant), so the null uses independent random unit "
            f"vectors; it does not control for data-manifold concentration, hence "
            f"the ≥0.9 matched-fraction is the concentration-robust signal.\n\n"
            f"**Framing** (Lan et al. 2024; Leask et al. 2025): cross-seed SAEs are "
            f"*expected* to share at most a subspace (weak universality), rarely "
            f"identical features (strong universality). An observed/null ratio well "
            f"above 1 with p<0.05 ⇒ weak universality present; at-null with p>0.05 "
            f"⇒ genuine non-identifiability at this data scale (M-002), now measured "
            f"rather than asserted.\n\n"
            f"Refs: Bricken 2023; Lan 2024 (arXiv:2410.06981); Leask 2025 "
            f"(arXiv:2502.04878); Kriegeskorte 2008. See "
            f"`docs/design/LITERATURE-SAE-STABILITY.md`.",
        ),
    ]
    report_path = config.paths.hidden_results_dir / "REPORT_stability_matched.md"
    write_report(
        report_path, "Path A — Matched (Permutation-Invariant) Stability", sections, summary
    )
    log.info(f"Report: {report_path}")
    return out_path


def main() -> None:
    run()
    run_matched()


if __name__ == "__main__":
    main()
