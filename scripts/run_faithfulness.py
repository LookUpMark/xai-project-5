"""run_faithfulness.py — Concept faithfulness vs clinical labels (Gap 2).

Point-biserial correlation of SAE feature activations against IU X-Ray
MeSH/Problems labels, calibrated against a triple null (analytical standard
error, per-feature shuffle-null p95, Benjamini-Hochberg FDR). CLI replacement
for the deleted ``notebooks/autoencoder/ablation/05_faithfulness`` notebook,
run on the current parity baseline (D=2048, k=32) instead of the historical
D=4096 one.

A feature is "faithful" if its activation pattern correlates with a clinical
label beyond chance: ``R = A_z^T Y_z / N`` (point-biserial), and for feature
``i`` we take ``max_j |R_ij|`` over the prevalence-filtered label set.

Output: ``results/<dataset>/baseline/faithfulness.json`` with per-seed and
aggregated (mean +/- std) statistics.

Usage:
    python scripts/run_faithfulness.py                       # IU X-Ray, 5 seeds
    python scripts/run_faithfulness.py --seeds 0             # primary seed only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/ -> config, utils, autoencoder
sys.path.insert(0, str(_HERE.parent))          # repo root -> xai_datasets

import config
import utils
from autoencoder.sae_module import SAEManager

MIN_PREV = 10      # a label needs >= MIN_PREV images to be correlatable
N_PERM = 200       # shuffle-null permutations
NULL_SEED = 0


def parse_labels(mesh: str, problems: str) -> set[str]:
    """MeSH/Problems -> set of base terms (split ';', base on '/', lower, excl 'normal')."""
    out: set[str] = set()
    for field in (mesh, problems):
        if not field:
            continue
        for item in field.split(";"):
            base = item.strip().split("/")[0].strip().lower()
            if base and base != "normal":
                out.add(base)
    return out


def zscore_cols(M: np.ndarray) -> np.ndarray:
    mu = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    return q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="iu_xray", help="dataset key (default iu_xray)")
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="seeds to evaluate (default: config.training.seeds, all 5)",
    )
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--min-prev", type=int, default=MIN_PREV)
    p.add_argument("--null-seed", type=int, default=NULL_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset != "iu_xray":
        # Faithfulness is defined against IU X-Ray MeSH/Problems labels only.
        raise SystemExit(
            f"faithfulness requires IU X-Ray labels; dataset={args.dataset} unsupported."
        )

    config.select_dataset(args.dataset)
    root = config.paths.project_root
    reports_csv = root / "data" / "iu_xray" / "reports" / "indiana_reports.csv"
    proj_csv = root / "data" / "iu_xray" / "reports" / "indiana_projections.csv"

    seeds = args.seeds if args.seeds is not None else list(config.training.seeds)

    print("=" * 64)
    print("  Faithfulness (point-biserial vs MeSH/Problems) — Gap 2")
    print(f"  dataset : {args.dataset}  | seeds={seeds}")
    print(f"  models  : {config.paths.models_dir}/sae_seed{{N}}")
    print(f"  reports : {reports_csv}")
    print("=" * 64)

    # --- clinical labels ----------------------------------------------------
    uid2labels: dict[int, set[str]] = {}
    with open(reports_csv) as f:
        for row in csv.DictReader(f):
            uid2labels[int(row["uid"])] = parse_labels(
                row.get("MeSH", ""), row.get("Problems", "")
            )

    file2uid: dict[str, int] = {}
    with open(proj_csv) as f:
        for row in csv.DictReader(f):
            file2uid[row["filename"]] = int(row["uid"])

    def basename_to_uid(bn: str) -> int | None:
        if bn in file2uid:
            return file2uid[bn]
        m = re.match(r"(\d+)_", bn)
        return int(m.group(1)) if m else None

    # --- test embeddings + sidecar (row-aligned image ids) -----------------
    test_emb = utils.load_tensor(config.paths.test_embeddings_path, device="cpu")
    sidecar = config.paths.test_embeddings_path.parent / "test_image_ids.json"
    with open(sidecar) as f:
        test_ids: list[str] = json.load(f)
    assert test_emb.shape[0] == len(test_ids), (
        f"row mismatch: test_emb {test_emb.shape[0]} vs sidecar {len(test_ids)}"
    )

    # aligned binary label matrix Y (rows in test-set order)
    all_terms = sorted(set().union(*uid2labels.values()))
    term2idx = {t: i for i, t in enumerate(all_terms)}
    Y = np.zeros((len(test_ids), len(all_terms)), dtype=np.float32)
    missing = 0
    for i, bn in enumerate(test_ids):
        uid = basename_to_uid(bn)
        if uid is None or uid not in uid2labels:
            missing += 1
            continue
        for t in uid2labels[uid]:
            Y[i, term2idx[t]] = 1.0

    # prevalence filter
    label_freq = Y.sum(0)
    keep = np.where(label_freq >= args.min_prev)[0]
    Yf = Y[:, keep]
    labels_f = [all_terms[j] for j in keep]

    N = len(test_ids)
    se = 1.0 / math.sqrt(N)

    print(
        f"  N_test={N} | labels total={len(all_terms)} | "
        f"after prevalence(>={args.min_prev})={len(keep)} | missing_uid={missing}"
    )
    print(f"  analytic null SE(r) = {se:.4f}")

    # --- per-seed faithfulness ---------------------------------------------
    per_seed: list[dict] = []
    for seed in seeds:
        model_dir = config.paths.models_dir / f"sae_seed{seed}"
        if not model_dir.exists():
            print(f"  [skip] seed {seed}: {model_dir} missing")
            continue

        mgr = SAEManager({"device": config.hardware.device})
        mgr.load(model_dir)
        A = mgr.encode(test_emb.to(config.hardware.device)).cpu().numpy().astype(np.float32)
        del mgr

        A_z = zscore_cols(A)
        Yf_z = zscore_cols(Yf)
        corr = (A_z.T @ Yf_z) / N
        abs_corr = np.abs(corr)
        per_feat_max = abs_corr.max(1)
        best_label = abs_corr.argmax(1)

        live = (A != 0).any(axis=0)
        live_idx = np.where(live)[0]
        n_live = int(live.sum())
        D = A.shape[1]

        # (a) analytic p (uncorrected), then BH-FDR over live x labels
        p_unc = 2 * norm.sf(per_feat_max / se)
        flat_p = p_unc[live_idx][:, None] * np.ones((1, len(labels_f)))
        q = bh_fdr(flat_p.ravel()).reshape(n_live, len(labels_f))
        feat_fdr_signif = (q < 0.05).any(1)

        # (b) shuffle-null per-feature (live features): permute labels B times,
        #     take p95 of the null max-|r| distribution per feature.
        rng = np.random.default_rng(args.null_seed)
        null_max = np.zeros((args.n_perm, n_live), dtype=np.float32)
        for b in range(args.n_perm):
            perm = rng.permutation(N)
            corr_b = (A_z.T @ Yf_z[perm]) / N
            null_max[b] = np.abs(corr_b).max(1)[live_idx]
        null_p95 = np.percentile(null_max, 95, axis=0)
        shuffle_signif = per_feat_max[live_idx] > null_p95

        frac_shuffle = float(shuffle_signif.mean())
        frac_fdr = float(feat_fdr_signif.mean())
        frac_strong = float((per_feat_max[live_idx] > 0.30).mean())

        # top faithful features (by live |r|)
        order = np.argsort(per_feat_max[live_idx])[::-1][:10]
        top = [
            {
                "feat": int(live_idx[o]),
                "abs_r": float(per_feat_max[live_idx[o]]),
                "label": labels_f[int(best_label[live_idx[o]])],
            }
            for o in order
        ]

        seed_rec = {
            "seed": seed,
            "n_live": n_live,
            "n_dead": D - n_live,
            "frac_live": float(live.mean()),
            "frac_faithful_shuffle_p95": frac_shuffle,
            "frac_faithful_fdr05": frac_fdr,
            "frac_strong_abs_r_gt_030": frac_strong,
            "max_abs_r": float(per_feat_max[live_idx].max()),
            "median_null_p95": float(np.median(null_p95)),
            "top_features": top,
        }
        per_seed.append(seed_rec)
        print(
            f"  [seed {seed}] live={n_live}/{D} ({100*live.mean():.1f}%) | "
            f"faithful(shuffle p95)={100*frac_shuffle:.1f}% | "
            f"max|r|={seed_rec['max_abs_r']:.3f}"
        )

    if not per_seed:
        raise SystemExit("no seed models found — nothing to report")

    # --- aggregate ----------------------------------------------------------
    def _agg(key: str) -> dict:
        vals = np.array([r[key] for r in per_seed], dtype=np.float64)
        return {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}

    aggregate = {
        "frac_faithful_shuffle_p95": _agg("frac_faithful_shuffle_p95"),
        "frac_faithful_fdr05": _agg("frac_faithful_fdr05"),
        "frac_strong_abs_r_gt_030": _agg("frac_strong_abs_r_gt_030"),
        "max_abs_r": _agg("max_abs_r"),
    }

    out = {
        "dataset": args.dataset,
        "metric": "point-biserial faithfulness (Gap 2)",
        "config": {
            "dict_size": config.sae.dict_size,
            "k": config.sae.k,
            "seeds": seeds,
            "n_test": N,
            "n_labels_total": len(all_terms),
            "n_labels_prevalence_filtered": int(len(keep)),
            "min_prev": args.min_prev,
            "n_perm": args.n_perm,
            "analytic_null_se": se,
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "note": (
            "Replaces the historical D=4096 ablation-05 notebook; run on the "
            "current parity baseline (D=2048, k=32). Faithful = live feature "
            "whose max point-biserial |r| exceeds the per-feature shuffle-null "
            "p95 (200 perms)."
        ),
    }

    out_dir = config.paths.baseline_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "faithfulness.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  faithful (shuffle p95): {100*aggregate['frac_faithful_shuffle_p95']['mean']:.1f}% "
          f"± {100*aggregate['frac_faithful_shuffle_p95']['std']:.1f}")
    print(f"  max |r|              : {aggregate['max_abs_r']['mean']:.3f} "
          f"± {aggregate['max_abs_r']['std']:.3f}")
    print(f"  strong |r|>0.30      : {100*aggregate['frac_strong_abs_r_gt_030']['mean']:.1f}% "
          f"± {100*aggregate['frac_strong_abs_r_gt_030']['std']:.1f}")
    print(f"  -> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
