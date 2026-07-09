"""run_consensus.py — Cross-seed consensus clustering with a shuffle-null (Gap 1).

Pools the live decoder rows of the 5 baseline SAE seeds, clusters them by
reciprocal cosine (connected components at threshold ``tau``), and asks whether
features reappear across seeds more than chance: ``consensus@>=m/5`` is the
fraction of live rows sitting in a cluster that spans at least ``m`` of the 5
seeds. The null permutes the seed tags across the (fixed) clustering, holding
cluster geometry constant, and recomputes ``consensus@>=m``; the p-value is the
fraction of null permutations reaching the observed consensus.

CLI replacement for the deleted ``notebooks/autoencoder/ablation/00_consensus``
notebook, run on the current parity baseline (D=2048, k=32) instead of the
historical D=4096 one. Decoder-only: needs no embeddings, no vocabulary, no
GPU. A Hungarian direction-matching rate (cosine >= tau, bipartite optimum) is
reported alongside as cross-seed context -- it is a *different* quantity from
the slot-wise index Jaccard and is not a correction of it.

Output: ``results/<dataset>/baseline/consensus.json``.

Usage:
    python scripts/run_consensus.py                # IU X-Ray, 5 seeds, tau=0.90
    python scripts/run_consensus.py --tau 0.85 --m 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/ -> config, utils, autoencoder

import config
from autoencoder.sae_module import SAEManager

DEAD_THRESHOLD = 1e-8   # matches _DEFAULTS['dead_threshold'] used by name_concepts
# Full sweep: high tau -> direction-identity (degenerate, 0 multi at D=2048);
# mid tau -> informative cross-seed test; low tau -> null-saturated.
TAU_GRID = (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
N_PERM = 200
NULL_SEED = 2026


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="iu_xray", help="dataset key (default iu_xray)")
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="seeds to pool (default: config.training.seeds, all 5)",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.70,
        help="cosine threshold for connected components (default 0.70: highest "
        "non-degenerate threshold at D=2048; tau>=0.80 yields ~0 multi-clusters)",
    )
    p.add_argument("--m", type=int, default=4, help="consensus threshold: cluster spans >= m seeds")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--null-seed", type=int, default=NULL_SEED)
    return p.parse_args()


def pool_live_decoders(seeds: list[int], device: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load each seed's decoder, keep live rows (norm >= DEAD_THRESHOLD), L2-normalize,
    concatenate. Returns (W_live (N,512) float64, tags (N,), per_seed_nlive)."""
    rows: list[np.ndarray] = []
    tags: list[int] = []
    per_seed_nlive: dict[int, int] = {}
    dict_size = config.sae.dict_size

    for seed in seeds:
        model_dir = config.paths.models_dir / f"sae_seed{seed}"
        if not model_dir.exists():
            print(f"  [skip] seed {seed}: {model_dir} missing")
            continue
        mgr = SAEManager({"device": device})
        mgr.load(model_dir)
        W = mgr.get_decoder_weights()  # (dict_size, 512)
        norms = W.norm(dim=1)
        live_mask = norms >= DEAD_THRESHOLD
        n_live = int(live_mask.sum().item())
        n_dead = dict_size - n_live
        W_live = F.normalize(W[live_mask], dim=1).cpu().numpy().astype(np.float64)
        rows.append(W_live)
        tags.extend([seed] * n_live)
        per_seed_nlive[seed] = n_live
        del mgr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  seed {seed:>3}: {n_live} live / {n_dead} dead ({100*n_dead/dict_size:.1f}% dead)")

    if not rows:
        raise SystemExit("no seed models found -- nothing to report")
    W = np.concatenate(rows, axis=0)
    return W, np.array(tags, dtype=np.int64), per_seed_nlive


def cluster_cosine(W: np.ndarray, tau: float) -> tuple[np.ndarray, dict]:
    """Connected components of the reciprocal-cosine graph at threshold tau."""
    N = W.shape[0]
    cos = W @ W.T
    np.fill_diagonal(cos, -1.0)
    adj = sparse.csr_matrix(cos > tau)
    n_comp, labels = connected_components(csgraph=adj, directed=False)
    sizes = np.bincount(labels)
    multi = sizes[sizes > 1]
    # mean intra-cluster cosine over multi-member clusters
    cohesion_terms = []
    for lab in np.where(sizes > 1)[0]:
        idx = np.where(labels == lab)[0]
        sub = cos[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        if iu[0].size:
            cohesion_terms.append(sub[iu].mean())
    cohesion = float(np.mean(cohesion_terms)) if cohesion_terms else float("nan")
    meta = {
        "n_components": int(n_comp),
        "n_multi": int(multi.size),
        "max_size": int(sizes.max()),
        "mean_intra_cohesion": cohesion,
    }
    return labels, meta


def consensus_at_m(labels: np.ndarray, tags: np.ndarray, m: int) -> float:
    """Fraction of live rows in a cluster spanning >= m distinct seeds."""
    row_seedcount = np.array(
        [len(np.unique(tags[labels == lab])) for lab in labels]
    )
    return float((row_seedcount >= m).mean())


def clusters_by_seed_count(labels: np.ndarray, tags: np.ndarray, n_seeds: int) -> dict:
    """Histogram: how many clusters span exactly k seeds, for k=1..n_seeds."""
    cluster_ids = np.unique(labels)
    seed_counts = [len(np.unique(tags[labels == lab])) for lab in cluster_ids]
    hist = np.bincount(seed_counts, minlength=n_seeds + 1)[1:]
    return {str(k): int(hist[k - 1]) for k in range(1, n_seeds + 1)}


def hungarian_match(W: np.ndarray, tags: np.ndarray, seeds: list[int], tau: float) -> dict:
    """Per-seed-pair bipartite-optimum direction match rate (cosine >= tau).
    Different quantity from slot-wise index Jaccard; reported as context."""
    # pad each seed's live matrix to a common block layout: live rows carry their
    # normalized direction; per-pair cosine is computed only over live x live.
    per_seed = {}
    for seed in seeds:
        mask = tags == seed
        per_seed[seed] = (W[mask], int(mask.sum()))

    rates = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            si, sj = seeds[i], seeds[j]
            Wi, ni = per_seed[si]
            Wj, nj = per_seed[sj]
            cos_ij = Wi @ Wj.T  # (ni, nj)
            row, col = linear_sum_assignment(-cos_ij)
            kept = int((cos_ij[row, col] >= tau).sum())
            denom = min(ni, nj)
            rate = kept / denom if denom else 0.0
            rates.append({"pair": [si, sj], "matched": kept, "denom": denom, "rate": rate})

    mean_rate = float(np.mean([r["rate"] for r in rates])) if rates else float("nan")
    return {"tau": tau, "mean_rate": mean_rate, "n_pairs": len(rates), "per_pair": rates}


def main() -> None:
    args = parse_args()
    config.select_dataset(args.dataset)
    seeds = args.seeds if args.seeds is not None else list(config.training.seeds)
    n_seeds = len(seeds)

    print("=" * 64)
    print("  Consensus clustering shuffle-null -- cross-seed stability (Gap 1)")
    print(f"  dataset : {args.dataset}  | seeds={seeds}")
    print(f"  models  : {config.paths.models_dir}/sae_seed{{N}}")
    print(f"  tau={args.tau} | consensus@>={args.m}/{n_seeds} | n_perm={args.n_perm}")
    print("=" * 64)

    # 1) pool live decoder rows across seeds ---------------------------------
    W, tags, per_seed_nlive = pool_live_decoders(seeds, config.hardware.device)
    n_live_total = W.shape[0]
    print(f"  pooled live rows: {n_live_total}  | per-seed: {per_seed_nlive}")

    # 2) tau sweep, then fix the headline tau --------------------------------
    tau_sweep: dict[str, dict] = {}
    labels_tau: dict[float, np.ndarray] = {}
    for tau in TAU_GRID:
        lab, meta = cluster_cosine(W, tau)
        tau_sweep[f"{tau:.2f}"] = meta
        labels_tau[tau] = lab
        print(f"  tau={tau:.2f}: {meta['n_components']:>5} components, "
              f"{meta['n_multi']:>4} multi, max={meta['max_size']:>4}, "
              f"cohesion={meta['mean_intra_cohesion']:.3f}")

    tau = args.tau
    if tau not in labels_tau:
        lab, meta = cluster_cosine(W, tau)
        tau_sweep[f"{tau:.2f}"] = meta
        labels_tau[tau] = lab
    labels = labels_tau[tau]

    # 3) observed consensus + reappearance histogram -------------------------
    hist = clusters_by_seed_count(labels, tags, n_seeds)
    consensus_at_3 = consensus_at_m(labels, tags, 3)
    consensus_at_4 = consensus_at_m(labels, tags, 4)
    observed = consensus_at_m(labels, tags, args.m)
    print(f"  reappearance (clusters by #seeds): {hist}")
    print(f"  consensus@>=3/5: {100*consensus_at_3:.2f}% | @>=4/5: {100*consensus_at_4:.2f}%")
    print(f"  observed consensus@>={args.m}/{n_seeds}: {100*observed:.2f}%")

    # 4) shuffle-null: permute seed tags, hold clusters fixed ----------------
    rng = np.random.default_rng(args.null_seed)
    null_samples = np.empty(args.n_perm, dtype=np.float64)
    for b in range(args.n_perm):
        null_samples[b] = consensus_at_m(labels, rng.permutation(tags), args.m)
    null_mean = float(null_samples.mean())
    p_value = float((null_samples >= observed).mean())
    gap = observed - null_mean
    verdict = (
        "real (observed >> chance)" if (gap > 0 and p_value < 0.05)
        else "no signal above chance (observed ~= null)"
    )
    print(f"  shuffle-null mean: {100*null_mean:.2f}% | p-value: {p_value:.4f} | {verdict}")

    # 5) Hungarian direction-matching (context, different quantity) ----------
    hung = hungarian_match(W, tags, seeds, tau)
    print(f"  Hungarian match (cosine>={tau}, direction-space): {100*hung['mean_rate']:.2f}%")

    # 6) persist -------------------------------------------------------------
    out = {
        "dataset": args.dataset,
        "metric": "cross-seed consensus clustering shuffle-null (Gap 1 stability)",
        "config": {
            "dict_size": config.sae.dict_size,
            "k": config.sae.k,
            "activation_dim": config.sae.activation_dim,
            "seeds": seeds,
            "tau": tau,
            "m": args.m,
            "n_perm": args.n_perm,
            "null_seed": args.null_seed,
            "dead_threshold": DEAD_THRESHOLD,
        },
        "pooling": {
            "n_live_total": n_live_total,
            "per_seed_nlive": per_seed_nlive,
        },
        "tau_sweep": tau_sweep,
        "consensus": {
            "tau": tau,
            "observed_at_3": consensus_at_3,
            "observed_at_4": consensus_at_4,
            f"observed_at_{args.m}": observed,
            "clusters_by_seed_count": hist,
        },
        "shuffle_null": {
            "null_mean": null_mean,
            "p_value": p_value,
            "gap": gap,
            "verdict": verdict,
            "n_perm": args.n_perm,
            "null_seed": args.null_seed,
        },
        "hungarian": hung,
        "note": (
            "Replaces the historical D=4096 00_consensus notebook; run on the "
            "current parity baseline (D=2048, k=32). Consensus@>=m is the fraction "
            "of pooled live decoder rows in a connected-component cluster (cosine "
            f">= {tau}) spanning >= m of the {n_seeds} seeds; the null permutes the "
            "seed tags across the fixed clustering. The Hungarian rate is a "
            "direction-space match (different from the slot-wise index Jaccard) "
            "and is reported as context, not a correction."
        ),
    }

    out_dir = config.paths.baseline_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "consensus.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  consensus@>={args.m}/{n_seeds}: {100*observed:.2f}%  | null {100*null_mean:.2f}%")
    print(f"  shuffle-null p-value: {p_value:.4f}  ({verdict})")
    print(f"  -> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
