"""train_hidden.py — Path A: train Top-K SAEs on the 768-d hidden state.

Trains one SAE per seed on the 768-d train split, evaluates each on the held-out
768-d test split. Audit-corrected hyperparameters live in config.SAEHiddenConfig
(steps 8k, lr 5e-5, dict_size 2048 — see ML-AUDIT-2026-06-25 M-002/M-006).

Run:
    python src/sae_hidden/train_hidden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import utils
from autoencoder.sae_module import SAEManager
from sae_hidden.reports import md_table, write_report

log = utils.setup_logging(__name__)


def hidden_sae_config() -> dict:
    """SAEManager config dict for the 768-d Path A SAE."""
    cfg = utils.dataclass_to_dict(config.sae_hidden)
    cfg["device"] = config.hardware.device
    return cfg


def train_single(seed: int) -> dict:
    """Train one SAE; return per-seed sanity metrics (measured on test_768)."""
    log.info(f"Training Path A SAE (768-d), seed={seed}")
    mgr = SAEManager(hidden_sae_config())
    model_dir = mgr.train(
        embeddings_path=config.paths.hidden_train_embeddings_path,
        seed=seed,
        save_dir=config.paths.hidden_models_dir,
        steps=config.sae_hidden.steps,
        batch_size=config.sae_hidden.batch_size,
    )

    # F-014: metrics on the FULL held-out test set — the 256-subset under-reported
    # the dead-feature rate (a feature can miss 256 samples but fire on 1515).
    test_emb = utils.load_tensor(config.paths.hidden_test_embeddings_path)

    mse = mgr.compute_reconstruction_mse(test_emb)
    cosine = mgr.compute_cosine_reconstruction(test_emb)
    sparsity = mgr.compute_sparsity_metrics(test_emb)

    log.info(f"  seed={seed} | MSE={mse:.6f} cos={cosine:.4f} "
             f"dead={sparsity['dead_features_pct']:.1f}% "
             f"util={sparsity['dict_utilization_pct']:.1f}%")
    return {
        "seed": seed,
        "model_dir": str(model_dir),
        "test_mse": mse,
        "test_cosine": cosine,
        **sparsity,
    }


def run() -> list[Path]:
    utils.set_global_seed(config.training.split_seed)

    if not config.paths.hidden_train_embeddings_path.exists():
        raise FileNotFoundError(
            f"768-d train embeddings missing: "
            f"{config.paths.hidden_train_embeddings_path}. Run extract_hidden.py first."
        )

    log.info(
        f"Path A training: seeds={config.training.seeds}, "
        f"activation_dim=768 dict_size={config.sae_hidden.dict_size} "
        f"k={config.sae_hidden.k} steps={config.sae_hidden.steps} lr={config.sae_hidden.lr}"
    )

    results = [train_single(seed) for seed in config.training.seeds]

    # Persist per-seed metrics for downstream consumers (ablation harness, run report).
    metrics_path = config.paths.hidden_results_dir / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Train metrics: {metrics_path}")

    # Report
    report_path = config.paths.hidden_results_dir / "REPORT_training.md"
    rows = [
        [
            r["seed"],
            f"{r['test_mse']:.6f}",
            f"{r['test_cosine']:.4f}",
            f"{r['dead_features_pct']:.1f}",
            f"{r['dict_utilization_pct']:.1f}",
            f"{r['l0_mean']:.1f}",
        ]
        for r in results
    ]
    dead_mean = float(np.mean([r["dead_features_pct"] for r in results]))
    summary = (
        f"Trained {len(results)} 768-d Top-K SAEs (seeds {config.training.seeds}). "
        f"Mean dead-feature rate {dead_mean:.1f}% (baseline 512-d: 40-60%). "
        f"dict_size={config.sae_hidden.dict_size}, steps={config.sae_hidden.steps}, "
        f"lr={config.sae_hidden.lr} (audit-corrected)."
    )
    sections = [
        (
            "Per-seed test metrics",
            md_table(
                ["seed", "test MSE", "test cosine", "dead %", "util %", "L0"],
                rows,
            ),
        ),
        (
            "Hyperparameters (audit-corrected)",
            md_table(
                ["param", "value", "rationale"],
                [
                    ["activation_dim", "768", "pre-projection CLS (Paradigm B)"],
                    ["dict_size", str(config.sae_hidden.dict_size), "M-002: down from 4096"],
                    ["k", str(config.sae_hidden.k), "Top-K"],
                    ["lr", str(config.sae_hidden.lr), "M-006: pinned low"],
                    ["steps", str(config.sae_hidden.steps), "M-006: down from 50k"],
                    ["input", "raw", "no per-sample L2 norm"],
                ],
            ),
        ),
        (
            "Caveat",
            "Reconstruction cosine is near-saturated for any overcomplete SAE on a "
            "low-dim manifold (M-004) — do NOT read it as evidence the SAE 'works'. "
            "Trust cross-seed Jaccard (see REPORT_stability.md) and naming instead.",
        ),
    ]
    write_report(report_path, "Path A — 768-d SAE Training", sections, summary)
    log.info(f"Report: {report_path}")
    return [Path(r["model_dir"]) for r in results]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
