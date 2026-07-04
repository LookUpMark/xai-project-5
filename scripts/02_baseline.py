"""02_baseline.py — orchestrate the full baseline (512-d SAE) pipeline.

Thin driver over the existing ``src/autoencoder`` stage ``run()`` / ``main()``
functions (each writes its own JSON). Adds: optional hyperparameter overrides,
``--tag`` isolation, and an aggregate ``REPORT_run.md``.

Reuses stages unchanged via ``autoencoder.variant.baseline_variant`` (swaps
``config.sae`` + ``config.paths.{models,results,figures}_dir`` for the chosen
variant, restores on exit). The default run writes results to
``results/baseline/`` (models stay at the canonical ``models/sae_seed{N}``), so
``--skip-train`` reuses an existing retrain with zero waste.

Usage:
    python scripts/02_baseline.py                          # full (train + stages)
    python scripts/02_baseline.py --skip-train             # regen naming/stability/explain from cached models
    python scripts/02_baseline.py --dict-size 1024 --k 16 --tag sm1024
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/  -> config, utils, autoencoder, sae_hidden
sys.path.insert(0, str(_HERE.parent))          # repo root -> xai_datasets

import config
import utils
from autoencoder import concept_naming, generate_explanations, stability_analysis, train_sae
from autoencoder.variant import baseline_variant
from sae_hidden.reports import md_table, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the baseline (512-d SAE) pipeline end-to-end.")
    p.add_argument("--skip-train", action="store_true", help="reuse cached trained models")
    p.add_argument("--dict-size", type=int, default=None, help="override SAEConfig.dict_size")
    p.add_argument("--k", type=int, default=None, help="override SAEConfig.k")
    p.add_argument("--steps", type=int, default=None, help="override SAEConfig.steps")
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="isolate to models/sae_baseline_{tag} + results/sae_baseline_{tag}",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=config.active_dataset.name,
        help=(
            f"Active dataset (default: {config.active_dataset.name}); re-routes "
            "embedding paths to embeddings/<dataset>/. Must be a key in "
            "xai_datasets.spec.DATASETS (e.g. iu_xray, padchest)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Re-route embedding paths to the selected dataset BEFORE any path is read.
    config.select_dataset(args.dataset)
    root = config.paths.project_root
    models_dir = root / "models" / f"sae_baseline_{args.tag}" if args.tag else config.paths.models_dir
    results_dir = root / "results" / f"sae_baseline_{args.tag}" if args.tag else config.paths.baseline_results_dir

    overrides: dict = {}
    if args.dict_size is not None:
        overrides["dict_size"] = args.dict_size
    if args.k is not None:
        overrides["k"] = args.k
    if args.steps is not None:
        overrides["steps"] = args.steps
    sae = replace(config.sae, **overrides) if overrides else None

    print("=" * 64)
    print("  Baseline (512-d) pipeline" + (f" — tag={args.tag}" if args.tag else ""))
    print(f"  models  : {models_dir}")
    print(f"  results : {results_dir}")
    if sae:
        print(f"  override: dict_size={sae.dict_size} k={sae.k} steps={sae.steps}")
    print("=" * 64)

    # Always isolate results into results/baseline (or results/sae_baseline_{tag});
    # models_dir only diverges from the canonical models/ when --tag is set, so the
    # default run reuses cached models/sae_seed{N} (--skip-train) but writes its
    # results into the baseline subdir, keeping results/ root clean.
    swap = {"results_dir": results_dir}
    if args.tag:
        swap["models_dir"] = models_dir

    stages: list[tuple[str, str, float]] = []
    t0 = time.time()
    with baseline_variant(sae=sae, **swap):
        def step(name: str, fn) -> None:
            ts = time.time()
            fn()
            stages.append((name, "ok", time.time() - ts))
            print(f"  [done] {name} ({stages[-1][2]:.0f}s)")

        if not args.skip_train:
            def _train_all_seeds():
                train_sae.prepare_split()  # idempotent; ensures the train/test split exists
                for s in config.training.seeds:
                    train_sae.train_single(s)
            step("train", _train_all_seeds)
        step("naming", concept_naming.run)
        step("stability", stability_analysis.run)
        step("explain", generate_explanations.run)

    total = time.time() - t0
    _write_run_report(args, sae, models_dir, results_dir, stages, total)
    print(f"\nDone in {total:.0f}s. Report: {results_dir / 'REPORT_run.md'}")


def _write_run_report(args, sae, models_dir, results_dir, stages, total):
    cfg = sae or config.sae
    summary = (
        f"Baseline (512-d) run complete in {total:.0f}s. "
        f"dict_size={cfg.dict_size} k={cfg.k} steps={cfg.steps}, "
        f"seeds={list(config.training.seeds)}. Stages wrote JSON to {results_dir}."
    )
    sections = [
        (
            "Run config",
            md_table(
                ["param", "value"],
                [
                    ["tag", args.tag or "—"],
                    ["models dir", str(models_dir)],
                    ["dict_size / k / steps", f"{cfg.dict_size} / {cfg.k} / {cfg.steps}"],
                    ["seeds", ", ".join(map(str, config.training.seeds))],
                    ["device", config.hardware.device],
                ],
            ),
        ),
        ("Stages", md_table(["stage", "status", "seconds"], [[n, s, f"{t:.1f}"] for n, s, t in stages])),
        (
            "Outputs",
            md_table(
                ["artifact", "path"],
                [
                    ["concept names", f"{results_dir}/concept_names.json"],
                    ["stability (jaccard)", f"{results_dir}/stability_analysis.json"],
                    ["stability (matched)", f"{results_dir}/stability_matched.json"],
                    ["explanations", f"{results_dir}/sample_explanations.json"],
                ],
            ),
        ),
        (
            "Reproducibility",
            "\n".join(utils.repro_info([
                ("train_embeddings", config.paths.train_embeddings_path),
                ("test_embeddings", config.paths.test_embeddings_path),
                ("text_vocab_embeddings", config.paths.vocab_embeddings_path),
                ("modality_gap", config.paths.models_dir / "modality_gap.pt"),
                ("primary_model", models_dir / f"sae_seed{config.training.primary_seed}" / "trainer_0" / "ae.pt"),
            ])),
        ),
    ]
    write_report(results_dir / "REPORT_run.md", "Baseline (512-d) — Pipeline Run", sections, summary)


if __name__ == "__main__":
    main()
