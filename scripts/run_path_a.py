"""run_path_a.py — orchestrate the full Path A (SAE on 768-d hidden state) pipeline.

Thin driver over the existing ``src/sae_hidden`` stage ``run()`` functions (each
already writes its own ``REPORT_*.md``). Adds: standard vs augmented variant,
optional hyperparameter overrides, and an aggregate ``REPORT_run.md``.

Reuses stages unchanged via ``sae_hidden.variant.hidden_variant`` (swaps
``config.sae_hidden`` + the hidden I/O paths for the chosen variant, restores
on exit).

Usage:
    python scripts/run_path_a.py                              # standard, full
    python scripts/run_path_a.py --variant augmented          # augmented (slow extract)
    python scripts/run_path_a.py --skip-extract --skip-train  # regen reports from cache
    python scripts/run_path_a.py --dict-size 1024 --k 16 --tag sm1024
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # src/  -> config, utils, sae_hidden
sys.path.insert(0, str(_HERE.parent))          # repo root -> xai_datasets

import config
import utils
from sae_hidden import (
    extract_hidden,
    generate_explanations_hidden,
    naming_hidden,
    stability_hidden,
    train_hidden,
)
from sae_hidden.reports import md_table, write_report
from sae_hidden.variant import hidden_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Path A (768-d SAE) pipeline end-to-end.")
    p.add_argument("--variant", choices=["standard", "augmented"], default="standard")
    p.add_argument("--skip-extract", action="store_true", help="reuse cached embeddings")
    p.add_argument("--skip-train", action="store_true", help="reuse cached trained models")
    p.add_argument("--dict-size", type=int, default=None, help="override SAEHiddenConfig.dict_size")
    p.add_argument("--k", type=int, default=None, help="override SAEHiddenConfig.k")
    p.add_argument("--steps", type=int, default=None, help="override SAEHiddenConfig.steps")
    p.add_argument("--tag", type=str, default=None, help="suffix for models/results dirs")
    return p.parse_args()


def variant_dirs(augmented: bool, tag: str | None) -> tuple[Path, Path, Path]:
    root = config.paths.project_root
    embeddings_dir = root / "embeddings" / ("augmented_hidden" if augmented else "standard_hidden")
    base = "sae_hidden_augmented" if augmented else "sae_hidden"
    if tag:
        base = f"sae_hidden_{tag}"
    return embeddings_dir, root / "models" / base, root / "results" / base


def main() -> None:
    args = parse_args()
    augmented = args.variant == "augmented"
    embeddings_dir, models_dir, results_dir = variant_dirs(augmented, args.tag)

    overrides = {}
    if args.dict_size is not None:
        overrides["dict_size"] = args.dict_size
    if args.k is not None:
        overrides["k"] = args.k
    if args.steps is not None:
        overrides["steps"] = args.steps
    sae_hidden = replace(config.sae_hidden, **overrides) if overrides else None

    print("=" * 64)
    print(f"  Path A pipeline — variant={args.variant}" + (f", tag={args.tag}" if args.tag else ""))
    print(f"  embeddings : {embeddings_dir}")
    print(f"  models     : {models_dir}")
    print(f"  results    : {results_dir}")
    if sae_hidden:
        print(f"  override   : dict_size={sae_hidden.dict_size} k={sae_hidden.k} steps={sae_hidden.steps}")
    print("=" * 64)

    stages: list[tuple[str, str, float]] = []
    t0 = time.time()
    with hidden_variant(
        sae_hidden=sae_hidden,
        embeddings_dir=embeddings_dir,
        models_dir=models_dir,
        results_dir=results_dir,
    ):
        def step(name: str, fn) -> None:
            ts = time.time()
            fn()
            stages.append((name, "ok", time.time() - ts))
            print(f"  [done] {name} ({stages[-1][2]:.0f}s)")

        if not args.skip_extract:
            step("extract", lambda: extract_hidden.run(augmented=augmented))
        if not args.skip_train:
            step("train", train_hidden.run)
        step("naming", naming_hidden.run)
        step("stability", stability_hidden.main)  # writes both slot-wise + matched reports
        step("explain", generate_explanations_hidden.run)

    total = time.time() - t0
    _write_run_report(args, sae_hidden, embeddings_dir, models_dir, results_dir, stages, total)
    print(f"\nDone in {total:.0f}s. Aggregate report: {results_dir / 'REPORT_run.md'}")


def _write_run_report(args, sae_hidden, embeddings_dir, models_dir, results_dir, stages, total):
    cfg = sae_hidden or config.sae_hidden
    summary = (
        f"Path A ({args.variant}) run complete in {total:.0f}s. "
        f"dict_size={cfg.dict_size} k={cfg.k} steps={cfg.steps}, "
        f"seeds={list(config.training.seeds)}. Each stage wrote its own REPORT_*.md "
        f"under {results_dir}."
    )
    stage_reports = sorted(
        p.name for p in results_dir.glob("REPORT_*.md") if p.name != "REPORT_run.md"
    )
    sections = [
        (
            "Run config",
            md_table(
                ["param", "value"],
                [
                    ["variant", args.variant],
                    ["tag", args.tag or "—"],
                    ["embeddings dir", str(embeddings_dir)],
                    ["models dir", str(models_dir)],
                    ["dict_size / k / steps", f"{cfg.dict_size} / {cfg.k} / {cfg.steps}"],
                    ["seeds", ", ".join(map(str, config.training.seeds))],
                    ["device", config.hardware.device],
                ],
            ),
        ),
        (
            "Stages",
            md_table(["stage", "status", "seconds"], [[n, s, f"{t:.1f}"] for n, s, t in stages]),
        ),
        (
            "Stage reports",
            "\n".join(f"- `{name}`" for name in stage_reports) or "_(none)_",
        ),
        (
            "Reproducibility",
            "\n".join(utils.repro_info([
                ("train_embeddings", config.paths.hidden_train_embeddings_path),
                ("test_embeddings", config.paths.hidden_test_embeddings_path),
                ("text_vocab_embeddings", config.paths.vocab_embeddings_path),
                ("modality_gap", config.paths.models_dir / "modality_gap.pt"),
                ("primary_model", models_dir / f"sae_seed{config.training.primary_seed}" / "trainer_0" / "ae.pt"),
            ])),
        ),
    ]
    write_report(
        results_dir / "REPORT_run.md",
        f"Path A — Pipeline Run ({args.variant})",
        sections,
        summary,
    )


if __name__ == "__main__":
    main()
