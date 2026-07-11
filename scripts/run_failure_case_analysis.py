"""run_failure_case_analysis.py — Aggregate a dataset's results into a failure-case report.

Reads the artifacts a baseline run produced (concept names, cross-seed stability,
matched stability, sample explanations, judge verdicts) and writes a structured
``REPORT.md`` + figures that frame the results as the **failure cases** the traccia
requires (unstable/noise concepts, naming ≈ random, judge disagreement), with the
root cause (data starvation → non-identifiable factorization) and a forward link
to the multi-dataset plan (PadChest scale test). Numbers feed ``docs/FINDINGS.md``.

Lightweight: JSON/CSV + matplotlib only — no GPU, no model load. Run after a
baseline + judge run for the dataset.

Usage:
    python scripts/run_failure_case_analysis.py                 # active dataset (iu_xray)
    python scripts/run_failure_case_analysis.py --dataset iu_xray
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent))

import config

import pandas as pd

# Headless matplotlib (figures saved, not shown).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_label(checkpoint_name: str) -> str:
    """Human-readable model label from a checkpoint filename."""
    stem = checkpoint_name
    for pre in [".judge_checkpoint_", "judge_checkpoint_"]:
        if stem.startswith(pre):
            stem = stem[len(pre):]
    stem = stem.removesuffix(".json")
    return stem.replace("_", " ")


def _verdict_distribution(records: list[dict]) -> Counter:
    """Verdict counts over non-error records."""
    c: Counter = Counter()
    for r in records:
        if str(r.get("raw_response", "")).startswith("ERROR:"):
            continue
        c[r.get("verdict", "?")] += 1
    return c


# ---------------------------------------------------------------------------
# Sections — each returns (rows, notes) for the report
# ---------------------------------------------------------------------------

def analyze_naming(concept_names: dict | None) -> dict:
    if not concept_names:
        return {"available": False}
    scores = [v["score"] for v in concept_names.values()]
    n = len(scores)
    n_dead = sum(1 for v in concept_names.values() if v.get("is_dead"))
    # "Best" concepts = highest non-dead score (the strongest the SAE found).
    ranked = sorted(
        ((fid, v) for fid, v in concept_names.items() if not v.get("is_dead")),
        key=lambda kv: kv[1]["score"],
        reverse=True,
    )
    best = [(v["name"], round(v["score"], 3)) for _, v in ranked[:15]]
    return {
        "available": True,
        "n_features": n,
        "n_dead": n_dead,
        "dead_pct": n_dead / n,
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "pct_lt_0_5": sum(1 for s in scores if s < 0.5) / n,
        "pct_lt_0_4": sum(1 for s in scores if s < 0.4) / n,
        "scores": scores,
        "best_concepts": best,
    }


def analyze_stability(stab: dict | None, matched: dict | None, k: int, D: int) -> dict:
    out = {"available": False, "chance_floor": k / (2 * D - k) if D > k else 0.0}
    if stab:
        st = stab.get("stability", {})
        out["available"] = True
        out["mean_jaccard"] = st.get("mean_jaccard")
        out["std_jaccard"] = st.get("std_jaccard")
        out["jaccard_matrix"] = st.get("jaccard_matrix")
        # per-seed reconstruction + dead% (shows the SAE fits well even if features are noise)
        per_seed = stab.get("per_seed_metrics", {})
        if per_seed:
            first = next(iter(per_seed.values()))
            out["recon_mse"] = first.get("mse")
            out["recon_cosine"] = first.get("cosine")
            out["l0_mean"] = first.get("l0_mean")
            out["dead_features_pct"] = first.get("dead_features_pct")
        out["clustering"] = stab.get("clustering", {})
    if matched:
        out["matched_available"] = True
        out["mean_best_match_cosine"] = matched.get("mean_best_match_cosine")
        out["null_mean"] = matched.get("null_mean")
        out["frac_matched_0_7"] = matched.get("mean_frac_matched_0.7")
        out["frac_matched_0_9"] = matched.get("mean_frac_matched_0.9")
        out["p_value"] = matched.get("p_value")
    return out


def analyze_judge(results_dir: Path) -> dict:
    """Per-model verdict distributions from checkpoints + the final aligned_scores."""
    out = {"models": [], "final": None}
    for cp in sorted(results_dir.glob("*judge_checkpoint*.json")):
        recs = _load_json(cp) or []
        dist = _verdict_distribution(recs)
        total = sum(dist.values())
        if total == 0:
            continue
        out["models"].append({
            "model": _model_label(cp.name),
            "n": total,
            "dist": dist,
            "aligned_pct": dist.get("Aligned", 0) / total,
        })
    final_csv = results_dir / "aligned_scores.csv"
    if final_csv.exists():
        df = pd.read_csv(final_csv)
        if "verdict" in df.columns:
            dist = Counter(df["verdict"].tolist())
            out["final"] = {"n": len(df), "dist": dist}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_naming(naming: dict, fig_dir: Path) -> Path | None:
    if not naming.get("available"):
        return None
    scores = naming["scores"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(scores, bins=40, color="#1565C0", alpha=0.8)
    ax.axvline(0.5, color="#C62828", ls="--", lw=1, label="0.5 (strong-match cutoff)")
    ax.set_xlabel("top-1 cosine (feature ↔ vocab)")
    ax.set_ylabel("# features")
    ax.set_title(f"Naming scores — {naming['pct_lt_0_5']:.1%} below 0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "failure_naming_scores.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _fig_jaccard(stab: dict, fig_dir: Path) -> Path | None:
    mat = stab.get("jaccard_matrix")
    if not mat:
        return None
    seeds = config.training.seeds
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(seeds)))
    ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticklabels(seeds)
    ax.set_title(f"Cross-seed Jaccard (mean {stab['mean_jaccard']:.4f})")
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            ax.text(j, i, f"{mat[i][j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=0.046, label="Jaccard")
    fig.tight_layout()
    p = fig_dir / "failure_jaccard_matrix.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _fig_judge(judge: dict, fig_dir: Path) -> Path | None:
    models = [m for m in judge["models"] if m["n"] >= 10]
    if not models:
        return None
    labels = [m["model"] for m in models]
    aligned = [m["dist"].get("Aligned", 0) / m["n"] for m in models]
    unaligned = [m["dist"].get("Unaligned", 0) / m["n"] for m in models]
    uncertain = [m["dist"].get("Uncertain", 0) / m["n"] for m in models]
    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x, aligned, label="Aligned", color="#2E7D32")
    ax.bar(x, unaligned, bottom=aligned, label="Unaligned", color="#C62828")
    ax.bar(x, uncertain, bottom=[a + u for a, u in zip(aligned, unaligned)],
           label="Uncertain", color="#9E9E9E")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("fraction of verdicts")
    ax.set_title("Judge disagreement on the SAME pseudo-reports")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "failure_judge_disagreement.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(ds: str, naming: dict, stab: dict, judge: dict, figures: dict, out: Path) -> None:
    # Image links are relative to the REPORT's directory so the report+figures
    # bundle stays portable (and survives the submission .zip).
    def _img(p: Path | None) -> str:
        return Path(os.path.relpath(p, out.parent)).as_posix() if p else ""

    lines: list[str] = []
    lines.append(f"# {ds} — Failure-Case Analysis\n")
    lines.append("_Generated by `scripts/run_failure_case_analysis.py` from the baseline run "
                 f"artifacts in `results/{ds}/`. Frames the results as the **failure cases** the "
                 "traccia requires, with the root cause and a forward link to the multi-dataset "
                 "plan. See `docs/FINDINGS.md` for the paper angles._\n")

    # TL;DR
    cf = stab.get("chance_floor")
    mj = stab.get("mean_jaccard")
    lines.append("## TL;DR\n")
    lines.append("The SAE discovers **unstable, near-random concepts**: cross-seed Jaccard sits "
                 f"at the **chance floor** ({mj:.4f} vs {cf:.4f}), the top-1 naming cosine is "
                 f"~{naming.get('mean', 0):.2f} ({naming.get('pct_lt_0_5', 0):.0%} of features < 0.5), "
                 "and three judge models return **contradictory** verdict distributions on the same "
                 "pseudo-reports. Root cause: **data starvation** (≈2.8 samples/feature) → a "
                 "**non-identifiable** sparse factorization (see FINDINGS A1, B1).\n")

    # 1. Naming
    lines.append("## 1. Naming failure — visual↔textual alignment is noise (B3)\n")
    if naming.get("available"):
        lines.append(f"- Features: **{naming['n_features']}** ({naming['n_dead']} dead, "
                     f"{naming['dead_pct']:.1%}).")
        lines.append(f"- Top-1 cosine: mean **{naming['mean']:.3f}**, median {naming['median']:.3f}, "
                     f"max **{naming['max']:.3f}**.")
        lines.append(f"- **{naming['pct_lt_0_5']:.1%}** of features score < 0.5 → essentially none "
                     "have a strong vocab match.")
        if figures.get("naming"):
            lines.append(f"\n![naming scores]({_img(figures['naming'])})\n")
        lines.append("\n**Top-15 'best' concepts** (highest score — what the SAE is most confident about):")
        for name, score in naming["best_concepts"]:
            lines.append(f"  - {name} ({score})")
        lines.append("\nThese are anatomically irrelevant to chest radiographs (ear/leg anatomy, "
                     "German-localised device labels from RadLex) — the cosine argmax is noise, not "
                     "a discovered chest concept.")
    else:
        lines.append("_concept_names.json not found — run the naming stage first._")

    # 2. Stability
    lines.append("\n## 2. Stability failure — concepts are not reproducible (B2)\n")
    if stab.get("available"):
        lines.append(f"- Cross-seed **mean Jaccard {mj:.4f}** vs analytical **chance floor {cf:.4f}** "
                     f"(k/(2D−k), k={config.sae.k}, D={config.sae.dict_size}) → **at chance**.")
        if stab.get("recon_cosine") is not None:
            lines.append(f"- Reconstruction is *good* (cosine {stab['recon_cosine']:.3f}, "
                         f"mse {stab['recon_mse']:.2e}, L0 {stab['l0_mean']:.0f}) — the SAE fits the "
                         "data, but the **features themselves are arbitrary**.")
        if stab.get("matched_available"):
            lines.append(f"- Matched (permutation-invariant) best-cosine **{stab['mean_best_match_cosine']:.3f}** "
                         f"vs null {stab['null_mean']:.3f}; fraction matched@0.7 = "
                         f"{stab['frac_matched_0_7']:.3f} → the *directions* don't align across seeds either.")
        if figures.get("jaccard"):
            lines.append(f"\n![jaccard]({_img(figures['jaccard'])})\n")
    else:
        lines.append("_stability_analysis.json not found — run the stability stage first._")

    # 3. Judge
    lines.append("\n## 3. Judge disagreement — the metric is model-dependent (C1)\n")
    if judge["models"]:
        lines.append("Verdict distributions on the **same** pseudo-reports, per judge model "
                     "(from resume checkpoints; counts are partial):\n")
        lines.append("| model | n | Aligned | Unaligned | Uncertain |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in judge["models"]:
            d = m["dist"]
            lines.append(f"| {m['model']} | {m['n']} | {d.get('Aligned', 0)} | {d.get('Unaligned', 0)} | {d.get('Uncertain', 0)} |")
        if figures.get("judge"):
            lines.append(f"\n![judge]({_img(figures['judge'])})\n")
        lines.append("\nLlama leans Uncertain, MedGemma leans Aligned, Gemma-26B leans Unaligned — "
                     "**three contradictory readings** of identical inputs. The Aligned/Unaligned/"
                     "Uncertain metric is therefore sensitive to the judge model (FINDINGS C1).")
        if judge.get("final"):
            f = judge["final"]
            lines.append(f"\n_Final `aligned_scores.csv` ({f['n']} rows): "
                         f"{dict(f['dist'])}. (A 100%-Uncertain final run indicates the judge mostly "
                         "fell back to Uncertain — parsing/ground-truth weakness, also a failure case.)_")
    else:
        lines.append("_No judge checkpoints found — run the LLM judge first._")

    # 4. Root cause + forward
    lines.append("\n## 4. Root cause & forward link (A1, B1)\n")
    lines.append("All three failures share one cause: with ~5,800 train images and `dict_size=2048` "
                 "(**≈2.8 samples/feature**), the sparse factorization is **non-identifiable** — the "
                 "loss-minimising decomposition is not unique and the learned directions are noise. "
                 "This is the project's central failure case and it **motivates the PadChest scale test** "
                 "(Phase 2): if scale is the cause, more data should raise stability off the chance floor "
                 "and the naming scores above 0.5. Compare this report with "
                 "`results/padchest/failure_cases/REPORT.md` once PadChest is run at scale.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate a dataset's results into a failure-case report.")
    p.add_argument(
        "--dataset",
        type=str,
        default=config.active_dataset.name,
        help=f"Dataset to analyse (default: {config.active_dataset.name}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.select_dataset(args.dataset)
    ds = config.active_dataset.name
    base = config.paths.baseline_results_dir          # results/<ds>/baseline/
    results_dir = config.paths.results_dir            # results/<ds>/
    fig_dir = config.paths.figures_dir                # results/<ds>/figures/
    out_dir = results_dir / "failure_cases"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Failure-case analysis — dataset={ds}")
    print(f"  baseline dir : {base}")

    naming = analyze_naming(_load_json(base / "concept_names.json"))
    stab = analyze_stability(
        _load_json(base / "stability_analysis.json"),
        _load_json(base / "stability_matched.json"),
        k=config.sae.k,
        D=config.sae.dict_size,
    )
    judge = analyze_judge(results_dir)

    figures = {
        "naming": _fig_naming(naming, fig_dir),
        "jaccard": _fig_jaccard(stab, fig_dir) if stab.get("available") else None,
        "judge": _fig_judge(judge, fig_dir),
    }
    for k, p in figures.items():
        if p:
            print(f"  figure      : {p}")

    write_report(ds, naming, stab, judge, figures, out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
