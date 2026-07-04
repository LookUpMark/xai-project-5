"""06_generate_null.py — random-k chance-floor baseline for the LLM judge (F-005).

For each test image, sample N random vocabulary terms and emit them in the
SAE-compatible schema. Run the LLM judge on this output exactly like any method;
its % Aligned is the chance floor. Compare methods by LIFT over this null
(pct_method / pct_null), not by raw % Aligned — methods emit different concept
counts per image (Baseline/Path A ~5, SPLiCE ~18), so raw % Aligned is not
comparable. See docs/LLM-JUDGE-COMPLETE-GUIDE.md (FASE 3).

By default N mirrors SPLiCE's per-image concept count (matched "lottery tickets");
pass --k for a fixed count.

Usage:
    python scripts/06_generate_null.py
    python scripts/06_generate_null.py --k 18          # fixed count
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a random-k null explanation set (chance floor for % Aligned)."
    )
    p.add_argument(
        "--k", type=int, default=None,
        help="fixed concepts/image; if omitted, mirror SPLiCE's per-image count",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--output", type=Path, default=None, help="output dir (default: results/null)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = args.output or (config.paths.results_dir / "null")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vocabulary terms (canonical ordering == text_vocab_embeddings.pt rows).
    with open(config.paths.vocab_labels_path) as f:
        vocab = json.load(f)
    terms = [t["term"] for t in vocab]

    # Image IDs in test order (canonical sidecar next to the embeddings).
    with open(config.paths.test_image_ids_path) as f:
        image_ids = json.load(f)

    # Match SPLiCE's per-image concept count so the lottery-ticket count is equal.
    spliece_path = config.spliece.output_dir / "sample_explanations.json"
    counts = None
    if args.k is None:
        if not spliece_path.exists():
            raise FileNotFoundError(
                f"{spliece_path} missing; run `python scripts/04_spliece.py` first, "
                "or pass --k to use a fixed count."
            )
        with open(spliece_path) as f:
            spliece = json.load(f)
        counts = {r["image_id"]: len(r["top_k_concepts"]) for r in spliece}
        print(f"  Matching SPLiCE per-image counts from {spliece_path} (n={len(spliece)})")

    top_n = config.explanation.explanation_top_n
    results = []
    for img_id in image_ids:
        n = args.k if counts is None else counts.get(img_id, config.spliece.k)
        n = max(1, min(n, len(terms)))
        chosen = random.sample(range(len(terms)), n)
        concepts = [
            # activation is irrelevant to the judge (it scores on `name`); a random
            # value keeps the field schema-valid and non-zero.
            {"feature_id": int(idx), "name": terms[idx], "activation": round(random.random(), 4)}
            for idx in chosen
        ]
        results.append({
            "image_id": img_id,
            "top_k_concepts": concepts,
            "pseudo_report": "Findings suggest: " + ", ".join(c["name"] for c in concepts[:top_n]),
        })

    out_path = output_dir / "sample_explanations.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {len(results)} null explanations -> {out_path}")
    print("  Next: run the LLM judge with EXPLANATIONS_PATH = "
          "paths.results_dir / 'null' / 'sample_explanations.json', then compare LIFT.")


if __name__ == "__main__":
    main()
