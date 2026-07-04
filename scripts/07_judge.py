#!/usr/bin/env python3
"""07_judge.py — numbered entry point for the LLM judge.

Thin wrapper around ``src/evaluate_llm_judge.py`` so the concept-discovery
pipeline has one ordered entry per stage. All CLI arguments are forwarded,
notably ``--input {baseline,hidden,spliece,null_k5,null_k13}`` to select which
method's ``sample_explanations.json`` to judge, and ``--dataset``.

The judge requires a GPU + HuggingFace credentials for the configured model
(``config.judge.model_name``); see ``docs/LLM-JUDGE-COMPLETE-GUIDE.md``.

Examples:
    python scripts/07_judge.py --input baseline --dataset iu_xray
    python scripts/07_judge.py --input spliece --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable (sibling-import convention; no package install).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate_llm_judge import main  # noqa: E402

if __name__ == "__main__":
    main()
