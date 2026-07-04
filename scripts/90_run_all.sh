#!/usr/bin/env bash
# 90_run_all.sh — single-entry pipeline that reproduces every paper result.
#
# Stages (one numbered script each, run in order):
#   00 build RadLex vocabulary -> text embeddings
#   01 extract BiomedCLIP embeddings (512-d standard)
#   10 baseline 512-d SAE: train -> naming -> explanations -> stability
#   11 Path A 768-d hidden SAE: extract_hidden -> train -> naming -> stability
#   20 SPLiCE (Path B): deterministic sparse decomposition
#   30 concept organization (clusters + RadLex families) for all 3 sources
#   40 generate count-matched null explanations (k=5 and k~13)
#   41 LLM judge (GPU + HF creds) for each method + null
#
# Usage:
#   ./scripts/90_run_all.sh                  # full pipeline
#   ./scripts/90_run_all.sh --skip-train     # reuse cached SAE models
#   ./scripts/90_run_all.sh --skip-extract   # reuse cached embeddings
#   ./scripts/90_run_all.sh --skip-judge     # skip the GPU-only judge stage
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
PY="${PY:-.venv/bin/python}"

SKIP_TRAIN=0; SKIP_EXTRACT=0; SKIP_JUDGE=0
for a in "$@"; do
  case "$a" in
    --skip-train)   SKIP_TRAIN=1;;
    --skip-extract) SKIP_EXTRACT=1;;
    --skip-judge)   SKIP_JUDGE=1;;
  esac
done
train_flag() { [ "$SKIP_TRAIN" = 1 ] && echo --skip-train; }

echo "=== 00 vocabulary ===";          "$PY" scripts/00_build_vocab.py
if [ "$SKIP_EXTRACT" = 0 ]; then
  echo "=== 01 embeddings (512-d) ==="; "$PY" scripts/01_extract_embeddings.py
fi
echo "=== 02 baseline SAE ===";        "$PY" scripts/02_baseline.py $(train_flag)
echo "=== 03 Path A hidden SAE ===";   "$PY" scripts/03_hidden.py --skip-extract $(train_flag)
echo "=== 04 SPLiCE ===";              "$PY" scripts/04_spliece.py
echo "=== 05 concept organization ==="
for src in spliece sae-baseline sae-hidden; do
  "$PY" scripts/05_concept_organization.py --source "$src"
done
echo "=== 06 null explanations ==="
"$PY" scripts/06_generate_null.py --k 13
"$PY" scripts/06_generate_null.py --k 5 --output results/iu_xray/null_k5
if [ "$SKIP_JUDGE" = 0 ]; then
  echo "=== 07 LLM judge (needs GPU + HF creds) ==="
  for m in baseline hidden spliece null_k5 null_k13; do
    "$PY" scripts/07_judge.py --input "$m"
  done
fi
echo "=== pipeline complete ==="
