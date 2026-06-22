"""
evaluate_llm_judge.py — LLM Judge Evaluation Pipeline

Uses LangGraph to orchestrate an LLM-based evaluation of discovered concepts
against radiology reports. For each (concept, report) pair, the judge outputs:
  - Aligned:   the report supports the concept
  - Unaligned: the report contradicts the concept
  - Uncertain: ambiguous or not enough evidence

Uses unsloth/medgemma-4b-it via HuggingFace transformers as the judge LLM.

Usage:
    python src/evaluate_llm_judge.py

    # Resume from checkpoint
    python src/evaluate_llm_judge.py --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from config import paths
from utils import setup_logging

logger = setup_logging(__name__)

# Paths
EXPLANATIONS_PATH = paths.results_dir / "sample_explanations.json"
REPORTS_CSV_PATH = paths.data_dir / "iu_xray" / "reports" / "indiana_reports.csv"
PROJECTIONS_CSV_PATH = paths.data_dir / "iu_xray" / "reports" / "indiana_projections.csv"
OUTPUT_CSV_PATH = paths.results_dir / "aligned_scores.csv"
CHECKPOINT_PATH = paths.results_dir / ".judge_checkpoint.json"

# Model config
MODEL_NAME = "unsloth/medgemma-4b-it"

# Prompt template
JUDGE_PROMPT_TEMPLATE = """You are a clinical AI evaluator specializing in radiology.

Given a radiology report and a concept discovered by an interpretability method,
determine whether the report supports the concept.

Radiology report:
\"{report}\"

Discovered concept:
\"{concept}\"

Does the report SUPPORT, CONTRADICT, or is AMBIGUOUS about this concept?
Answer with exactly one word: Aligned, Unaligned, or Uncertain."""

# Appended to the prompt on retries to reinforce the expected format
RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous answer was not in the expected format. "
    "You MUST answer with exactly one word: Aligned, Unaligned, or Uncertain."
)

VALID_VERDICTS = ("Aligned", "Unaligned", "Uncertain")


# ============================================================================
# Model loader (singleton — loaded once, reused across all calls)
# ============================================================================

_pipe = None

def get_pipeline():
    """Load the MedGemma pipeline once and cache it globally.

    Device/dtype adapts to the host: CUDA uses bfloat16 + device_map='auto'
    (the original config); Apple Silicon uses float16 + 'mps' (bfloat16 is
    incompletely supported on MPS); CPU falls back to float32.
    """
    global _pipe
    if _pipe is None:
        logger.info("Loading model %s ...", MODEL_NAME)
        if torch.cuda.is_available():
            kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        elif torch.backends.mps.is_available():
            kwargs = dict(torch_dtype=torch.float16, device="mps")
        else:
            kwargs = dict(torch_dtype=torch.float32, device="cpu")
        logger.info("Device/dtype: %s", kwargs)
        _pipe = pipeline("text-generation", model=MODEL_NAME, **kwargs)
        logger.info("Model loaded successfully.")
    return _pipe


# ============================================================================
# LangGraph Judge
# ============================================================================

def build_judge_graph():
    """
    Build and compile a LangGraph StateGraph for the LLM judge.

    The graph has the following structure:
        [START] → prepare_prompt → call_llm → parse_and_validate
                                                  ↓ valid → [END]
                                                  ↓ retry → call_llm (max 2 retries)
                                                  ↓ end   → [END] (fallback: Uncertain)
    """

    # --- State schema ---
    class JudgeState(TypedDict):
        concept: str
        report: str
        prompt: str
        raw_response: str
        result: str
        retries: int

    # --- Node: prepare_prompt ---
    def prepare_prompt(state: JudgeState) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            report=state["report"],
            concept=state["concept"],
        )
        return {"prompt": prompt}

    # --- Node: call_llm ---
    def call_llm(state: JudgeState) -> dict:
        retries = state.get("retries", 0)
        prompt_text = state["prompt"]

        # On retries, append a reinforcement suffix so the model gets
        # a different (stronger) prompt — and use sampling to break
        # the deterministic loop that greedy decoding would cause.
        if retries > 0:
            prompt_text = prompt_text + RETRY_SUFFIX

        pipe = get_pipeline()

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a clinical AI evaluator specializing in radiology. "
                            "Answer with exactly one word."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # First attempt: greedy (deterministic). Retries: sample with
        # low temperature so the model can explore different outputs.
        generation_kwargs = {
            "max_new_tokens": 10,
            "do_sample": retries > 0,
        }
        if retries > 0:
            generation_kwargs["temperature"] = 0.3

        outputs = pipe(messages, **generation_kwargs)

        # Extract the assistant's reply from the generated output
        generated = outputs[0]["generated_text"]
        # The pipeline returns the full conversation; the last message is the assistant's
        if isinstance(generated, list):
            raw = generated[-1]["content"].strip()
        else:
            # Fallback: if it returns a plain string, strip the prompt
            raw = str(generated).strip()

        return {"raw_response": raw}

    # --- Node: parse_and_validate ---
    def parse_and_validate(state: JudgeState) -> dict:
        raw = state.get("raw_response", "")
        # Try to extract a valid verdict from the response
        # The LLM might wrap the answer in punctuation or extra text
        parsed = _extract_verdict(raw)
        if parsed:
            return {"result": parsed}
        # Invalid — increment retries
        return {"result": raw, "retries": state.get("retries", 0) + 1}

    # --- Conditional edge after parse_and_validate ---
    def route_after_validation(state: JudgeState) -> Literal["valid", "retry", "fallback"]:
        if state.get("result", "") in VALID_VERDICTS:
            return "valid"
        if state.get("retries", 0) < 2:
            return "retry"
        return "fallback"

    # --- Node: fallback ---
    def fallback(state: JudgeState) -> dict:
        return {"result": "Uncertain"}

    # --- Build the graph ---
    graph = StateGraph(JudgeState)

    graph.add_node("prepare_prompt", prepare_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("parse_and_validate", parse_and_validate)
    graph.add_node("fallback", fallback)

    # Edges
    graph.set_entry_point("prepare_prompt")
    graph.add_edge("prepare_prompt", "call_llm")
    graph.add_edge("call_llm", "parse_and_validate")
    graph.add_conditional_edges(
        "parse_and_validate",
        route_after_validation,
        {
            "valid": END,
            "retry": "call_llm",
            "fallback": "fallback",
        },
    )
    graph.add_edge("fallback", END)

    return graph.compile()


def _extract_verdict(raw_text: str) -> str | None:
    """
    Try to extract a valid verdict from the LLM's raw response.
    Handles cases like "Aligned.", "The answer is Unaligned", etc.
    """
    raw_lower = raw_text.lower().strip().rstrip(".")
    for verdict in VALID_VERDICTS:
        if verdict.lower() == raw_lower:
            return verdict
    # Fuzzy: check if the verdict appears as a standalone word
    for verdict in VALID_VERDICTS:
        if verdict.lower() in raw_lower.split():
            return verdict
    return None


# ============================================================================
# Checkpoint helpers
# ============================================================================

def load_checkpoint() -> set:
    """Load set of already-evaluated (image_id, concept) pairs."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        return {(r["image_id"], r["concept"]) for r in data}
    return set()


def load_checkpoint_records() -> list[dict]:
    """Load the full list of checkpoint records."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(records: list[dict]):
    """Persist intermediate results to checkpoint file."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(records, f, indent=2)


# ============================================================================
# Main evaluation loop
# ============================================================================

def evaluate(
    resume: bool = False,
    batch_save_every: int = 25,
    explanations_path: Path | None = None,
    reports_path: Path | None = None,
    projections_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Run the LLM judge on all (concept, report) pairs from sample_explanations.json.

    Args:
        resume: if True, skip already-evaluated pairs from checkpoint.
        batch_save_every: save checkpoint every N evaluations.
        explanations_path: Override for the explanations JSON path.
            Defaults to ``results/sample_explanations.json``.
        reports_path: Override for the reports CSV path.
            Defaults to ``data/iu_xray/indiana_reports.csv``.
        projections_path: Override for the projections CSV path.
            Defaults to ``data/iu_xray/indiana_projections.csv``.
        output_path: Override for the output CSV path.
            Defaults to ``results/aligned_scores.csv``.

    Returns:
        Path to the saved output CSV.
    """
    eff_explanations = explanations_path or EXPLANATIONS_PATH
    eff_reports = reports_path or REPORTS_CSV_PATH
    eff_projections = projections_path or PROJECTIONS_CSV_PATH
    eff_output = output_path or OUTPUT_CSV_PATH

    # --- Load inputs ---
    if not eff_explanations.exists():
        logger.error("Explanations file not found: %s", eff_explanations)
        logger.error("Generate explanations first.")
        sys.exit(1)

    if not eff_reports.exists():
        logger.error("Reports CSV not found: %s", eff_reports)
        logger.error("Ensure data/iu_xray/indiana_reports.csv exists.")
        sys.exit(1)

    if not eff_projections.exists():
        logger.error("Projections CSV not found: %s", eff_projections)
        logger.error("Ensure data/iu_xray/indiana_projections.csv exists.")
        sys.exit(1)

    with open(eff_explanations, "r") as f:
        explanations = json.load(f)

    reports_df = pd.read_csv(eff_reports)
    projections_df = pd.read_csv(eff_projections)
    logger.info("Loaded %d sample explanations", len(explanations))
    logger.info("Loaded %d reports, %d projections", len(reports_df), len(projections_df))

    # Build a fast lookup: filename → report text.
    #
    # The explanations JSON uses image filenames (e.g. "3222_IM-1522-2001.dcm.png")
    # as image_id.  The indiana_reports.csv uses a numeric `uid` key and stores
    # the report text in `findings` and `impression` columns.  We bridge them
    # via indiana_projections.csv which maps filename → uid.
    #
    # Step 1: Build uid → combined report text (findings + impression)
    def _combine_report(row) -> str:
        parts = []
        if pd.notna(row.get("findings")):
            parts.append(str(row["findings"]).strip())
        if pd.notna(row.get("impression")):
            parts.append(str(row["impression"]).strip())
        return " ".join(parts) if parts else ""

    reports_df["combined_text"] = reports_df.apply(_combine_report, axis=1)
    uid_to_report = dict(
        zip(reports_df["uid"].astype(str), reports_df["combined_text"])
    )

    # Step 2: Build filename → uid mapping from projections
    filename_to_uid = dict(
        zip(projections_df["filename"], projections_df["uid"].astype(str))
    )

    # Step 3: Build the final filename → report text lookup
    report_lookup = {}
    for filename, uid in filename_to_uid.items():
        report_text = uid_to_report.get(uid)
        if report_text:
            report_lookup[filename] = report_text

    # --- Resume support ---
    if resume:
        done_keys = load_checkpoint()
        records = load_checkpoint_records()
        logger.info("Resuming: %d pairs already evaluated", len(done_keys))
    else:
        done_keys = set()
        records = []

    # --- Build evaluation pairs ---
    eval_pairs = []
    skipped_no_report = 0
    for item in explanations:
        # image_id in the explanations JSON is the image filename
        # (e.g. "3222_IM-1522-2001.dcm.png")
        image_id = item.get("image_id", "")
        report = report_lookup.get(image_id)
        if not report:
            skipped_no_report += 1
            continue

        # The explanations JSON uses "top_k_concepts" with sub-keys
        # "feature_id", "name", "activation"
        concepts_list = item.get("top_k_concepts", [])
        for concept_info in concepts_list:
            concept_name = concept_info.get("name", "")
            if not concept_name:
                continue
            key = (image_id, concept_name)
            if key in done_keys:
                continue
            eval_pairs.append({
                "image_id": image_id,
                "concept_name": concept_name,
                "feature_id": concept_info["feature_id"],
                "activation": concept_info["activation"],
                "report": report,
            })

    if skipped_no_report > 0:
        logger.warning("Skipped %d samples with missing reports", skipped_no_report)

    total = len(eval_pairs)
    logger.info("Pairs to evaluate: %d", total)

    if total == 0:
        logger.info("Nothing to evaluate. Saving final results.")
        return _save_final(records, eff_output)

    # --- Load model and compile judge graph ---
    logger.info("Building LangGraph judge (model=%s)...", MODEL_NAME)
    get_pipeline()  # pre-load the model
    judge = build_judge_graph()

    # --- Run evaluation ---
    t_start = time.time()
    errors = 0

    for i, pair in enumerate(tqdm(eval_pairs, desc="LLM Judge Evaluation")):
        try:
            result_state = judge.invoke({
                "concept": pair["concept_name"],
                "report": pair["report"],
                "prompt": "",
                "raw_response": "",
                "result": "",
                "retries": 0,
            })

            verdict = result_state.get("result", "Uncertain")
            if verdict not in VALID_VERDICTS:
                verdict = "Uncertain"

            records.append({
                "image_id": pair["image_id"],
                "feature_id": pair["feature_id"],
                "concept": pair["concept_name"],
                "activation": pair["activation"],
                "verdict": verdict,
                "raw_response": result_state.get("raw_response", ""),
            })

        except Exception as e:
            errors += 1
            tqdm.write(f"  Error on {pair['image_id']}/{pair['concept_name']}: {e}")
            records.append({
                "image_id": pair["image_id"],
                "feature_id": pair["feature_id"],
                "concept": pair["concept_name"],
                "activation": pair["activation"],
                "verdict": "Uncertain",
                "raw_response": f"ERROR: {e}",
            })

        # Periodic checkpoint
        if (i + 1) % batch_save_every == 0:
            save_checkpoint(records)

    elapsed = time.time() - t_start

    # --- Save final results ---
    save_checkpoint(records)
    output_csv = _save_final(records, eff_output)

    # --- Print summary statistics ---
    _print_summary(records, elapsed, errors)

    return output_csv


def _save_final(records: list[dict], output_path: Path | None = None) -> Path:
    """Save the final aligned_scores.csv (without raw_response column).

    Args:
        records: List of evaluation record dicts.
        output_path: Destination path. Defaults to the module-level
            ``OUTPUT_CSV_PATH``.

    Returns:
        The path to the written CSV file.
    """
    dest = output_path or OUTPUT_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    # Drop raw_response for the clean output file; keep naming_confidence
    output_cols = [
        "image_id", "feature_id", "concept",
        "activation", "verdict",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df[existing_cols].to_csv(dest, index=False)
    logger.info("Results saved to: %s", dest)
    return dest


def _print_summary(records: list[dict], elapsed: float, errors: int):
    """Print evaluation summary statistics."""
    df = pd.DataFrame(records)
    total = len(df)

    print("\n" + "=" * 60)
    print("  LLM JUDGE EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"  Total evaluations  : {total}")
    print(f"  Errors (fallback)  : {errors}")
    print(f"  Elapsed time       : {elapsed:.1f}s ({elapsed / max(total, 1):.2f}s/pair)")
    print()

    if total > 0:
        counts = df["verdict"].value_counts()
        print("  Verdict Distribution:")
        for verdict in ["Aligned", "Unaligned", "Uncertain"]:
            n = counts.get(verdict, 0)
            pct = 100.0 * n / total
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {verdict:<12s} {n:>5d}  ({pct:5.1f}%)  {bar}")

        aligned_rate = counts.get("Aligned", 0) / total
        print(f"\n  Alignment Rate     : {aligned_rate:.1%}")

    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Judge Evaluation — evaluate discovered concepts against radiology reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation
  python src/evaluate_llm_judge.py

  # Resume an interrupted run
  python src/evaluate_llm_judge.py --resume
        """,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-evaluated pairs)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint every N evaluations (default: 25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("LLM Judge — model=%s", MODEL_NAME)
    logger.info("  Explanations : %s", EXPLANATIONS_PATH)
    logger.info("  Reports      : %s", REPORTS_CSV_PATH)
    logger.info("  Projections  : %s", PROJECTIONS_CSV_PATH)
    logger.info("  Output       : %s", OUTPUT_CSV_PATH)

    output_csv = evaluate(
        resume=args.resume,
        batch_save_every=args.checkpoint_every,
    )
    logger.info("Done. Results at: %s", output_csv)


if __name__ == "__main__":
    main()
