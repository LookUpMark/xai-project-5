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
from typing import Literal

from langgraph.graph import StateGraph, END
from typing import TypedDict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from config import paths
from utils import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Project paths (derived from centralised config.PathsConfig)
# ---------------------------------------------------------------------------
EXPLANATIONS_PATH = paths.results_dir / "sample_explanations.json"
REPORTS_CSV_PATH = paths.data_dir / "iu_xray" / "reports.csv"
OUTPUT_CSV_PATH = paths.results_dir / "aligned_scores.csv"
CHECKPOINT_PATH = paths.results_dir / ".judge_checkpoint.json"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/medgemma-4b-it"

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

VALID_VERDICTS = {"Aligned", "Unaligned", "Uncertain"}


# ============================================================================
# Model loader (singleton — loaded once, reused across all calls)
# ============================================================================

_pipe = None


def get_pipeline():
    """Load the MedGemma pipeline once and cache it globally."""
    global _pipe
    if _pipe is None:
        logger.info("Loading model %s ...", MODEL_NAME)
        _pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
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
):
    """
    Run the LLM judge on all (concept, report) pairs from sample_explanations.json.

    Args:
        resume: if True, skip already-evaluated pairs from checkpoint
        batch_save_every: save checkpoint every N evaluations
    """
    # --- Load inputs ---
    if not EXPLANATIONS_PATH.exists():
        logger.error("Explanations file not found: %s", EXPLANATIONS_PATH)
        logger.error("Run 04_generate_explanations.py first.")
        sys.exit(1)

    if not REPORTS_CSV_PATH.exists():
        logger.error("Reports CSV not found: %s", REPORTS_CSV_PATH)
        logger.error("Ensure data/iu_xray/reports.csv exists.")
        sys.exit(1)

    with open(EXPLANATIONS_PATH, "r") as f:
        explanations = json.load(f)

    reports_df = pd.read_csv(REPORTS_CSV_PATH)
    logger.info("Loaded %d sample explanations", len(explanations))
    logger.info("Loaded %d reports", len(reports_df))

    # Build a fast lookup: image_id → combined_text
    report_lookup = dict(
        zip(reports_df["image_id"].astype(str), reports_df["combined_text"])
    )

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
        image_id = str(item["image_id"])
        report = report_lookup.get(image_id)
        if report is None or (isinstance(report, float) and pd.isna(report)):
            skipped_no_report += 1
            continue
        for concept_info in item["top_k_concepts"]:
            key = (image_id, concept_info["name"])
            if key in done_keys:
                continue
            eval_pairs.append({
                "image_id": image_id,
                "concept_name": concept_info["name"],
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
        _save_final(records)
        return

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
    _save_final(records)

    # --- Print summary statistics ---
    _print_summary(records, elapsed, errors)


def _save_final(records: list[dict]):
    """Save the final aligned_scores.csv (without raw_response column)."""
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    # Drop raw_response for the clean output file
    output_cols = ["image_id", "feature_id", "concept", "activation", "verdict"]
    existing_cols = [c for c in output_cols if c in df.columns]
    df[existing_cols].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Results saved to: %s", OUTPUT_CSV_PATH)


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
    logger.info("  Output       : %s", OUTPUT_CSV_PATH)

    evaluate(
        resume=args.resume,
        batch_save_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
