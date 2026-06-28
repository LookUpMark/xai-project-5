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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import paths, judge as judge_cfg, training as training_cfg
from utils import setup_logging, set_global_seed

logger = setup_logging(__name__)

# Paths
EXPLANATIONS_PATH = paths.baseline_results_dir / "sample_explanations.json"
REPORTS_CSV_PATH = paths.data_dir / "iu_xray" / "indiana_reports.csv"
PROJECTIONS_CSV_PATH = paths.data_dir / "iu_xray" / "indiana_projections.csv"
OUTPUT_CSV_PATH = paths.results_dir / "aligned_scores.csv"
CHECKPOINT_PATH = paths.results_dir / ".judge_checkpoint.json"
SCORES_JSON_PATH = paths.results_dir / "judge_scores.json"
COVERAGE_JSON_PATH = paths.results_dir / "judge_coverage.json"

# Model config — sourced from config.judge 
MODEL_NAME = judge_cfg.model_name

# Prompt template — includes Rules block mapping verbs to labels 
JUDGE_PROMPT_TEMPLATE = """You are a clinical AI evaluator specializing in radiology.

Given a radiology report and a concept discovered by an interpretability method,
determine whether the report supports the concept.

Rules:
- SUPPORTS (Aligned): The report explicitly mentions or implies this finding/concept.
- CONTRADICTS (Unaligned): 
    1. The report explicitly denies this concept.
    2. OR the concept is a pathology/abnormality (e.g., pneumonia, mass, fracture) and the report does NOT mention it. In radiology, unmentioned pathologies are assumed absent.
- AMBIGUOUS (Uncertain): The concept is a normal anatomical structure or artifact (e.g., ribs, spine, devices) that might be in the image but is simply not mentioned by the radiologist because it is normal or irrelevant.

Examples:

Radiology report: "There is an increased opacity in the right upper lobe with associated atelectasis."
Discovered concept: "flexible Spule"
Answer format: The report discusses lung opacities, but a flexible Spule (coil) is a completely unrelated artifact. | Verdict: Uncertain

Radiology report: "The heart is top normal in size. The lungs are clear. No acute disease."
Discovered concept: "cardiomegaly"
Answer format: The report states the heart is normal size, which explicitly contradicts cardiomegaly (enlarged heart). | Verdict: Unaligned

Radiology report: "There is an increased opacity in the right upper lobe with possible mass."
Discovered concept: "mass lesion"
Answer format: The report explicitly mentions a possible mass, which aligns with the concept of a mass lesion. | Verdict: Aligned

Now evaluate the following:

Radiology report:
"{report}"

Discovered concept:
"{concept}"

Answer format: <max 15 words explanation> | Verdict: <Aligned/Unaligned/Uncertain>"""

# Appended to the prompt on retries to reinforce the expected format
RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous answer was not in the expected format. "
    "You MUST follow the format: <max 15 words explanation> | Verdict: <Aligned/Unaligned/Uncertain>"
)

VALID_VERDICTS = ("Aligned", "Unaligned", "Uncertain")

# Verb→label alias map so verb-shaped LLM answers are correctly mapped (F-001)
_VERB_ALIASES: dict[str, str] = {
    "support": "Aligned", "supports": "Aligned", "supported": "Aligned",
    "contradict": "Unaligned", "contradicts": "Unaligned", "contradicted": "Unaligned",
    "deny": "Unaligned", "denies": "Unaligned", "denied": "Unaligned",
    "ambiguous": "Uncertain", "ambiguity": "Uncertain", "unclear": "Uncertain",
}


# ============================================================================
# Model loader (singleton — loaded once, reused across all calls)
# ============================================================================

_pipe = None

def get_pipeline():
    """Load the MedGemma pipeline once and cache it globally.

    Uses BitsAndBytes for 4-bit quantization if CUDA is available,
    otherwise falls back to unquantized float16/float32.
    """
    global _pipe
    if _pipe is None:
        import os
        from huggingface_hub import login

        # Try to load .env file gracefully if python-dotenv is installed
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.info("Logging into Hugging Face via environment token.")
            login(token=hf_token)
        else:
            logger.warning("No HF_TOKEN found in environment. Restricted models may fail to load.")

        logger.info("Loading model %s ...", MODEL_NAME)
        if torch.cuda.is_available():
            logger.info("CUDA available: loading in 4-bit quantized mode.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            _pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,
            )
        else:
            if torch.backends.mps.is_available():
                kwargs = dict(torch_dtype=torch.float16, device="mps")
            else:
                kwargs = dict(torch_dtype=torch.float32, device="cpu")
            logger.info("Device/dtype (unquantized): %s", kwargs)
            _pipe = pipeline(
                "text-generation", 
                model=MODEL_NAME, 
                return_full_text=False,
                **kwargs
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
        # a different (stronger) prompt.  We keep do_sample=False
        # (greedy / deterministic) on all attempts — the RETRY_SUFFIX
        # prompt variation is sufficient to break greedy loops
        if retries > 0:
            prompt_text = prompt_text + RETRY_SUFFIX

        pipe = get_pipeline()

        # F-008: fold system-role content into the user message to match
        # the spec and avoid unverified system-turn behaviour in the
        # MedGemma chat template.
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a clinical AI evaluator specializing in radiology. "
                            "Answer in the exact format: <explanation> | Verdict: <label>"
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

        # Greedy decoding on every attempt for full determinism
        generation_kwargs = {
            "max_new_tokens": judge_cfg.max_new_tokens,
            "do_sample": False,
        }

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
        if state.get("retries", 0) < judge_cfg.max_retries:
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

    Handles the CoT format: "Explanation | Verdict: Aligned".
    Also maps verb-shaped answers (``"Supports"``, ``"Contradicts"``,
    ``"Ambiguous"``) to their corresponding label via ``_VERB_ALIASES``
    """
    # 1. Try to split by "Verdict:"
    parts = raw_text.split("Verdict:")
    if len(parts) > 1:
        verdict_str = parts[-1].strip().strip(".").lower()
        # Sort by length descending to check 'Unaligned' before 'Aligned'
        for verdict in sorted(VALID_VERDICTS, key=len, reverse=True):
            if verdict.lower() in verdict_str:
                return verdict
        for alias, label in _VERB_ALIASES.items():
            if alias in verdict_str:
                return label

    # 2. Fallback: search the entire raw_text
    raw_lower = raw_text.lower().strip()
    for verdict in VALID_VERDICTS:
        if verdict.lower() in raw_lower.split():
            return verdict
    for word in raw_lower.split():
        if word in _VERB_ALIASES:
            return _VERB_ALIASES[word]
    return None


# ============================================================================
# Checkpoint helpers
# ============================================================================

def load_checkpoint() -> set:
    """Load set of already-evaluated (image_id, concept) pairs."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        return {
            (r["image_id"], r["concept"])
            for r in data
            if not str(r.get("raw_response", "")).startswith("ERROR:")
        }
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
            Defaults to ``results/baseline/sample_explanations.json``.
        reports_path: Override for the reports CSV path.
            Defaults to ``data/iu_xray/indiana_reports.csv``.
        projections_path: Override for the projections CSV path.
            Defaults to ``data/iu_xray/indiana_projections.csv``.
        output_path: Override for the output CSV path.
            Defaults to ``results/aligned_scores.csv``.

    Returns:
        Path to the saved output CSV.
    """
    # Set all random seeds for full reproducibility before any
    # model inference.  Reuses the existing helper and primary_seed.
    set_global_seed(training_cfg.primary_seed)
    logger.info("Global seed set to %d", training_cfg.primary_seed)

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
    skipped_image_ids: list[str] = []  # track which images were dropped
    for item in explanations:
        # image_id in the explanations JSON is the image filename
        # (e.g. "3222_IM-1522-2001.dcm.png")
        image_id = item.get("image_id", "")
        report = report_lookup.get(image_id)
        if not report:
            skipped_no_report += 1
            skipped_image_ids.append(image_id)
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
            done_keys.add(key)  # F-005: dedup within this run too, not only on --resume
            eval_pairs.append({
                "image_id": image_id,
                "concept_name": concept_name,
                "feature_id": concept_info.get("feature_id", -1),       # F-009
                "activation": concept_info.get("activation", 0.0),       # F-009
                "report": report,
            })

    if skipped_no_report > 0:
        logger.warning("Skipped %d samples with missing reports", skipped_no_report)

    # persist skipped images for downstream coverage auditing
    _save_coverage(
        skipped_no_report, skipped_image_ids, len(explanations),
    )

    eval_pairs = eval_pairs[:300]
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
        if (i + 1) % judge_cfg.batch_save_every == 0:
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
    output_cols = [
        "image_id", "feature_id", "concept",
        "activation", "verdict", "raw_response",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df[existing_cols].to_csv(dest, index=False)
    logger.info("Results saved to: %s", dest)
    return dest


def _save_coverage(
    skipped_count: int,
    skipped_ids: list[str],
    total_images: int,
) -> None:
    """Persist coverage statistics to ``results/judge_coverage.json`` (F-006)."""
    COVERAGE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    exclusion_rate = skipped_count / max(total_images, 1)
    payload = {
        "skipped_no_report_count": skipped_count,
        "skipped_image_ids": skipped_ids,
        "total_images": total_images,
        "exclusion_rate": round(exclusion_rate, 4),
    }
    with open(COVERAGE_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Coverage stats saved to: %s", COVERAGE_JSON_PATH)


def _print_summary(records: list[dict], elapsed: float, errors: int):
    """Print evaluation summary statistics."""
    df = pd.DataFrame(records)
    total = len(df)

    # compute error count from raw_response, not just the caller counter
    if "raw_response" in df.columns:
        error_mask = df["raw_response"].astype(str).str.startswith("ERROR:")
        error_count = int(error_mask.sum())
    else:
        error_count = errors
    total_valid = total - error_count

    print("\n" + "=" * 60)
    print("  LLM JUDGE EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"  Total evaluations  : {total}")
    print(f"  Infra errors       : {error_count}")
    print(f"  Valid verdicts     : {total_valid}")
    print(f"  Elapsed time       : {elapsed:.1f}s ({elapsed / max(total, 1):.2f}s/pair)")
    print()

    aligned_count = 0
    unaligned_count = 0
    uncertain_count = 0

    if total_valid > 0:
        #compute over valid verdicts only (exclude infra errors)
        if "raw_response" in df.columns:
            df_valid = df[~error_mask]
        else:
            df_valid = df
        counts = df_valid["verdict"].value_counts()
        print("  Verdict Distribution (valid only):")
        for verdict in ["Aligned", "Unaligned", "Uncertain"]:
            n = counts.get(verdict, 0)
            pct = 100.0 * n / total_valid
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {verdict:<12s} {n:>5d}  ({pct:5.1f}%)  {bar}")

        aligned_count = int(counts.get("Aligned", 0))
        unaligned_count = int(counts.get("Unaligned", 0))
        uncertain_count = int(counts.get("Uncertain", 0))
        aligned_rate = aligned_count / total_valid
        print(f"\n  Alignment Rate (valid) : {aligned_rate:.1%}")
    elif total > 0:
        aligned_rate = 0.0
        print("  No valid verdicts (all records are infra errors).")
    else:
        aligned_rate = 0.0

    print("=" * 60)

    # persist Score(c) to JSON so downstream consumers have a
    # canonical artifact and don't have to recompute from the CSV.
    SCORES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores_payload = {
        "aligned": aligned_count,
        "unaligned": unaligned_count,
        "uncertain": uncertain_count,
        "aligned_rate": round(aligned_rate, 4),
        "n_total": total,
        "n_errors": error_count,
        "n_valid": total_valid,
    }
    with open(SCORES_JSON_PATH, "w") as f:
        json.dump(scores_payload, f, indent=2)
    logger.info("Scores saved to: %s", SCORES_JSON_PATH)


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
