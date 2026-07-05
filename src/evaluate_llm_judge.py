"""
evaluate_llm_judge.py — LLM Judge Evaluation Pipeline

Uses LangGraph to orchestrate an LLM-based evaluation of discovered concepts
against radiology reports. For each (concept, report) pair, the judge outputs:
  - Aligned:   the report supports the concept
  - Unaligned: the report contradicts the concept
  - Uncertain: ambiguous or not enough evidence

Uses unsloth/medgemma-4b-it via HuggingFace transformers as the judge LLM.

Usage:
    # Judge the baseline explanations (default source/model)
    python src/evaluate_llm_judge.py --dataset iu_xray

    # Judge the SPLiCE explanations -> aligned_scores_spliec.csv (no cp/mv)
    python src/evaluate_llm_judge.py --dataset iu_xray --source spliec

    # Resume from checkpoint (retries ERROR: pairs, skips already-done pairs)
    python src/evaluate_llm_judge.py --dataset iu_xray --source spliec --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

# Repo root on sys.path so xai_datasets.spec is importable when run as
# `python src/evaluate_llm_judge.py` (src/ is already sys.path[0]).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

import config
from config import judge as judge_cfg, training as training_cfg
from utils import setup_logging, set_global_seed
from xai_datasets.spec import DatasetSpec, get_dataset

logger = setup_logging(__name__)

# Model config — sourced from config.judge (overridable via --model).
MODEL_NAME = judge_cfg.model_name
USE_LM_STUDIO = False
LM_STUDIO_URL = "http://localhost:1234/v1"

# Per-dataset paths + prompt. (Re)resolved by evaluate() from the active
# DatasetSpec after config.select_dataset(); the helpers below read these module
# globals, so evaluate() MUST set them before any checkpoint/score call.
JUDGE_PROMPT: str = ""
EXPLANATIONS_PATH: Path = Path()
OUTPUT_CSV_PATH: Path = Path()
CHECKPOINT_PATH: Path = Path()
SCORES_JSON_PATH: Path = Path()

# The judge prompt is dataset-specific (English chest for IU X-Ray, Spanish for
# PadChest) and is loaded from the active DatasetSpec (``JUDGE_PROMPT``) inside
# evaluate(). It exposes ``{report}`` and ``{pseudo_report}`` placeholders.

# Appended to the prompt on retries to reinforce the expected format
RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous answer was not in the expected format. "
    "You MUST follow the format: <max 25 words explanation> | Verdict: <Aligned/Unaligned/Uncertain>"
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
                dtype=torch.bfloat16,
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
                kwargs = dict(dtype=torch.float16, device="mps")
            else:
                kwargs = dict(dtype=torch.float32, device="cpu")
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
        pseudo_report: str
        report: str
        prompt: str
        raw_response: str
        result: str
        retries: int

    # --- Node: prepare_prompt ---
    def prepare_prompt(state: JudgeState) -> dict:
        prompt = JUDGE_PROMPT.format(
            report=state["report"],
            pseudo_report=state["pseudo_report"],
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

        if USE_LM_STUDIO:
            import urllib.request
            import json
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical AI evaluator specializing in radiology. "
                        "Answer in the exact format: <explanation> | Verdict: <label>"
                    )
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max(judge_cfg.max_new_tokens, 1024),
            }
            
            url = f"{LM_STUDIO_URL.rstrip('/')}/chat/completions"
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                msg = resp_data["choices"][0]["message"]
                raw = msg.get("content", "")
                if not raw and "reasoning_content" in msg:
                    raw = msg["reasoning_content"]
                raw = raw.strip()
        else:
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
    """Load set of already-evaluated (image_id, pseudo_report) pairs.

    The dedup key is ``(image_id, pseudo_report)`` — matching the schema written
    by :func:`evaluate` (one record per pseudo-report) and the ``done_keys`` set
    built at eval time. Records whose ``raw_response`` starts with ``ERROR:``
    are excluded so transient infra failures are retried on ``--resume``.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        return {
            (r.get("image_id", ""), r.get("pseudo_report", ""))
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

def _resolve_source_paths(
    source: str, results_dir: Path, safe_model: str
) -> dict[str, Path]:
    """Resolve per-source judge I/O paths under ``results/<dataset>/``.

    ``source`` names the method dir holding ``sample_explanations.json``
    (``baseline``, ``sae_hidden``, ``spliece``, ``null``, ``null_k5``, …).
    ``baseline`` keeps the legacy unsuffixed filenames (``aligned_scores.csv``)
    for backward compatibility; any other source tags outputs with ``_<source>``
    (e.g. ``aligned_scores_spliec.csv``) so the five methods don't collide and
    no cp/mv into the baseline dir is needed.

    Args:
        source: Method dir name under ``results/<dataset>/``.
        results_dir: ``config.paths.results_dir`` (already dataset-routed).
        safe_model: ``MODEL_NAME`` with ``/``/``:`` → ``_`` for filenames.

    Returns:
        Dict with keys ``explanations``, ``output_csv``, ``checkpoint``,
        ``scores_json`` (all absolute Paths).
    """
    source_dir = results_dir / source
    suffix = "" if source == "baseline" else f"_{source}"
    return {
        "explanations": source_dir / "sample_explanations.json",
        "output_csv": results_dir / f"aligned_scores{suffix}.csv",
        "checkpoint": results_dir / f"judge_checkpoint{suffix}_{safe_model}.json",
        "scores_json": results_dir / f"judge_scores{suffix}_{safe_model}.json",
    }


def evaluate(
    resume: bool = False,
    batch_save_every: int = 25,
    source: str = "baseline",
) -> Path:
    """
    Run the LLM judge on all (concept, report) pairs from sample_explanations.json.

    Reads the active dataset's judge prompt + report lookup from the DatasetSpec
    (resolved via ``config.active_dataset.name`` after ``config.select_dataset``)
    and routes all I/O — explanations in, scores/checkpoints out — under
    ``results/<dataset>/``. Call ``config.select_dataset(name)`` (or the CLI
    ``--dataset`` flag) before this so the paths point at the right dataset.

    Args:
        resume: if True, skip already-evaluated pairs from checkpoint.
        batch_save_every: save checkpoint every N evaluations.
        source: name of the method dir under ``results/<dataset>/`` holding the
            ``sample_explanations.json`` to judge (``baseline``, ``sae_hidden``,
            ``spliece``, ``null``, ``null_k5``, …). ``"baseline"`` (default)
            preserves the legacy path/outputs; any other value reads from
            ``results/<dataset>/<source>/`` and tags the output artifacts with a
            ``_<source>`` suffix (e.g. ``aligned_scores_spliec.csv``), so the
            five methods can be judged without cp/mv gymnastics.

    Returns:
        Path to the saved output CSV.
    """
    global JUDGE_PROMPT, EXPLANATIONS_PATH, OUTPUT_CSV_PATH, CHECKPOINT_PATH, SCORES_JSON_PATH

    set_global_seed(training_cfg.primary_seed)
    logger.info("Global seed set to %d", training_cfg.primary_seed)

    # --- Resolve the active dataset spec + per-source paths/prompt ---
    spec: DatasetSpec = get_dataset(config.active_dataset.name)
    safe_model = MODEL_NAME.replace("/", "_").replace(":", "_")
    JUDGE_PROMPT = spec.judge_prompt
    # Source dir under results/<dataset>/<source>/ (baseline, sae_hidden,
    # spliece, null, null_k5, ...). Default "baseline" preserves the legacy
    # baseline_results_dir path + unsuffixed outputs; any other source reads its
    # own sample_explanations.json and tags outputs with _<source> so the
    # cp-into-baseline trick is no longer required.
    src_paths = _resolve_source_paths(source, config.paths.results_dir, safe_model)
    EXPLANATIONS_PATH = src_paths["explanations"]
    OUTPUT_CSV_PATH = src_paths["output_csv"]
    CHECKPOINT_PATH = src_paths["checkpoint"]
    SCORES_JSON_PATH = src_paths["scores_json"]
    logger.info(
        "Active dataset: %s (%s) | model=%s", spec.name, spec.language, MODEL_NAME
    )

    if not JUDGE_PROMPT:
        raise ValueError(
            f"Dataset {spec.name!r} has no judge_prompt — set it in its DatasetSpec."
        )

    # --- Load explanations + the dataset's report lookup ---
    if not EXPLANATIONS_PATH.exists():
        logger.error("Explanations file not found: %s", EXPLANATIONS_PATH)
        logger.error(
            "Run generate_explanations for dataset %r (source %r) first.",
            spec.name, source,
        )
        sys.exit(1)

    with open(EXPLANATIONS_PATH, "r") as f:
        explanations = json.load(f)
    logger.info("Loaded %d sample explanations", len(explanations))

    report_lookup = spec.build_report_lookup()
    logger.info(
        "Loaded %d report-lookup entries (dataset=%s)", len(report_lookup), spec.name
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
        # image_id is the image filename used as the sidecar/lookup key
        # (IU X-Ray: "3222_IM-1522-2001.dcm.png"; PadChest: "..._rarh4r.png").
        image_id = item.get("image_id", "")
        report = report_lookup.get(image_id)
        if not report:
            skipped_no_report += 1
            continue

        pseudo_report = item.get("pseudo_report", "")
        if not pseudo_report:
            continue
        key = (image_id, pseudo_report)
        if key in done_keys:
            continue
        done_keys.add(key)
        eval_pairs.append({
            "image_id": image_id,
            "pseudo_report": pseudo_report,
            "report": report,
        })

    if skipped_no_report > 0:
        logger.warning("Skipped %d samples with missing reports", skipped_no_report)

    total = len(eval_pairs)
    logger.info("Pairs to evaluate: %d", total)

    if total == 0:
        logger.info("Nothing to evaluate. Saving final results.")
        return _save_final(records, OUTPUT_CSV_PATH)

    # --- Load model and compile judge graph ---
    logger.info("Building LangGraph judge (model=%s)...", MODEL_NAME)
    if not USE_LM_STUDIO:
        get_pipeline()  # pre-load the model
    judge = build_judge_graph()

    # --- Run evaluation ---
    t_start = time.time()
    errors = 0

    for i, pair in enumerate(tqdm(eval_pairs, desc="LLM Judge Evaluation")):
        try:
            result_state = judge.invoke({
                "pseudo_report": pair["pseudo_report"],
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
                "pseudo_report": pair["pseudo_report"],
                "verdict": verdict,
                "raw_response": result_state.get("raw_response", ""),
            })

        except Exception as e:
            errors += 1
            tqdm.write(f"  Error on {pair['image_id']}/{pair['pseudo_report']}: {e}")
            records.append({
                "image_id": pair["image_id"],
                "pseudo_report": pair["pseudo_report"],
                "verdict": "Uncertain",
                "raw_response": f"ERROR: {e}",
            })

        # Periodic checkpoint
        if (i + 1) % judge_cfg.batch_save_every == 0:
            save_checkpoint(records)

    elapsed = time.time() - t_start

    # --- Save final results ---
    save_checkpoint(records)
    output_csv = _save_final(records, OUTPUT_CSV_PATH)

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
        "image_id", "pseudo_report",
        "verdict", "raw_response",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df[existing_cols].to_csv(dest, index=False)
    logger.info("Results saved to: %s", dest)
    return dest


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
  # Judge the baseline explanations (default source)
  python src/evaluate_llm_judge.py --dataset iu_xray

  # Judge the SPLiCE explanations -> aligned_scores_spliec.csv (no cp/mv)
  python src/evaluate_llm_judge.py --dataset iu_xray --source spliec

  # Resume an interrupted run (retries ERROR: pairs, skips done pairs)
  python src/evaluate_llm_judge.py --dataset iu_xray --source spliec --resume
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
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.active_dataset.name,
        help=(
            f"Active dataset (default: {config.active_dataset.name}); must be a key "
            "in xai_datasets.spec.DATASETS (e.g. iu_xray, padchest). Re-routes "
            "results/checkpoints to results/<dataset>/."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="baseline",
        help=(
            "Method dir under results/<dataset>/ holding sample_explanations.json "
            "(default: baseline). For the 5-method comparison pass the method name "
            "(sae_hidden, spliece, null, null_k5): reads "
            "results/<dataset>/<source>/sample_explanations.json and writes "
            "aligned_scores_<source>.csv (no cp/mv needed)."
        ),
    )
    parser.add_argument(
        "--lm-studio",
        action="store_true",
        help="Use LM Studio (OpenAI compatible endpoint at localhost:1234) instead of local transformers",
    )
    parser.add_argument(
        "--lm-studio-url",
        type=str,
        default="http://localhost:1234/v1",
        help="URL for LM Studio (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (e.g. for LM studio)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global MODEL_NAME, USE_LM_STUDIO, LM_STUDIO_URL

    if args.model:
        MODEL_NAME = args.model
    if args.lm_studio:
        USE_LM_STUDIO = True
        LM_STUDIO_URL = args.lm_studio_url

    # Re-route paths to the selected dataset BEFORE evaluate() resolves them.
    config.select_dataset(args.dataset)

    logger.info(
        "LLM Judge — dataset=%s source=%s model=%s (LM Studio: %s)",
        args.dataset, args.source, MODEL_NAME, USE_LM_STUDIO,
    )

    output_csv = evaluate(
        resume=args.resume,
        batch_save_every=args.checkpoint_every,
        source=args.source,
    )
    logger.info("Done. Results at: %s", output_csv)


if __name__ == "__main__":
    main()
