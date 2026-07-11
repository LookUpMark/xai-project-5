"""test_judge_source_resume.py — Tests for the member-3 judge fixes.

Covers the three fixes that unblock the 5-method judge run on the T4:
  1. ``_resolve_source_paths`` — per-method I/O routing so the cp-into-baseline
     trick is no longer needed (``--source spliec`` → ``aligned_scores_spliec.csv``).
  2. ``load_checkpoint`` dedup key = ``(image_id, pseudo_report)`` (was
     ``(image_id, concept)`` → KeyError + never-matching dedup, breaking --resume).
  3. Default judge model is ``unsloth/medgemma-4b-it`` (config.JudgeConfig).

Heavy optional deps (torch/transformers/langgraph) are mocked so this runs with
no GPU and no model download, mirroring tests/test_llm_judge.py's pattern.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Mock heavy optional deps before importing the module under test --------
_mock_modules = {}
for _mod in ["torch", "transformers"]:
    if _mod not in sys.modules:
        _mock_modules[_mod] = MagicMock()
        sys.modules[_mod] = _mock_modules[_mod]
if "torch" in _mock_modules:
    sys.modules["torch"].bfloat16 = "bfloat16"
if "transformers" in _mock_modules:
    sys.modules["transformers"].pipeline = MagicMock()

try:
    from langgraph.graph import StateGraph, END  # noqa: F401
except (ImportError, OSError):
    _lg = MagicMock()
    for _sub in [
        "langgraph", "langgraph.graph", "langgraph.graph.state",
        "langgraph.graph.message", "langgraph.cache", "langgraph.cache.base",
        "langgraph.checkpoint", "langgraph.checkpoint.serde",
        "langgraph.checkpoint.serde.jsonplus",
    ]:
        sys.modules.setdefault(_sub, _lg)

# --- Minimal config stub (evaluate_llm_judge imports config at top) ---------
import logging
import types

_mock_config = types.ModuleType("config")


class _FakePaths:
    project_root = PROJECT_ROOT
    results_dir = PROJECT_ROOT / "results"
    data_dir = PROJECT_ROOT / "data"


class _FakeJudge:
    # Mirror the real default — the fix under test.
    model_name = "unsloth/medgemma-4b-it"
    max_new_tokens = 64
    max_retries = 2
    batch_save_every = 25
    seed = 42


class _FakeTraining:
    primary_seed = 42


_mock_config.paths = _FakePaths()
_mock_config.judge = _FakeJudge()
_mock_config.training = _FakeTraining()
sys.modules.setdefault("config", _mock_config)

_mock_utils = types.ModuleType("utils")


def _setup_logging(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_mock_utils.setup_logging = _setup_logging
_mock_utils.set_global_seed = lambda _seed: None
sys.modules.setdefault("utils", _mock_utils)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib

judge_module = importlib.import_module("evaluate_llm_judge")


# ============================================================================
# 1. _resolve_source_paths — per-method routing (kills the cp-trick)
# ============================================================================

class TestResolveSourcePaths(unittest.TestCase):
    def _resolve(self, source, results_dir, safe_model="unsloth_medgemma-4b-it"):
        return judge_module._resolve_source_paths(source, results_dir, safe_model)

    def test_baseline_keeps_legacy_unsuffixed_filenames(self):
        """baseline → aligned_scores.csv (no suffix), backward compatible."""
        rd = Path("/repo/results/iu_xray")
        p = self._resolve("baseline", rd)
        self.assertEqual(p["explanations"], rd / "baseline" / "sample_explanations.json")
        self.assertEqual(p["output_csv"], rd / "aligned_scores.csv")
        self.assertEqual(
            p["checkpoint"],
            rd / "judge_checkpoint_unsloth_medgemma-4b-it.json",
        )
        self.assertEqual(
            p["scores_json"],
            rd / "judge_scores_unsloth_medgemma-4b-it.json",
        )

    def test_non_baseline_tags_outputs_with_source_suffix(self):
        """spliec → aligned_scores_spliec.csv + reads spliec/ dir (no cp/mv)."""
        rd = Path("/repo/results/iu_xray")
        p = self._resolve("spliec", rd)
        self.assertEqual(p["explanations"], rd / "spliec" / "sample_explanations.json")
        self.assertEqual(p["output_csv"], rd / "aligned_scores_spliec.csv")
        self.assertEqual(
            p["checkpoint"],
            rd / "judge_checkpoint_spliec_unsloth_medgemma-4b-it.json",
        )

    def test_each_method_gets_distinct_outputs(self):
        """All 5 member-3 methods must not collide on output/checkpoint paths."""
        rd = Path("/repo/results/iu_xray")
        out = {m: self._resolve(m, rd)["output_csv"].name for m in
               ["baseline", "sae_hidden", "spliec", "null", "null_k5"]}
        self.assertEqual(out["baseline"], "aligned_scores.csv")
        self.assertEqual(out["sae_hidden"], "aligned_scores_sae_hidden.csv")
        self.assertEqual(out["spliec"], "aligned_scores_spliec.csv")
        self.assertEqual(out["null"], "aligned_scores_null.csv")
        self.assertEqual(out["null_k5"], "aligned_scores_null_k5.csv")
        # All distinct.
        self.assertEqual(len(set(out.values())), 5)

    def test_underscore_sanitization_in_safe_model(self):
        """safe_model with '/' already '_'ed — suffix composes correctly."""
        rd = Path("/r")
        p = self._resolve("null_k5", rd, safe_model="google_medgemma-4b-it")
        self.assertEqual(
            p["checkpoint"].name,
            "judge_checkpoint_null_k5_google_medgemma-4b-it.json",
        )


# ============================================================================
# 2. load_checkpoint dedup key — (image_id, pseudo_report), no KeyError
# ============================================================================

class TestCheckpointDedupKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig = judge_module.CHECKPOINT_PATH
        judge_module.CHECKPOINT_PATH = Path(self.tmp) / ".judge_checkpoint.json"

    def tearDown(self):
        judge_module.CHECKPOINT_PATH = self._orig
        cp = Path(self.tmp) / ".judge_checkpoint.json"
        if cp.exists():
            cp.unlink()
        import os
        os.rmdir(self.tmp)

    def _save(self, records):
        with open(judge_module.CHECKPOINT_PATH, "w") as f:
            json.dump(records, f)

    def test_real_schema_roundtrip(self):
        """Records written by evaluate() use pseudo_report, not concept."""
        self._save([
            {"image_id": "img_001", "pseudo_report": "cardiomegaly present",
             "verdict": "Aligned", "raw_response": "Aligned"},
        ])
        keys = judge_module.load_checkpoint()
        self.assertIn(("img_001", "cardiomegaly present"), keys)

    def test_no_keyerror_on_records_without_concept(self):
        """Regression for the original bug: r["concept"] KeyError on resume.
        Real saved records have NO concept key — must load without crashing."""
        self._save([
            {"image_id": "img_001", "pseudo_report": "report A",
             "verdict": "Aligned", "raw_response": "Aligned"},
        ])
        # Must not raise KeyError.
        keys = judge_module.load_checkpoint()
        self.assertEqual(len(keys), 1)

    def test_missing_pseudo_report_falls_back_to_empty(self):
        """A malformed record (no pseudo_report) yields (image_id, ""), not a crash."""
        self._save([{"image_id": "img_001", "verdict": "Aligned"}])
        keys = judge_module.load_checkpoint()
        self.assertIn(("img_001", ""), keys)

    def test_error_records_excluded_from_done_keys(self):
        """F-002: ERROR: raw_response → retried on --resume (excluded from done)."""
        self._save([
            {"image_id": "img_001", "pseudo_report": "ok report",
             "verdict": "Aligned", "raw_response": "Aligned"},
            {"image_id": "img_002", "pseudo_report": "boom report",
             "verdict": "Uncertain", "raw_response": "ERROR: CUDA OOM"},
        ])
        keys = judge_module.load_checkpoint()
        self.assertIn(("img_001", "ok report"), keys)
        self.assertNotIn(("img_002", "boom report"), keys)

    def test_concept_keyed_legacy_records_do_not_collide(self):
        """Old (buggy) records keyed on concept must not match the pseudo_report
        dedup key — so a stale concept-keyed checkpoint is effectively ignored
        (those pairs get re-evaluated), which is the safe behavior."""
        self._save([
            {"image_id": "img_001", "concept": "cardiomegaly",
             "verdict": "Aligned", "raw_response": "Aligned"},
        ])
        keys = judge_module.load_checkpoint()
        # concept field is ignored; pseudo_report defaults to "".
        self.assertEqual(keys, {("img_001", "")})


# ============================================================================
# 3. Default model = unsloth/medgemma-4b-it (config drift fix)
# ============================================================================

class TestDefaultJudgeModel(unittest.TestCase):
    def test_module_model_name_matches_medgemma_default(self):
        """MODEL_NAME sourced from config.judge.model_name at import → MedGemma 4B,
        not the drifted Llama-3.1-8B default."""
        self.assertEqual(judge_module.MODEL_NAME, "unsloth/medgemma-4b-it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
