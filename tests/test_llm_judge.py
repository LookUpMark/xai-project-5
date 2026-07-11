"""
tests/test_llm_judge.py — Unit & integration tests for evaluate_llm_judge.py

All tests mock the HuggingFace pipeline so they can run without GPU or model
download.  Run with:

    python -m pytest tests/test_llm_judge.py -v

Or directly:

    python tests/test_llm_judge.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Mock heavy optional dependencies BEFORE importing the module under test.
# This allows the tests to run even if torch / transformers are not installed
# in the current Python environment, and avoids the ormsgpack DLL issue
# with langgraph on Python 3.14.
# ---------------------------------------------------------------------------
_mock_modules = {}
for mod_name in ["torch", "transformers"]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = MagicMock()
        sys.modules[mod_name] = _mock_modules[mod_name]

# Make torch.bfloat16 a simple value so dtype= doesn't fail
if "torch" in _mock_modules:
    sys.modules["torch"].bfloat16 = "bfloat16"

# Make `from transformers import pipeline` return a MagicMock
if "transformers" in _mock_modules:
    sys.modules["transformers"].pipeline = MagicMock()

# Mock langgraph to avoid ormsgpack DLL import failure.
# We need real-enough StateGraph / END objects for the graph builder to work,
# so we import langgraph and only intercept if the import itself fails.
try:
    from langgraph.graph import StateGraph, END  # noqa: F401
except (ImportError, OSError):
    _lg_mock = MagicMock()
    for sub in [
        "langgraph",
        "langgraph.graph",
        "langgraph.graph.state",
        "langgraph.graph.message",
        "langgraph.cache",
        "langgraph.cache.base",
        "langgraph.checkpoint",
        "langgraph.checkpoint.serde",
        "langgraph.checkpoint.serde.jsonplus",
    ]:
        sys.modules.setdefault(sub, _lg_mock)

    # Provide minimal stubs for StateGraph and END so the graph builder
    # creates a functional (albeit no-op) compiled graph.
    class _StubEND:
        """Sentinel for the END node."""
        pass

    class _StubStateGraph:
        """Minimal StateGraph stand-in that records nodes/edges and returns
        a compiled graph whose .invoke() runs the nodes sequentially."""

        def __init__(self, state_schema):
            self._schema = state_schema
            self._nodes = {}
            self._entry = None
            self._edges = {}
            self._conditional_edges = {}

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def set_entry_point(self, name):
            self._entry = name

        def add_edge(self, src, dst):
            self._edges[src] = dst

        def add_conditional_edges(self, src, router_fn, mapping):
            self._conditional_edges[src] = (router_fn, mapping)

        def compile(self):
            return _CompiledGraph(self)

    class _CompiledGraph:
        def __init__(self, graph):
            self._g = graph

        def invoke(self, state):
            current = self._g._entry
            while current is not None and not isinstance(current, type) and current is not _StubEND:
                if current == "__end__":
                    break
                fn = self._g._nodes.get(current)
                if fn:
                    updates = fn(state)
                    if updates:
                        state.update(updates)
                # Check conditional edges first
                if current in self._g._conditional_edges:
                    router_fn, mapping = self._g._conditional_edges[current]
                    route = router_fn(state)
                    target = mapping.get(route)
                    if target is _StubEND or target == "__end__" or target is None:
                        break
                    current = target
                elif current in self._g._edges:
                    target = self._g._edges[current]
                    if target is _StubEND or target == "__end__":
                        break
                    current = target
                else:
                    break
            return state

    # Patch the mock module to expose our stubs
    sys.modules["langgraph.graph"].StateGraph = _StubStateGraph
    sys.modules["langgraph.graph"].END = "__end__"

# ---------------------------------------------------------------------------
# Mock config and utils modules that evaluate_llm_judge now imports.
# config.py imports torch at module level, so we provide a lightweight stub
# with the same interface (paths.results_dir, paths.data_dir).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ---------------------------------------------------------------------------
import logging
import types

_mock_config = types.ModuleType("config")

class _FakePathsConfig:
    """Minimal stand-in for config.PathsConfig used by the judge module."""
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results_dir = PROJECT_ROOT / "results"
        self.data_dir = PROJECT_ROOT / "data"

_mock_config.paths = _FakePathsConfig()

# Mock judge and training configs that evaluate_llm_judge now imports (F-007)
class _FakeJudgeConfig:
    model_name = "unsloth/medgemma-4b-it"
    max_new_tokens = 10
    max_retries = 2
    batch_save_every = 25
    seed = 42

class _FakeTrainingConfig:
    primary_seed = 42

_mock_config.judge = _FakeJudgeConfig()
_mock_config.training = _FakeTrainingConfig()

sys.modules.setdefault("config", _mock_config)

_mock_utils = types.ModuleType("utils")

def _setup_logging(name: str = __name__) -> logging.Logger:
    """Lightweight logger for tests — avoids importing the real utils.py."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def _noop_seed(seed: int) -> None:
    """No-op replacement for utils.set_global_seed in tests."""
    pass

_mock_utils.setup_logging = _setup_logging
_mock_utils.set_global_seed = _noop_seed
sys.modules.setdefault("utils", _mock_utils)

# ---------------------------------------------------------------------------
# Make sure `src/` is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib
judge_module = importlib.import_module("evaluate_llm_judge")


# ============================================================================
# 1. Test _extract_verdict  (pure logic — no mocking needed)
# ============================================================================

class TestExtractVerdict(unittest.TestCase):
    """Test the fuzzy verdict parser."""

    def test_exact_aligned(self):
        self.assertEqual(judge_module._extract_verdict("Aligned"), "Aligned")

    def test_exact_unaligned(self):
        self.assertEqual(judge_module._extract_verdict("Unaligned"), "Unaligned")

    def test_exact_uncertain(self):
        self.assertEqual(judge_module._extract_verdict("Uncertain"), "Uncertain")

    def test_with_trailing_period(self):
        self.assertEqual(judge_module._extract_verdict("Aligned."), "Aligned")

    def test_with_whitespace(self):
        self.assertEqual(judge_module._extract_verdict("  Unaligned  "), "Unaligned")

    def test_case_insensitive(self):
        self.assertEqual(judge_module._extract_verdict("aligned"), "Aligned")
        self.assertEqual(judge_module._extract_verdict("UNCERTAIN"), "Uncertain")

    def test_embedded_in_sentence(self):
        result = judge_module._extract_verdict("The answer is Aligned based on the report")
        self.assertEqual(result, "Aligned")

    def test_invalid_returns_none(self):
        self.assertIsNone(judge_module._extract_verdict("Maybe"))
        self.assertIsNone(judge_module._extract_verdict(""))
        self.assertIsNone(judge_module._extract_verdict("I think so"))

    def test_garbage_returns_none(self):
        self.assertIsNone(judge_module._extract_verdict("asdf1234!@#"))

    # --- F-001: verb→label alias mapping ---
    def test_verb_supports_maps_to_aligned(self):
        self.assertEqual(judge_module._extract_verdict("Supports"), "Aligned")

    def test_verb_support_maps_to_aligned(self):
        self.assertEqual(judge_module._extract_verdict("Support"), "Aligned")

    def test_verb_contradicts_maps_to_unaligned(self):
        self.assertEqual(judge_module._extract_verdict("Contradicts"), "Unaligned")

    def test_verb_ambiguous_maps_to_uncertain(self):
        self.assertEqual(judge_module._extract_verdict("Ambiguous"), "Uncertain")

    def test_verb_unclear_maps_to_uncertain(self):
        self.assertEqual(judge_module._extract_verdict("Unclear"), "Uncertain")

    def test_verb_embedded_in_sentence(self):
        result = judge_module._extract_verdict("The report clearly supports this concept")
        self.assertEqual(result, "Aligned")

    def test_verb_denies_maps_to_unaligned(self):
        self.assertEqual(judge_module._extract_verdict("Denies"), "Unaligned")


# ============================================================================
# 2. Test checkpoint helpers  (filesystem I/O)
# ============================================================================

class TestCheckpointHelpers(unittest.TestCase):
    """Test save/load checkpoint using a temp directory."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._orig_checkpoint = judge_module.CHECKPOINT_PATH
        judge_module.CHECKPOINT_PATH = Path(self.tmp_dir) / ".judge_checkpoint.json"

    def tearDown(self):
        judge_module.CHECKPOINT_PATH = self._orig_checkpoint
        # Clean up temp files
        cp = Path(self.tmp_dir) / ".judge_checkpoint.json"
        if cp.exists():
            cp.unlink()
        os.rmdir(self.tmp_dir)

    def test_load_empty_checkpoint(self):
        """No checkpoint file → empty set."""
        keys = judge_module.load_checkpoint()
        self.assertEqual(keys, set())

    def test_load_empty_records(self):
        """No checkpoint file → empty list."""
        records = judge_module.load_checkpoint_records()
        self.assertEqual(records, [])

    def test_save_and_load_roundtrip(self):
        """Save records, reload, verify contents.

        Done-keys use the (image_id, pseudo_report) tuple — matching the schema
        written by evaluate() and the dedup key built at eval time.
        """
        records = [
            {"image_id": "img_001", "pseudo_report": "cardiomegaly present", "verdict": "Aligned"},
            {"image_id": "img_002", "pseudo_report": "pleural effusion noted", "verdict": "Unaligned"},
        ]
        judge_module.save_checkpoint(records)

        loaded = judge_module.load_checkpoint_records()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["verdict"], "Aligned")

        keys = judge_module.load_checkpoint()
        self.assertIn(("img_001", "cardiomegaly present"), keys)
        self.assertIn(("img_002", "pleural effusion noted"), keys)

    def test_overwrite_checkpoint(self):
        """Saving again overwrites the previous checkpoint."""
        judge_module.save_checkpoint([{"image_id": "a", "concept": "b"}])
        judge_module.save_checkpoint([{"image_id": "x", "concept": "y"}])
        loaded = judge_module.load_checkpoint_records()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["image_id"], "x")

    def test_resume_retries_errored_pair(self):
        """F-002: errored records (raw_response starts with ERROR:) must NOT
        appear in done_keys so that --resume retries them."""
        records = [
            {
                "image_id": "img_001",
                "pseudo_report": "cardiomegaly present",
                "verdict": "Aligned",
                "raw_response": "Aligned",
            },
            {
                "image_id": "img_002",
                "pseudo_report": "pleural effusion noted",
                "verdict": "Uncertain",
                "raw_response": "ERROR: CUDA out of memory",
            },
        ]
        judge_module.save_checkpoint(records)
        keys = judge_module.load_checkpoint()
        # The successful record should be in done_keys
        self.assertIn(("img_001", "cardiomegaly present"), keys)
        # The errored record should NOT be in done_keys
        self.assertNotIn(("img_002", "pleural effusion noted"), keys)

    def test_load_checkpoint_no_keyerror_on_legacy_concept_records(self):
        """Regression: load_checkpoint previously read r["concept"] (KeyError)
        and used (image_id, concept) as the dedup key, which never matched the
        (image_id, pseudo_report) key built at eval time. Saved records have no
        "concept" field, so a checkpoint must load without KeyError and fall
        back to "" for the missing field rather than crashing on --resume.
        """
        records = [
            {"image_id": "img_001", "pseudo_report": "cardiomegaly present", "verdict": "Aligned"},
        ]
        judge_module.save_checkpoint(records)
        # Must not raise; legacy record without pseudo_report yields ("img", "").
        keys = judge_module.load_checkpoint()
        self.assertIn(("img_001", "cardiomegaly present"), keys)


# ============================================================================
# 3. Test LangGraph judge graph  (mock the pipeline)
# ============================================================================

def _make_mock_pipeline(response_text: str):
    """Create a mock HF pipeline that always returns `response_text`."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            "generated_text": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": response_text},
            ]
        }
    ]
    return mock_pipe


class TestJudgeGraph(unittest.TestCase):
    """Test the LangGraph state machine with mocked LLM calls."""

    def _invoke_graph(self, response_text: str):
        """Helper: patch get_pipeline, build graph, invoke it."""
        mock_pipe = _make_mock_pipeline(response_text)
        with patch.object(judge_module, "get_pipeline", return_value=mock_pipe):
            graph = judge_module.build_judge_graph()
            result = graph.invoke({
                "concept": "cardiomegaly",
                "report": "The heart is enlarged.",
                "prompt": "",
                "raw_response": "",
                "result": "",
                "retries": 0,
            })
        return result

    def test_valid_aligned(self):
        result = self._invoke_graph("Aligned")
        self.assertEqual(result["result"], "Aligned")

    def test_valid_unaligned(self):
        result = self._invoke_graph("Unaligned")
        self.assertEqual(result["result"], "Unaligned")

    def test_valid_uncertain(self):
        result = self._invoke_graph("Uncertain")
        self.assertEqual(result["result"], "Uncertain")

    def test_valid_with_trailing_dot(self):
        """LLM responds 'Aligned.' — should still parse correctly."""
        result = self._invoke_graph("Aligned.")
        self.assertEqual(result["result"], "Aligned")

    def test_fallback_after_retries(self):
        """If the LLM always returns garbage, fallback to 'Uncertain' after 2 retries."""
        result = self._invoke_graph("I don't know what to say")
        self.assertEqual(result["result"], "Uncertain")
        # retries should have reached 2 (or more) before fallback
        self.assertGreaterEqual(result["retries"], 2)

    def test_prompt_formatting(self):
        """Verify the prompt includes the concept and report text."""
        mock_pipe = _make_mock_pipeline("Aligned")
        with patch.object(judge_module, "get_pipeline", return_value=mock_pipe):
            graph = judge_module.build_judge_graph()
            result = graph.invoke({
                "concept": "pleural effusion",
                "report": "Bilateral effusions noted.",
                "prompt": "",
                "raw_response": "",
                "result": "",
                "retries": 0,
            })
        # Check that the prompt was constructed (it's stored in state)
        self.assertIn("pleural effusion", result["prompt"])
        self.assertIn("Bilateral effusions noted.", result["prompt"])

    def test_verb_response_maps_correctly(self):
        """F-001: if the LLM responds with a verb like 'Supports', the graph
        should parse it to 'Aligned' on the first try (no retries needed)."""
        result = self._invoke_graph("Supports")
        self.assertEqual(result["result"], "Aligned")
        # No retries should have been needed
        self.assertEqual(result["retries"], 0)


# ============================================================================
# 4. End-to-end test with mock data files
# ============================================================================

class TestEndToEnd(unittest.TestCase):
    """Full pipeline test: mock data files + mock LLM → verify CSV output."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.tmp_dir) / "results"
        self.results_dir.mkdir()
        self.data_dir = Path(self.tmp_dir) / "data" / "iu_xray" / "reports"
        self.data_dir.mkdir(parents=True)

        # Create mock sample_explanations.json
        explanations = [
            {
                "image_id": "img_001",
                "top_k_concepts": [
                    {"feature_id": 10, "name": "cardiomegaly", "activation": 0.85},
                    {"feature_id": 22, "name": "pleural effusion", "activation": 0.72},
                ],
                "pseudo_report": "The model identified: cardiomegaly (0.85), pleural effusion (0.72)",
            },
            {
                "image_id": "img_002",
                "top_k_concepts": [
                    {"feature_id": 5, "name": "normal", "activation": 0.91},
                ],
                "pseudo_report": "The model identified: normal (0.91)",
            },
        ]
        with open(self.results_dir / "sample_explanations.json", "w") as f:
            json.dump(explanations, f)

        # Create mock indiana_reports.csv  (uid + findings + impression)
        import pandas as pd
        reports_df = pd.DataFrame({
            "uid": ["1001", "1002"],
            "findings": [
                "The heart is enlarged. Bilateral pleural effusions.",
                "No acute cardiopulmonary abnormality.",
            ],
            "impression": [
                "Cardiomegaly with effusions.",
                "Normal.",
            ],
        })
        reports_df.to_csv(self.data_dir / "indiana_reports.csv", index=False)

        # Create mock indiana_projections.csv  (filename → uid)
        projections_df = pd.DataFrame({
            "filename": ["img_001", "img_002"],
            "uid": ["1001", "1002"],
        })
        projections_df.to_csv(self.data_dir / "indiana_projections.csv", index=False)

        # Patch module-level paths
        self._orig_explanations = judge_module.EXPLANATIONS_PATH
        self._orig_reports = judge_module.REPORTS_CSV_PATH
        self._orig_projections = judge_module.PROJECTIONS_CSV_PATH
        self._orig_output = judge_module.OUTPUT_CSV_PATH
        self._orig_checkpoint = judge_module.CHECKPOINT_PATH
        self._orig_scores = judge_module.SCORES_JSON_PATH
        self._orig_coverage = judge_module.COVERAGE_JSON_PATH

        judge_module.EXPLANATIONS_PATH = self.results_dir / "sample_explanations.json"
        judge_module.REPORTS_CSV_PATH = self.data_dir / "indiana_reports.csv"
        judge_module.PROJECTIONS_CSV_PATH = self.data_dir / "indiana_projections.csv"
        judge_module.OUTPUT_CSV_PATH = self.results_dir / "aligned_scores.csv"
        judge_module.CHECKPOINT_PATH = self.results_dir / ".judge_checkpoint.json"
        judge_module.SCORES_JSON_PATH = self.results_dir / "judge_scores.json"
        judge_module.COVERAGE_JSON_PATH = self.results_dir / "judge_coverage.json"

    def tearDown(self):
        # Restore original paths
        judge_module.EXPLANATIONS_PATH = self._orig_explanations
        judge_module.REPORTS_CSV_PATH = self._orig_reports
        judge_module.PROJECTIONS_CSV_PATH = self._orig_projections
        judge_module.OUTPUT_CSV_PATH = self._orig_output
        judge_module.CHECKPOINT_PATH = self._orig_checkpoint
        judge_module.SCORES_JSON_PATH = self._orig_scores
        judge_module.COVERAGE_JSON_PATH = self._orig_coverage

        # Clean up temp dir
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_pipeline(self):
        """Run the full evaluate() with mocked LLM, verify output CSV."""
        import pandas as pd

        # Mock pipeline: always returns "Aligned"
        mock_pipe = _make_mock_pipeline("Aligned")
        with patch.object(judge_module, "get_pipeline", return_value=mock_pipe):
            judge_module.evaluate(resume=False, batch_save_every=10)

        # Verify output CSV was created
        output_path = self.results_dir / "aligned_scores.csv"
        self.assertTrue(output_path.exists(), "aligned_scores.csv was not created")

        df = pd.read_csv(output_path)
        # We had 2 images: img_001 (2 concepts) + img_002 (1 concept) = 3 rows
        self.assertEqual(len(df), 3)
        self.assertListEqual(
            sorted(df.columns.tolist()),
            sorted(["image_id", "feature_id", "concept", "activation", "verdict", "raw_response"]),
        )
        # All verdicts should be "Aligned" since our mock always returns that
        self.assertTrue((df["verdict"] == "Aligned").all())

    def test_resume_skips_done(self):
        """After a partial run, resuming should skip already-evaluated pairs."""
        import pandas as pd

        # Pre-populate checkpoint with 1 of the 3 pairs
        checkpoint_records = [
            {
                "image_id": "img_001",
                "feature_id": 10,
                "concept": "cardiomegaly",
                "activation": 0.85,
                "verdict": "Aligned",
                "raw_response": "Aligned",
            }
        ]
        judge_module.save_checkpoint(checkpoint_records)

        call_count = 0

        def counting_pipe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [
                {
                    "generated_text": [
                        {"role": "system", "content": "..."},
                        {"role": "user", "content": "..."},
                        {"role": "assistant", "content": "Unaligned"},
                    ]
                }
            ]

        mock_pipe = MagicMock(side_effect=counting_pipe)
        with patch.object(judge_module, "get_pipeline", return_value=mock_pipe):
            judge_module.evaluate(resume=True, batch_save_every=10)

        # Should have only evaluated 2 remaining pairs (not 3)
        self.assertEqual(call_count, 2)

        df = pd.read_csv(self.results_dir / "aligned_scores.csv")
        self.assertEqual(len(df), 3)  # 1 from checkpoint + 2 new


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
