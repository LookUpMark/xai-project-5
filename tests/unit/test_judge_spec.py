"""test_judge_spec.py — Tests for the spec-driven LLM-judge wiring (P2.5).

Covers the pieces made dataset-specific by the spec-driven refactor: the
per-dataset judge prompts (IU English chest, PadChest Spanish incl. the
report-truncation caveat, docs/FINDINGS.md A2), the IU 2-hop report lookup, and
that both specs wire prompt + lookup. The verdict parser and LangGraph
end-to-end are exercised by tests/test_llm_judge.py. Light deps only
(xai_datasets — no langgraph/torch import here).
"""

import csv
from pathlib import Path

from xai_datasets.iu_xray import IU_XRAY_JUDGE_PROMPT, build_iu_xray_report_lookup
from xai_datasets.padchest import PADCHEST_JUDGE_PROMPT
from xai_datasets.spec import IU_XRAY_SPEC, PADCHEST_SPEC


class TestJudgePrompts:
    def test_iu_prompt_placeholders_and_verdict(self):
        assert "{report}" in IU_XRAY_JUDGE_PROMPT
        assert "{pseudo_report}" in IU_XRAY_JUDGE_PROMPT
        assert "Verdict:" in IU_XRAY_JUDGE_PROMPT
        assert "Aligned" in IU_XRAY_JUDGE_PROMPT and "Unaligned" in IU_XRAY_JUDGE_PROMPT

    def test_padchest_prompt_is_spanish_with_truncation_note(self):
        assert "{report}" in PADCHEST_JUDGE_PROMPT
        assert "{pseudo_report}" in PADCHEST_JUDGE_PROMPT
        # Verdict labels stay English (shared parser).
        assert "Verdict:" in PADCHEST_JUDGE_PROMPT
        # Spanish cues.
        assert "radiolog" in PADCHEST_JUDGE_PROMPT.lower()
        # The anonymisation/truncation caveat (FINDINGS A2) must be stated so the
        # judge infers truncated words.
        lower = PADCHEST_JUDGE_PROMPT.lower()
        assert "truncad" in lower or "abbreviad" in lower

    def test_specs_wire_their_prompts(self):
        assert IU_XRAY_SPEC.judge_prompt is IU_XRAY_JUDGE_PROMPT
        assert PADCHEST_SPEC.judge_prompt is PADCHEST_JUDGE_PROMPT
        assert IU_XRAY_SPEC.judge_prompt   # non-empty
        assert PADCHEST_SPEC.judge_prompt  # non-empty

    def test_both_specs_have_callable_report_lookup(self):
        assert callable(IU_XRAY_SPEC.build_report_lookup)
        assert callable(PADCHEST_SPEC.build_report_lookup)


class TestIUXrayReportLookup:
    """The IU 2-hop bridge: filename -> uid (projections) -> findings+impression."""

    @staticmethod
    def _write_csvs(iu_dir):
        iu_dir = Path(iu_dir)
        with (iu_dir / "indiana_reports.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["uid", "findings", "impression"])
            w.writeheader()
            w.writerow({"uid": "1", "findings": "Heart enlarged.", "impression": "Cardiomegaly."})
            w.writerow({"uid": "2", "findings": "Clear lungs.", "impression": ""})
        with (iu_dir / "indiana_projections.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["filename", "uid"])
            w.writeheader()
            w.writerow({"filename": "img_1.png", "uid": "1"})
            w.writerow({"filename": "img_2.png", "uid": "2"})
            w.writerow({"filename": "img_3.png", "uid": "999"})  # uid without a report

    def test_two_hop_join(self, tmp_path):
        self._write_csvs(tmp_path)
        lookup = build_iu_xray_report_lookup(tmp_path)
        assert lookup["img_1.png"] == "Heart enlarged. Cardiomegaly."
        assert lookup["img_2.png"] == "Clear lungs."
        # img_3 maps to a uid with no report -> dropped.
        assert "img_3.png" not in lookup
