"""Shared markdown report writer for the Path A (SAE-768) pipeline.

Keeps every stage's REPORT formatting in one place so the five scripts don't
each re-implement table/section scaffolding.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a markdown table from headers and row tuples."""
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def write_report(
    path: Path,
    title: str,
    sections: Sequence[tuple[str, str]],
    summary: str | None = None,
) -> Path:
    """Write a markdown report.

    Args:
        path: Destination .md path.
        title: Top-level ``#`` heading.
        sections: Ordered list of ``(heading, body)``; body is markdown text
            (tables produced via :func:`md_table`).
        summary: Optional one-paragraph executive summary under a Summary header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"_Generated: {date.today().isoformat()}_", ""]
    if summary:
        lines += ["## Summary", "", summary, ""]
    for heading, body in sections:
        lines += [f"## {heading}", "", body.strip(), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
