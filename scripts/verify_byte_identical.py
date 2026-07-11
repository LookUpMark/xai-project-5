"""verify_byte_identical.py — sha256 gate for refactors that must be lossless.

Confirms a code refactor (e.g. the Phase 0 ``DatasetSpec`` migration, or any
future change that should not alter behavior) leaves the on-disk artifacts
byte-identical. Workflow:

    # 1. Snapshot the CURRENT artifacts (before re-running anything):
    python scripts/verify_byte_identical.py snapshot embeddings/standard -o before.json

    # 2. Re-run extraction + split (and/or vocab) with the refactored code.
    #    Either overwrite in place, or write to a temp dir and point step 3 there.

    # 3. Snapshot again and compare:
    python scripts/verify_byte_identical.py snapshot embeddings/standard -o after.json
    python scripts/verify_byte_identical.py compare before.json after.json

``compare`` exits 0 only when the two snapshots have identical file sets AND
every shared file has the same sha256. Stdlib only — no project imports.

Run:
    python scripts/verify_byte_identical.py snapshot --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(directory: Path, out: Path) -> int:
    """Hash every file under *directory* (recursively) and write the map to *out*.

    Args:
        directory: Directory to snapshot (all files, any extension).
        out: Destination JSON (relative path -> sha256).

    Returns:
        0 on success.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    entries: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = path.relative_to(directory).as_posix()
            entries[rel] = _sha256(path)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"snapshot: {len(entries)} files under {directory} -> {out}")
    return 0


def compare(a: Path, b: Path) -> int:
    """Compare two snapshot JSONs and print a diff summary.

    Args:
        a: "before" snapshot.
        b: "after" snapshot.

    Returns:
        0 if file sets are identical and every shared file matches, else 1.
    """
    sa = json.loads(a.read_text(encoding="utf-8"))
    sb = json.loads(b.read_text(encoding="utf-8"))
    keys_a, keys_b = set(sa), set(sb)

    only_before = sorted(keys_a - keys_b)
    only_after = sorted(keys_b - keys_a)
    mismatched = sorted(k for k in keys_a & keys_b if sa[k] != sb[k])
    matched = len(keys_a & keys_b) - len(mismatched)

    print(f"matched: {matched}   mismatched: {len(mismatched)}   "
          f"only-before: {len(only_before)}   only-after: {len(only_after)}")
    for k in mismatched:
        print(f"  MISMATCH {k}")
    for k in only_before:
        print(f"  REMOVED  {k}")
    for k in only_after:
        print(f"  ADDED    {k}")

    ok = not (mismatched or only_before or only_after)
    print("RESULT: BYTE-IDENTICAL" if ok else "RESULT: DIFFERS")
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sha256 snapshot/compare to verify a refactor is byte-identical."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="hash all files under a dir -> JSON")
    p_snap.add_argument("directory", type=Path, help="directory to snapshot")
    p_snap.add_argument("-o", "--out", type=Path, required=True, help="output JSON path")

    p_cmp = sub.add_parser("compare", help="compare two snapshot JSONs")
    p_cmp.add_argument("before", type=Path, help="'before' snapshot JSON")
    p_cmp.add_argument("after", type=Path, help="'after' snapshot JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "snapshot":
        return snapshot(args.directory, args.out)
    if args.cmd == "compare":
        return compare(args.before, args.after)
    return 1  # unreachable (subparser required)


if __name__ == "__main__":
    sys.exit(main())
