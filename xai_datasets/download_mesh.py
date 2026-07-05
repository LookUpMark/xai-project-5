"""download_mesh.py — Download MeSH XML (NLM, free, no license).

MeSH is redistributed **freely by NLM without a UMLS license**, so (unlike the
UMLS Metathesaurus / API) there is no access gate. This fetches the MeSH XML
descriptor file (gzipped ``desc<year>.gz``), which
``mesh_support.load_and_filter_mesh`` parses to build the ROCOv2 radiology
vocabulary.

.. note::
   NLM **discontinued the ASCII ``d<year>.bin`` serialization in January 2026**.
   The previous ASCII URL (``/projects/mesh/<year>/ascii/d<year>.bin``) now serves
   a 200-OK HTML "NLM URL Not Found" page — which is why this module now downloads
   the **XML** distribution (``/projects/mesh/MESH_FILES/xmlmesh/desc<year>.gz``)
   and **validates the content** before accepting it (rejecting HTML error pages
   and truncated/tiny files), so a bad URL fails loudly instead of silently saving
   a corrupt file downstream.

Stages to ``data/mesh/desc<year>.gz`` (~17 MB on disk; ~300 MB uncompressed).

Usage::
    python xai_datasets/download_mesh.py
    python xai_datasets/download_mesh.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Current MeSH production year (the year at the stable ``MESH_FILES/xmlmesh/`` path;
# the date is July 2026, so 2026 is live). Older years move to NLM's archive path.
MESH_YEAR_DEFAULT = "2026"
MESH_XML_GZ_URL_TMPL = (
    "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.gz"
)
# A correct download is gzipped, ~17 MB, and gunzips to a MeSH XML descriptor set.
_MIN_GZ_MB = 5.0


@dataclass(frozen=True)
class MeSHDownloadConfig:
    """Paths and settings for the MeSH download step."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    year: str = MESH_YEAR_DEFAULT
    dry_run: bool = False

    @property
    def mesh_dir(self) -> Path:
        return self.project_root / "data" / "mesh"

    @property
    def dest(self) -> Path:
        return self.mesh_dir / f"desc{self.year}.gz"

    @property
    def url(self) -> str:
        return MESH_XML_GZ_URL_TMPL.format(year=self.year)


def _validate_mesh_gz(path: Path, url: str) -> None:
    """Reject non-MeSH downloads (HTML error pages, truncated/tiny files).

    The original bug was a 200-OK HTML error page saved silently as the MeSH file,
    so we validate the bytes, not the HTTP status. Raises ``SystemExit`` (with the
    URL) on anything that is not a gzip decompressing to a MeSH ``DescriptorRecordSet``.
    """
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head[:2] != b"\x1f\x8b":
        # Not gzip -> almost certainly an HTML error page. Give a helpful message.
        sample = head.decode("utf-8", "replace").lower()
        with open(path, "rb") as fh:
            sample = fh.read(512).decode("utf-8", "replace").lower()
        if sample.startswith(("<!doctype", "<html")) or "<html" in sample[:200]:
            raise SystemExit(
                f"Downloaded {path} is an HTML page, not MeSH (URL: {url}). "
                "The MeSH ASCII format was discontinued in 2026 — the XML/gz path "
                "should be used; check the year/URL."
            )
        raise SystemExit(
            f"Downloaded {path} is not a gzip file (magic {head[:2].hex()!r}); URL: {url}."
        )
    # Gzip ok -> confirm it gunzips to MeSH XML and is plausibly complete.
    with gzip.open(path, "rb") as gz:
        xml_head = gz.read(2048)
    if b"DescriptorRecordSet" not in xml_head:
        raise SystemExit(
            f"Downloaded {path} gunzips but is not a MeSH descriptor set "
            f"(head: {xml_head[:120]!r}); URL: {url}."
        )
    size_mb = path.stat().st_size / 1e6
    if size_mb < _MIN_GZ_MB:
        raise SystemExit(
            f"Downloaded {path} is only {size_mb:.2f} MB — the full MeSH descriptor "
            f"set is ~17 MB; the download is likely truncated. Re-run to retry. URL: {url}."
        )
    logger.info("validated: MeSH XML (gzip), %.1f MB", size_mb)


def download_mesh(cfg: MeSHDownloadConfig) -> Path:
    """Download + validate the MeSH XML descriptor file (skip if already present).

    Streams to a ``.part`` temp file, validates it (gzip + MeSH XML + min size),
    then atomically moves it into place — so a failed/bad download never leaves a
    corrupt ``desc<year>.gz`` for the parser to choke on later.
    """
    if cfg.dest.exists():
        logger.info("Already present, skipping: %s", cfg.dest)
        return cfg.dest
    if cfg.dry_run:
        logger.info("[dry-run] would download %s -> %s", cfg.url, cfg.dest)
        return cfg.dest

    cfg.mesh_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.dest.with_suffix(cfg.dest.suffix + ".part")
    logger.info("Downloading MeSH XML (gzipped): %s ...", cfg.url)
    try:
        with urllib.request.urlopen(cfg.url) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        _validate_mesh_gz(tmp, cfg.url)
        os.replace(tmp, cfg.dest)  # atomic: bad download never lands at dest
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    logger.info("  saved %s (%.1f MB gzipped)", cfg.dest, cfg.dest.stat().st_size / 1e6)
    return cfg.dest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MeSH XML (NLM, free, no license) and stage it for the pipeline."
    )
    p.add_argument(
        "--year",
        type=str,
        default=MESH_YEAR_DEFAULT,
        help=f"MeSH release year (default: {MESH_YEAR_DEFAULT}).",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview without downloading.")
    p.add_argument(
        "--project-root", type=Path, default=None, help="Override repository root."
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    kwargs: dict = {"year": args.year, "dry_run": args.dry_run}
    if args.project_root is not None:
        kwargs["project_root"] = args.project_root
    download_mesh(MeSHDownloadConfig(**kwargs))
