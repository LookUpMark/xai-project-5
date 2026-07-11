"""download_padchest.py — Download and stage PadChest (BIMCV), license-safe.

PadChest (>160k chest X-rays + Spanish free-text radiology reports) is
distributed by the Biomedical Imaging Lab Valencia (BIMCV) under terms that
**forbid redistribution**. The download URL(s) MUST therefore be supplied at
runtime — they are NEVER hard-coded or committed. Provide them via, in priority
order:

  1. the ``--url`` CLI flag (repeatable for multi-part archives);
  2. the ``PADCHEST_DOWNLOAD_URL`` environment variable (comma-separated for
     multi-part archives);
  3. a local ``.env`` file (gitignored — see ``.env.example``) holding
     ``PADCHEST_DOWNLOAD_URL=https://.../a.zip,https://.../b.zip``.

Stages into the layout the pipeline expects::

    data/padchest/images/   <- all *.png radiographs (flattened; ImageIDs unique)
    data/padchest/reports/  <- the labels+reports CSV(s)
    data/padchest/_raw/     <- downloaded archives + extracted tree (cache)

Usage::

    python xai_datasets/download_padchest.py
    python xai_datasets/download_padchest.py --url https://.../a.zip --url https://.../b.zip
    python xai_datasets/download_padchest.py --dry-run

NOTE: ``data/padchest/*`` is gitignored; only ``data/padchest/images/.gitkeep``
is tracked. Do NOT commit the data or the URLs.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Env var holding the (comma-separated) download URL(s). NEVER hard-coded.
ENV_VAR = "PADCHEST_DOWNLOAD_URL"


@dataclass(frozen=True)
class PadChestDownloadConfig:
    """Paths and settings for the PadChest download step.

    Args:
        project_root: Absolute path to the repository root.
        urls: Archive URL(s) from the CLI (env var is merged in separately).
        dry_run: If True, log actions without downloading/moving anything.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    urls: tuple[str, ...] = ()
    dry_run: bool = False

    @property
    def data_dir(self) -> Path:
        """Root staging directory inside the repository."""
        return self.project_root / "data" / "padchest"

    @property
    def raw_dir(self) -> Path:
        """Cache for downloaded archives + extracted tree."""
        return self.data_dir / "_raw"

    @property
    def images_dest(self) -> Path:
        """Flattened PNG radiographs (one file per ImageID)."""
        return self.data_dir / "images"

    @property
    def reports_dest(self) -> Path:
        """Labels + Spanish-report CSV(s)."""
        return self.data_dir / "reports"


def _resolve_urls(cli_urls: tuple[str, ...]) -> tuple[str, ...]:
    """Gather URLs from CLI then env var / .env (comma-separated). Never hard-coded.

    Args:
        cli_urls: URLs passed via ``--url``.

    Returns:
        De-duplicated URL tuple. Empty if none provided.
    """
    # Load a local .env (gitignored) if python-dotenv is installed.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    urls: list[str] = list(cli_urls)
    env_raw = os.environ.get(ENV_VAR, "").strip()
    if env_raw:
        urls.extend(u.strip() for u in env_raw.split(",") if u.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return tuple(unique)


def _archive_name(url: str) -> str:
    """Best-effort filename from a URL (strip query string)."""
    tail = url.rsplit("/", 1)[-1]
    name = tail.split("?", 1)[0]
    return name or "download.bin"


def _download(url: str, dest: Path, dry_run: bool) -> Path:
    """Stream-download *url* into *dest* (skip if already present).

    Args:
        url: Archive URL (presigned/BIMCV link).
        dest: Directory to download into.
        dry_run: If True, log only.

    Returns:
        Path of the (would-be) downloaded archive.
    """
    target = dest / _archive_name(url)
    if target.exists():
        logger.info("Already present, skipping download: %s", target)
        return target
    if dry_run:
        logger.info("[dry-run] would download %s -> %s", url, target)
        return target

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)
    with urllib.request.urlopen(url) as resp, open(target, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    logger.info("  saved %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def _extract(archive: Path, raw_dir: Path, dry_run: bool) -> None:
    """Extract a .zip / .tar.gz / .tgz / .tar into *raw_dir* (idempotent)."""
    if dry_run:
        logger.info("[dry-run] would extract %s", archive)
        return
    name = archive.name.lower()
    try:
        if name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(raw_dir)
        elif name.endswith(".tar"):
            with tarfile.open(archive, "r:") as tar:
                tar.extractall(raw_dir)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
        else:
            logger.warning("Unknown archive type, leaving as-is: %s", archive)
    except Exception as exc:  # noqa: BLE001 — log and continue with other archives
        logger.error("Failed to extract %s: %s", archive, exc)


def _stage(
    raw_dir: Path,
    images_dest: Path,
    reports_dest: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """Flatten PNGs to *images_dest* and copy CSVs to *reports_dest*.

    Returns:
        (n_images, n_reports) staged (or that would be, in dry-run).
    """
    pngs = sorted(raw_dir.rglob("*.png"))
    csvs = sorted(raw_dir.rglob("*.csv"))

    if not dry_run:
        images_dest.mkdir(parents=True, exist_ok=True)
        reports_dest.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for src in pngs:
        dst = images_dest / src.name
        if dry_run:
            logger.info("[dry-run] would stage image %s", src.name)
        else:
            shutil.copy2(src, dst)
        n_images += 1

    n_reports = 0
    for src in csvs:
        dst = reports_dest / src.name
        if dry_run:
            logger.info("[dry-run] would stage csv %s", src.name)
        else:
            shutil.copy2(src, dst)
        n_reports += 1

    return n_images, n_reports


def download_and_stage(cfg: PadChestDownloadConfig) -> None:
    """Download PadChest archives and stage images + reports for the pipeline.

    Args:
        cfg: Download configuration (URLs come from ``cfg.urls`` + the env var).

    Raises:
        SystemExit: If no URLs are provided (they must never be hard-coded).
    """
    urls = _resolve_urls(cfg.urls)
    if not urls:
        raise SystemExit(
            f"No PadChest URLs provided. Supply them via --url, the {ENV_VAR} "
            f"env var, or a .env file. URLs must NOT be committed (BIMCV license)."
        )

    logger.info("PadChest download: %d archive(s)", len(urls))
    if not cfg.dry_run:
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        archive = _download(url, cfg.raw_dir, cfg.dry_run)
        _extract(archive, cfg.raw_dir, cfg.dry_run)

    n_images, n_reports = _stage(
        cfg.raw_dir, cfg.images_dest, cfg.reports_dest, cfg.dry_run
    )
    logger.info("Staged: %d images -> %s", n_images, cfg.images_dest)
    logger.info("Staged: %d CSV(s)  -> %s", n_reports, cfg.reports_dest)

    if n_images == 0:
        logger.warning(
            "No PNGs staged — check the extracted tree in %s", cfg.raw_dir
        )
    if n_reports == 0:
        logger.warning(
            "No CSVs staged — check the extracted tree in %s", cfg.raw_dir
        )

    if not cfg.dry_run:
        logger.info(
            "PadChest ready.\n  images : %s\n  reports: %s",
            cfg.images_dest,
            cfg.reports_dest,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PadChest (BIMCV) and stage it for the SAE pipeline."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Archive URL (repeatable). Never committed (BIMCV license).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview downloads/extraction/staging without writing anything.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Override repository root (default: parent of this file's parent).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    kwargs: dict = {"urls": tuple(args.url), "dry_run": args.dry_run}
    if args.project_root is not None:
        kwargs["project_root"] = args.project_root

    download_and_stage(PadChestDownloadConfig(**kwargs))
