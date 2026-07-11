"""
datasets/download_iu_xray.py — Download and stage the IU X-Ray dataset from Kaggle.

Downloads ``raddar/chest-xrays-indiana-university`` via kagglehub and
organises the raw files into the directory layout expected by
``src/config.py`` (``VLMConfig``):

    data/iu_xray/images/images_normalized/  ← PNG radiographs
    data/iu_xray/reports/                   ← CSV clinical reports (indiana_reports.csv)

Usage::

    python datasets/download_iu_xray.py           # defaults
    python datasets/download_iu_xray.py --dry-run # preview without moving files

Requires:
    pip install kagglehub
    KAGGLE_USERNAME and KAGGLE_KEY environment variables (or ~/.kaggle/kaggle.json)
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    """Paths and settings for the IU X-Ray download step.

    Args:
        project_root: Absolute path to the repository root.
        kaggle_slug: Kaggle dataset identifier ``<owner>/<dataset>``.
        dry_run: If True, print actions without copying any files.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    kaggle_slug: str = "raddar/chest-xrays-indiana-university"
    dry_run: bool = False

    @property
    def data_dir(self) -> Path:
        """Root staging directory inside the repository."""
        return self.project_root / "data" / "iu_xray"

    @property
    def images_dest(self) -> Path:
        """Destination for normalised PNG radiographs (matches VLMConfig)."""
        return self.data_dir / "images" / "images_normalized"

    @property
    def reports_dest(self) -> Path:
        """Destination for CSV clinical reports (matches VLMConfig)."""
        return self.data_dir / "reports"


def _ensure_kagglehub() -> None:
    """Raise ImportError with install hint if kagglehub is missing."""
    try:
        import kagglehub  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the dataset.\n"
            "Install it with:  pip install kagglehub"
        ) from exc


def _download_raw(slug: str) -> Path:
    """Download dataset via kagglehub and return the local cache path.

    Args:
        slug: Kaggle dataset identifier, e.g. ``raddar/chest-xrays-indiana-university``.

    Returns:
        Path to the downloaded dataset directory.
    """
    import kagglehub

    logger.info("Downloading Kaggle dataset '%s' …", slug)
    raw_path = Path(kagglehub.dataset_download(slug))
    logger.info("Dataset cached at: %s", raw_path)
    return raw_path


def _copy_files(
    src: Path,
    dest: Path,
    pattern: str,
    dry_run: bool,
) -> int:
    """Recursively copy files matching *pattern* from *src* into *dest*.

    Args:
        src: Source directory to search recursively.
        dest: Destination directory (created if absent).
        pattern: Glob pattern relative to *src*, e.g. ``**/*.png``.
        dry_run: If True, log intended copies without executing them.

    Returns:
        Number of files copied (or that would be copied in dry-run mode).
    """
    matches = list(src.rglob(pattern))
    if not matches:
        logger.warning("No files matched pattern '%s' in %s", pattern, src)
        return 0

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_file in matches:
        dst_file = dest / src_file.name
        if dry_run:
            logger.info("[dry-run] would copy %s → %s", src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)
        count += 1

    return count


def download_and_stage(cfg: DownloadConfig) -> None:
    """Download the IU X-Ray dataset and stage it for the pipeline.

    Args:
        cfg: Download configuration (paths, slug, dry-run flag).

    Raises:
        ImportError: If kagglehub is not installed.
        FileNotFoundError: If the downloaded archive contains no images or reports.
    """
    _ensure_kagglehub()

    raw_path = _download_raw(cfg.kaggle_slug)

    n_images = _copy_files(
        src=raw_path,
        dest=cfg.images_dest,
        pattern="**/*.png",
        dry_run=cfg.dry_run,
    )
    logger.info("Images staged: %d PNG → %s", n_images, cfg.images_dest)

    if n_images == 0:
        raise FileNotFoundError(
            f"No PNG files found under {raw_path}. "
            "Check the Kaggle dataset structure or your download."
        )

    n_reports = _copy_files(
        src=raw_path,
        dest=cfg.reports_dest,
        pattern="**/*.csv",
        dry_run=cfg.dry_run,
    )
    logger.info("Reports staged: %d CSV → %s", n_reports, cfg.reports_dest)

    if n_reports == 0:
        raise FileNotFoundError(
            f"No CSV report files found under {raw_path}. "
            "Check the Kaggle dataset structure or your download."
        )

    if not cfg.dry_run:
        logger.info(
            "Dataset ready.\n  images : %s\n  reports: %s",
            cfg.images_dest,
            cfg.reports_dest,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download IU X-Ray from Kaggle and stage for the SAE pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file movements without copying anything.",
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

    kwargs: dict = {"dry_run": args.dry_run}
    if args.project_root is not None:
        kwargs["project_root"] = args.project_root

    cfg = DownloadConfig(**kwargs)
    download_and_stage(cfg)
