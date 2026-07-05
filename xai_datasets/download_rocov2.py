"""download_rocov2.py — Download and stage ROCOv2 (CC BY, no secrets).

ROCOv2 (Rückert et al., Sci. Data 2024) is distributed on Zenodo (record 10821435)
under CC BY / CC BY-NC — **redistributable**, the Zenodo record id is hard-coded and 
the file list is fetched from the public API.

Stages into the layout the pipeline expects (``xai_datasets/rocov2.py`` / spec)::
    data/rocov2/images/             <- all *.jpg (flattened across train/valid/test)
    data/rocov2/captions.csv        <- {train,valid,test}_captions.csv concatenated
    data/rocov2/concepts_manual.csv <- {train,valid,test}_concepts_manual.csv concatenated
    data/rocov2/cui_mapping.csv     <- copied as-is
    data/rocov2/_raw/               <- downloaded archives + per-split CSVs (cache)

Usage::
    python xai_datasets/download_rocov2.py
    python xai_datasets/download_rocov2.py --dry-run
    python xai_datasets/download_rocov2.py --max-image-archives 1   # stage a subset only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ZENODO_RECORD_ID = "10821435"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Per-split CSVs that get concatenated into a single file at the rocov2 root.
# Only captions are staged: per-image CUIs are NOT used (building the vocab from
# them would be circular with any CUI-based eval, and the free UMLS CUI crosswalk
# is unavailable; the vocab comes from external MeSH instead). See docs/FINDINGS.md.
_CONCAT_SPECS = {
    "captions.csv": ("train", "valid", "test"),  # output name -> splits to concat
}


@dataclass(frozen=True)
class ROCOv2DownloadConfig:
    """Paths and settings for the ROCOv2 download step."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    dry_run: bool = False
    max_image_archives: int | None = None  # cap the number of *_images.zip to stage

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "rocov2"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "_raw"

    @property
    def images_dest(self) -> Path:
        return self.data_dir / "images"


# ---------------------------------------------------------------------------
# Zenodo file listing + download
# ---------------------------------------------------------------------------

def _list_zenodo_files() -> dict[str, str]:
    """Fetch the record's file list from the Zenodo API -> {filename: download_url}."""
    logger.info("Fetching Zenodo record %s file list ...", ZENODO_RECORD_ID)
    with urllib.request.urlopen(ZENODO_API) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    files: dict[str, str] = {}
    for entry in data.get("files", []):
        key = entry.get("key")
        url = entry.get("links", {}).get("self")
        if key and url:
            files[key] = url
    logger.info("Zenodo record lists %d files.", len(files))
    return files


def _download(url: str, dest: Path, dry_run: bool) -> Path:
    """Stream-download *url* into *dest* (skip if already present)."""
    if dest.exists():
        logger.info("Already present, skipping download: %s", dest)
        return dest
    if dry_run:
        logger.info("[dry-run] would download %s -> %s", url, dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", dest.name)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    logger.info("  saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _extract(archive: Path, raw_dir: Path, dry_run: bool) -> None:
    """Extract a .zip / .tar.gz / .tgz / .tar into *raw_dir*."""
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
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to extract %s: %s", archive, exc)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _stage_images(raw_dir: Path, images_dest: Path, dry_run: bool) -> int:
    """Flatten every *.jpg found under *raw_dir* into *images_dest*."""
    jpgs = sorted(raw_dir.rglob("*.jpg"))
    if not dry_run:
        images_dest.mkdir(parents=True, exist_ok=True)
    for src in jpgs:
        dst = images_dest / src.name
        if dry_run:
            continue
        shutil.copy2(src, dst)
    return len(jpgs)


def _concat_csvs(raw_dir: Path, out_name: str, splits: tuple[str, ...], dest: Path, dry_run: bool) -> int:
    """Concatenate per-split ``{split}_<out_name>`` CSVs into a single *dest* file."""
    sources = [raw_dir / f"{split}_{out_name}" for split in splits]
    sources = [s for s in sources if s.exists()]
    if not sources:
        logger.warning("No per-split CSVs found for %s (expected %s)", out_name, sources)
        return 0
    if dry_run:
        logger.info("[dry-run] would concat %s -> %s", [s.name for s in sources], dest)
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(dest, "w", newline="", encoding="utf-8") as out_fh:
        writer = None
        for src in sources:
            with open(src, newline="", encoding="utf-8") as in_fh:
                reader = csv.reader(in_fh)
                header = next(reader, None)
                if writer is None:
                    writer = csv.writer(out_fh)
                    if header is not None:
                        writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1
    return n_rows


def _copy_file(raw_dir: Path, name: str, dest: Path, dry_run: bool) -> bool:
    src = raw_dir / name
    if not src.exists():
        logger.warning("File not found in extracted tree: %s", src)
        return False
    if dry_run:
        logger.info("[dry-run] would copy %s -> %s", name, dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def download_and_stage(cfg: ROCOv2DownloadConfig) -> None:
    files = _list_zenodo_files()
    if not files:
        raise SystemExit(f"Zenodo record {ZENODO_RECORD_ID} returned no files.")

    if not cfg.dry_run:
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download + extract the image archives (the big ones).
    image_archives = sorted(k for k in files if k.endswith("_images.zip"))
    if cfg.max_image_archives is not None:
        image_archives = image_archives[: cfg.max_image_archives]
    logger.info("Image archives to fetch: %d", len(image_archives))
    for key in image_archives:
        archive = _download(files[key], cfg.raw_dir / key, cfg.dry_run)
        _extract(archive, cfg.raw_dir, cfg.dry_run)

    # 2. Download the per-split captions CSVs (small).
    needed_csvs = [f"{split}_captions.csv" for split in ("train", "valid", "test")]
    for key in needed_csvs:
        if key in files:
            _download(files[key], cfg.raw_dir / key, cfg.dry_run)

    # 3. Stage: flatten images, concat per-split captions into captions.csv.
    n_img = _stage_images(cfg.raw_dir, cfg.images_dest, cfg.dry_run)
    logger.info("Staged %d images -> %s", n_img, cfg.images_dest)
    for out_name, splits in _CONCAT_SPECS.items():
        n = _concat_csvs(cfg.raw_dir, out_name, splits, cfg.data_dir / out_name, cfg.dry_run)
        logger.info("Staged %s (%d rows) -> %s", out_name, n, cfg.data_dir / out_name)

    if not cfg.dry_run:
        logger.info(
            "ROCOv2 ready.\n  images  : %s\n  captions: %s",
            cfg.images_dest,
            cfg.data_dir / "captions.csv",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ROCOv2 (Zenodo, CC BY) and stage it for the SAE pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview downloads/extraction/staging without writing anything.",
    )
    parser.add_argument(
        "--max-image-archives",
        type=int,
        default=None,
        help="Cap the number of *_images.zip archives to fetch (subset/smoke test).",
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
    if args.max_image_archives is not None:
        kwargs["max_image_archives"] = args.max_image_archives
    if args.project_root is not None:
        kwargs["project_root"] = args.project_root

    download_and_stage(ROCOv2DownloadConfig(**kwargs))
