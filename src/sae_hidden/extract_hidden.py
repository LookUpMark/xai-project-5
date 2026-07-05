"""extract_hidden.py — Path A: extract the 768-d CLS hidden state (pre-projection).

Grabs ``model.vision_model(pixel_values=...).last_hidden_state[:, 0, :]`` — the CLS
token of BiomedCLIP's ViT *before* the frozen ``visual_projection`` (768->512). The
SAE therefore factorises the rich pre-projection residual stream (Paradigm B in
ML-AUDIT-2026-06-25), not the lossy 512-d projected space.

Input is RAW (no per-sample L2 norm), per SAE-on-residual-stream literature. Output
is cached at embeddings/standard_hidden/visual_embeddings_768.pt; the group-aware
train/test split reuses utils.split_embeddings so no radiograph study straddles both
partitions (the fix from ML-AUDIT-2026-06-23).

Run:
    python src/sae_hidden/extract_hidden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))         # src/  -> config, utils, sae_hidden
sys.path.insert(0, str(_HERE.parent.parent))  # repo root -> xai_datasets

import config
import utils
from augmentation.transforms import get_safe_cxr_transforms
from embedding_extraction.extract_embeddings import (
    _WorkerResizedDataset,
    _autocast_ctx,
    _dataloader_kwargs,
)
from sae_hidden.reports import md_table, write_report
from xai_datasets.spec import get_dataset

log = utils.setup_logging(__name__)


def extract_hidden_embeddings(model, processor, dataset, vlm_config, augmented: bool = False):
    """Extract raw 768-d CLS tokens pre-projection.

    Args:
        augmented: when True, encode ``config.augmentation.num_augmentations``
            augmented views per image (rotation + light crop, see
            ``augmentation.transforms``). Every view keeps the ORIGINAL image's
            basename as its id, so ``utils.split_embeddings`` still groups all
            views of a radiograph study into one partition (no train/test leakage).

    Returns:
        (Tensor(N, 768), list[str]) — embeddings and row-aligned PNG basenames.
    """
    loader = DataLoader(dataset, **_dataloader_kwargs(vlm_config))

    # standard: one view (the originals); augmented: n_views random augmentations.
    aug_transform = get_safe_cxr_transforms(config.augmentation) if augmented else None
    n_views = config.augmentation.num_augmentations if augmented else 1

    all_embeddings, all_ids = [], []
    model.eval()
    with torch.no_grad():
        for batch_images, batch_paths in loader:
            view_groups = (
                [[aug_transform(img) for img in batch_images] for _ in range(n_views)]
                if augmented else [list(batch_images)]
            )
            for group in view_groups:
                inputs = processor.image_processor(
                    images=group, return_tensors="pt"
                ).to(vlm_config.device)
                # CLS token of the last hidden state: (B, 197, 768) -> (B, 768), RAW.
                # fp16 autocast on CUDA (no-op on MPS/CPU); cast back to fp32 before
                # save so downstream SAE training sees no dtype change.
                with _autocast_ctx(vlm_config):
                    out = model.vision_model(pixel_values=inputs["pixel_values"])
                    cls = out.last_hidden_state[:, 0, :]
                all_embeddings.append(cls.float().cpu())
                # Original basename -> augmented views share the study key (no leakage).
                all_ids.extend(Path(p).name for p in batch_paths)

    embeddings = torch.cat(all_embeddings)
    if embeddings.shape[1] != config.sae_hidden.activation_dim:
        raise RuntimeError(
            f"Expected CLS dim {config.sae_hidden.activation_dim}, "
            f"got {embeddings.shape[1]}"
        )
    return embeddings, all_ids


def run(augmented: bool = False) -> Path:
    """Extract + split + report. Returns the visual embeddings path.

    Args:
        augmented: extract augmented views (see :func:`extract_hidden_embeddings`).
    """
    utils.set_global_seed(config.training.split_seed)  # F-009: deterministic extraction
    spec = get_dataset(config.active_dataset.name)  # dataset-aware (image_dir, dataset class, group key)
    out_path = config.paths.hidden_visual_embeddings_path
    ids_path = config.paths.hidden_visual_image_ids_path
    report_path = config.paths.hidden_results_dir / "REPORT_extraction.md"

    if out_path.exists():
        embeddings = utils.load_tensor(out_path)
        with open(ids_path) as f:
            all_ids = json.load(f)
        log.info(f"768-d embeddings already present ({embeddings.shape}); skipping extraction")
    else:
        model, processor = utils.load_vlm(config.vlm)
        image_dir = spec.image_dir
        dataset = _WorkerResizedDataset(spec.image_dataset_cls(image_dir))
        mode = f"augmented (x{config.augmentation.num_augmentations})" if augmented else "standard"
        log.info(f"Extracting 768-d CLS hidden state for {len(dataset)} images "
                 f"[{mode}, dataset={spec.name}, dir={image_dir}]...")
        # config.vlm is already device-tuned (batch=64, workers=4, use_half on cuda);
        # _WorkerResizedDataset downscales in the worker so forking is host-RAM-safe
        # (medical images are 1024-2500px; the ViT-B/16 processor resizes to 224 next).
        embeddings, all_ids = extract_hidden_embeddings(
            model, processor, dataset, config.vlm, augmented=augmented
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, out_path)
        with open(ids_path, "w") as f:
            json.dump(all_ids, f)
        # Free the VLM — only extraction needs it.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Group-aware split into train/test_768 + sidecars. group_key_fn is dataset-aware
    # (IU/PadChest: study key; ROCOv2: None => random split, figures are independent).
    group_key_fn = spec.make_group_key_fn()
    log.info(f"Creating train/test split ({'study-level' if group_key_fn else 'random'})...")
    train_emb, test_emb = utils.split_embeddings(
        source_path=out_path,
        train_path=config.paths.hidden_train_embeddings_path,
        test_path=config.paths.hidden_test_embeddings_path,
        train_ratio=config.training.train_split_ratio,
        seed=config.training.split_seed,
        group_key_fn=group_key_fn,
        source_ids_path=ids_path,
        train_ids_path=config.paths.hidden_train_image_ids_path,
        test_ids_path=config.paths.hidden_test_image_ids_path,
    )

    norms = embeddings.norm(dim=1)
    mode = f"augmented (x{config.augmentation.num_augmentations})" if augmented else "standard"
    summary = (
        f"Extracted {embeddings.shape[0]} raw 768-d CLS tokens (pre-projection, {mode}) "
        f"and split them at the radiograph-study level. CLS dim = {embeddings.shape[1]} "
        f"(confirms pre-projection path)."
    )
    sections = [
        (
            "Extraction",
            md_table(
                ["property", "value"],
                [
                    ["shape", str(tuple(embeddings.shape))],
                    ["mode", mode],
                    ["dtype", str(embeddings.dtype)],
                    ["activation_dim (config)", str(config.sae_hidden.activation_dim)],
                    ["L2-norm mean / std", f"{norms.mean():.4f} / {norms.std():.4f}"],
                    ["norm min / max", f"{norms.min():.4f} / {norms.max():.4f}"],
                ],
            ),
        ),
        (
            "Train / test split (study-level, no leakage)",
            md_table(
                ["split", "samples", "fraction"],
                [
                    ["train", train_emb.shape[0], f"{train_emb.shape[0]/len(embeddings):.1%}"],
                    ["test", test_emb.shape[0], f"{test_emb.shape[0]/len(embeddings):.1%}"],
                    ["total", len(embeddings), "100.0%"],
                ],
            ),
        ),
        (
            "Outputs",
            md_table(
                ["file", "shape"],
                [
                    ["visual_embeddings_768.pt", str(tuple(embeddings.shape))],
                    ["train_embeddings_768.pt", str(tuple(train_emb.shape))],
                    ["test_embeddings_768.pt", str(tuple(test_emb.shape))],
                ],
            ),
        ),
        (
            "Notes",
            "Raw activations (no per-sample L2 norm) — SAE-on-residual-stream "
            "convention. Group split enforced inside utils.split_embeddings "
            "(anti-leak assertion).",
        ),
    ]
    write_report(report_path, "Path A — 768-d Hidden-State Extraction", sections, summary)
    log.info(f"Report: {report_path}")
    return out_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
