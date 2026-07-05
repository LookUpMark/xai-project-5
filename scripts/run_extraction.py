"""run_extraction.py — Extract BiomedCLIP visual + text embeddings (CLI).

Prerequisites:
    - The active dataset is staged (e.g. ``data/iu_xray/...`` or
      ``data/padchest/...``); see ``xai_datasets/download_*.py``.

Usage:
    python scripts/run_extraction.py                           # IU X-Ray, defaults
    python scripts/run_extraction.py --dataset padchest        # PadChest
    python scripts/run_extraction.py --dataset padchest --skip-text --batch-size 64
    python scripts/run_extraction.py --augmentation            # augmented views
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))   # src/  -> config, utils, embedding_extraction
sys.path.insert(0, str(_HERE.parent))           # repo root -> xai_datasets

import torch

import config
import utils
from embedding_extraction.extract_embeddings import (
    extract_text_embeddings,
    extract_visual_embeddings,
)
from xai_datasets.spec import get_dataset


def _load_biomedclip(vlm_cfg: config.VLMConfig):
    """Load BiomedCLIP (model + processor) with compatibility patches.

    Mirrors the notebook's loading cell. Patches: (1) CLIPConfig now rejects
    positional args but the cached BiomedCLIP config passes them; (2) BiomedCLIP
    custom code does ``from modeling_clip import *`` but ``__all__`` restricts
    exports; (3) the text tower checks ``config.is_decoder``; (4) the
    position_ids buffer is corrupted under PyTorch 2.12 (persistent=False bug).
    """
    from transformers import AutoModel, AutoProcessor
    from transformers.models.clip.configuration_clip import CLIPConfig
    import transformers.models.clip.modeling_clip as _clip_mod

    # (1) CLIPConfig positional-arg compat
    _orig_clip_init = CLIPConfig.__init__

    def _clip_init_compat(self, *args, **kwargs):
        if args:
            names = ["text_config", "vision_config", "projection_dim", "logit_scale_init_value"]
            for name, val in zip(names, args):
                if name not in kwargs:
                    kwargs[name] = val
        _orig_clip_init(self, **kwargs)

    CLIPConfig.__init__ = _clip_init_compat

    # (2) drop __all__ so BiomedCLIP's ``from modeling_clip import *`` works
    if hasattr(_clip_mod, "__all__"):
        del _clip_mod.__all__

    logger = utils.setup_logging(__name__)
    logger.info("Loading model: %s ...", vlm_cfg.model_name)
    processor = AutoProcessor.from_pretrained(vlm_cfg.processor_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(vlm_cfg.model_name, trust_remote_code=True)

    # (3) BiomedCLIP text tower checks config.is_decoder (BERT-style)
    if not hasattr(model.text_model.config, "is_decoder"):
        model.text_model.config.is_decoder = False

    # (4) fix corrupted position_ids buffer (PyTorch 2.12 + persistent=False)
    max_pos = model.text_model.config.max_position_embeddings
    model.text_model.embeddings.position_ids = torch.arange(max_pos).unsqueeze(0)

    model = model.to(vlm_cfg.device).eval()
    logger.info("Model on %s — params: %s", vlm_cfg.device, f"{sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract BiomedCLIP visual + text embeddings for a dataset."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=config.active_dataset.name,
        help=(
            f"Active dataset (default: {config.active_dataset.name}); must be a key "
            "in xai_datasets.spec.DATASETS (e.g. iu_xray, padchest). Re-routes "
            "outputs to embeddings/<dataset>/."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=config.vlm.device,
        help=f"Compute device (default: {config.vlm.device}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=config.vlm.batch_size,
        help=f"Batch size (default: {config.vlm.batch_size}).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=config.vlm.num_workers,
        help=f"DataLoader workers (default: {config.vlm.num_workers}; 0 on macOS).",
    )
    p.add_argument(
        "--augmentation",
        action="store_true",
        help="Generate augmented views (rotation + light crop) per image.",
    )
    p.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip report/caption text-embedding extraction (visual only).",
    )
    p.add_argument(
        "--no-half",
        action="store_true",
        help="Disable fp16 autocast (force fp32 forward). Default uses fp16 on CUDA.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Enable augmentation BEFORE select_dataset so the path routing
    # (embeddings/<dataset>/<aug>/) reflects it. The flag is read live in two
    # places (config.augmentation and the extract_embeddings alias), so set both.
    if args.augmentation:
        config.augmentation = config.AugmentationConfig(enabled=True)
        import embedding_extraction.extract_embeddings as _ee
        _ee.augmentation = config.augmentation

    # Re-route embedding paths to the selected dataset (must precede any path read).
    config.select_dataset(args.dataset)
    spec = get_dataset(args.dataset)

    vlm_cfg = config.VLMConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_half=not args.no_half,
    )
    emb_cfg = config.EmbeddingConfig(
        image_dir=str(spec.image_dir),
        reports_dir=str(spec.text_source),
        output_base=str(config.paths.project_root / "embeddings"),
    )

    aug_lbl = "augmented" if args.augmentation else "standard"
    print("=" * 64)
    print("  Embedding Extraction")
    print("=" * 64)
    print(f"  dataset     : {spec.name} ({spec.language}, {spec.domain})")
    print(f"  model       : {vlm_cfg.model_name}")
    print(f"  device      : {vlm_cfg.device}  (batch={vlm_cfg.batch_size}, workers={vlm_cfg.num_workers}, half={vlm_cfg.use_half})")
    print(f"  mode        : {aug_lbl}")
    print(f"  image_dir   : {emb_cfg.image_dir}")
    print(f"  text_source : {emb_cfg.reports_dir}")
    print(f"  visual_out  : {emb_cfg.visual_output_path}")
    print(f"  text_out    : {emb_cfg.text_output_path}")
    print("=" * 64)

    model, processor = _load_biomedclip(vlm_cfg)

    # --- Visual embeddings ---
    image_dataset = spec.image_dataset_cls(spec.image_dir)
    print(f"\nImages: {len(image_dataset)}")
    extract_visual_embeddings(model, processor, image_dataset, vlm_cfg, emb_cfg)

    # --- Text embeddings (optional; reports/captions) ---
    if not args.skip_text:
        text_dataset = spec.text_dataset_cls(spec.text_source, spec.image_dir)
        print(f"\nReports: {len(text_dataset)}")
        extract_text_embeddings(model, processor, text_dataset, vlm_cfg, emb_cfg)
    else:
        print("\n--skip-text: skipping report/caption embedding extraction.")

    # Free the VLM (only extraction needs it).
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Verify ---
    visual_emb = torch.load(emb_cfg.visual_output_path, map_location="cpu", weights_only=True)
    print("\n" + "=" * 64)
    print(f"  visual_embeddings.pt : {tuple(visual_emb.shape)}  "
          f"mean norm {visual_emb.norm(dim=-1).mean():.4f} (expected ~1.0)")
    print("=" * 64)


if __name__ == "__main__":
    main()
