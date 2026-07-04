"""
run_vocab_building_pipeline.py — Build the medical vocabulary (RadLex or MeSH).

Dispatches on the active dataset's ``vocab_source`` (``DatasetSpec``):
  - iu_xray / padchest  (``radlex``): RadLex chest terms ranked by a CXR-anchor
    centroid + NIH ChestX-ray14 seeds (the existing chest vocabulary).
  - rocov2              (``mesh``):   external MeSH descriptors (XML,
    ``data/mesh/desc<year>.gz``) filtered to the radiology branches (A/C/E) and
    ranked by radiology anchors — independent of ROCOv2's labels (no circularity;
    see ``docs/FINDINGS.md`` B4). Downloaded by ``xai_datasets/download_mesh.py``.

Output (per-dataset): ``vocabulary.json`` + ``text_vocab_embeddings.pt`` under
``embeddings/<dataset>/``.

Usage:
    python scripts/run_vocab_building_pipeline.py                       # active dataset
    python scripts/run_vocab_building_pipeline.py --dataset iu_xray --topk 300
    python scripts/run_vocab_building_pipeline.py --dataset rocov2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root -> xai_datasets

import config
from config import VLMConfig, VocabularyConfig
from utils import load_vlm
from vocabulary_building.build_vocabulary import build_vocabulary_pipeline
from vocabulary_building.mesh_vocab import build_mesh_vocabulary
from xai_datasets.spec import get_dataset


def parse_args(vlm_config: VLMConfig, vocab_config: VocabularyConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the medical vocabulary (RadLex chest or UMLS multimodal)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.active_dataset.name,
        help=f"Active dataset (default: {config.active_dataset.name}); selects the vocab source.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=vocab_config.input_csv_path,
        help="RadLex input CSV (radlex datasets only; default from VocabularyConfig).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=vocab_config.top_k,
        help=f"Top-k terms to keep (default: {vocab_config.top_k}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=vlm_config.device,
        help=f"Device for inference (default: {vlm_config.device}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=vlm_config.batch_size,
        help=f"Batch size for text encoding (default: {vlm_config.batch_size}).",
    )
    return parser.parse_args()


def main():
    vlm_config = VLMConfig()
    vocab_config = VocabularyConfig()
    args = parse_args(vlm_config, vocab_config)

    # Route paths to the selected dataset BEFORE reading any (vocabulary.json is
    # per-dataset: RadLex-chest for IU/PadChest, external MeSH for ROCOv2).
    config.select_dataset(args.dataset)
    spec = get_dataset(args.dataset)

    vlm_config.device = args.device
    vlm_config.batch_size = args.batch_size
    vocab_config.top_k = args.topk

    print("=" * 60)
    print("  Build Medical Vocabulary")
    print("=" * 60)
    print(f"  dataset      : {spec.name} ({spec.language}, {spec.domain})")
    print(f"  vocab_source : {spec.vocab_source}")
    print(f"  output JSON  : {vocab_config.output_file}")
    print(f"  output emb   : {vocab_config.embeddings_file}")
    print(f"  device       : {vlm_config.device}  (batch={vlm_config.batch_size})")
    print("=" * 60)

    model, processor = load_vlm(vlm_config)

    if spec.vocab_source == "mesh":
        if not spec.mesh_file:
            raise SystemExit(
                f"Dataset {spec.name!r} has vocab_source='mesh' but no mesh_file on its spec."
            )
        vocabulary = build_mesh_vocabulary(
            spec.mesh_file,
            model,
            processor,
            vlm_config,
            vocab_config,
            top_k=args.topk,
        )
    else:
        vocab_config.input_csv_path = args.csv  # RadLex input override
        vocabulary = build_vocabulary_pipeline(model, processor, vlm_config, vocab_config)

    print(f"\nDone. {len(vocabulary)} terms in final vocabulary.")


if __name__ == "__main__":
    main()
