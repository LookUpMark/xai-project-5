"""
00_build_vocab.py — Build CXR-relevant medical vocabulary from RadLex.

Pipeline:
    1. Load RadLex CSV (~47K terms) and filter obsolete entries
    2. Encode all terms with BiomedCLIP text encoder
    3. Compute a "relevance centroid" from ~20 CXR-specific anchor queries
    4. Rank RadLex terms by cosine similarity to the centroid
    5. Keep the top-k most relevant terms + 14 NIH ChestX-ray14 seed terms
    6. Save vocabulary JSON and pre-computed embeddings

Usage:
    python scripts/00_build_vocab.py
    python scripts/00_build_vocab.py --csv data/radlex.csv --topk 300 --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import VLMConfig, VocabularyConfig
from utils import load_vlm
from vocabulary_building.build_vocabulary import build_vocabulary_pipeline


def parse_args(vlm_config: VLMConfig, vocab_config: VocabularyConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CXR-relevant medical vocabulary from RadLex."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=vocab_config.input_csv_path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=vocab_config.top_k,
        help=f"Number of top input terms to keep (default: {vocab_config.top_k}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=vocab_config.output_path,
        help="Output path for the vocabulary JSON.",
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
    # Instantiate default configs exactly once
    vlm_config = VLMConfig()
    vocab_config = VocabularyConfig()

    # Parse args using the existing configs to read default values
    args = parse_args(vlm_config, vocab_config)

    # Override config values with parsed CLI arguments
    vlm_config.device = args.device
    vlm_config.batch_size = args.batch_size

    vocab_config.input_csv_path = args.csv
    vocab_config.top_k = args.topk
    vocab_config.output_path = args.output

    print("=" * 60)
    print("  Build Medical Vocabulary")
    print("=" * 60)
    print(f"  Input CSV      : {vocab_config.input_csv}")
    print(f"  Top-K          : {vocab_config.top_k}")
    print(f"  NIH seeds      : {len(vocab_config.nih_seed_terms)}")
    print(f"  Anchor queries : {len(vocab_config.anchor_queries)}")
    print(f"  Output JSON    : {vocab_config.output_file}")
    print(f"  Output embeds  : {vocab_config.embeddings_file}")
    print(f"  Device         : {vlm_config.device}")
    print(f"  Batch size     : {vlm_config.batch_size}")
    print("=" * 60)

    # Load BiomedCLIP
    model, processor = load_vlm(vlm_config)

    # Run the pipeline (RadLex terms are loaded + filtered inside the pipeline).
    vocabulary = build_vocabulary_pipeline(
        model, processor, vlm_config, vocab_config
    )

    print(f"\nDone. {len(vocabulary)} terms in final vocabulary.")


if __name__ == "__main__":
    main()
