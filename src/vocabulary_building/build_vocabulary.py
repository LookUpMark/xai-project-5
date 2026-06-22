import json
import torch
from tqdm import tqdm
from typing import List, Tuple

from config import VLMConfig, VocabularyConfig


# Encoding records (of the deduplicated & filtered CSV) using BiomedCLIP text encoder
def encode_texts(
    texts: List[str],
    model,
    processor,
    vlm_config: VLMConfig,
) -> torch.Tensor:
    """
    Encode a list of text strings into L2-normalized embeddings using 
    BiomedCLIP's text encoder.

    Args:
        texts: list of strings to encode.
        model: loaded BiomedCLIP model.
        processor: loaded BiomedCLIP processor.
        vlm_config (VLMConfig): model runtime parameters.

    Returns:
        torch.Tensor: (len(texts), 512) L2-normalized embeddings on CPU.
    """
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), vlm_config.batch_size)):
            batch = texts[i : i + vlm_config.batch_size]

            inputs = processor.tokenizer(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(vlm_config.device)

            outputs = model.get_text_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

            all_embeddings.append(outputs.cpu())

    return torch.cat(all_embeddings)  # (len(texts), 512)


# Compute one centroid per clinically defined anchor group (hard-clustering)
def compute_anchor_centroids(
    model,
    processor,
    vlm_config: VLMConfig,
    vocab_config: VocabularyConfig,
) -> torch.Tensor:
    """
    Encode anchor queries grouped by clinical sub-domain and return
    one L2-normalized centroid per group.

    Groups are defined in VocabularyConfig.anchor_groups (hard-clustered
    by clinical domain knowledge).  Within each group, the centroid is
    the mean of semantically coherent anchors (e.g. all cardiac terms).

    Args:
        model: loaded BiomedCLIP model.
        processor: loaded BiomedCLIP processor.
        vlm_config (VLMConfig): model runtime parameters.
        vocab_config (VocabularyConfig): vocabulary parameters.

    Returns:
        torch.Tensor: (K, 512) cluster centroids, K = number of groups.
    """
    centroids = []
    total_anchors = 0

    print(f"\nComputing {len(vocab_config.anchor_groups)} clinically defined "
          f"centroids:\n")

    for group_name, queries in vocab_config.anchor_groups.items():
        group_embeddings = encode_texts(
            texts=queries,
            model=model,
            processor=processor,
            vlm_config=vlm_config,
        )  # (len(queries), 512)

        centroid = group_embeddings.mean(dim=0)
        centroid = centroid / centroid.norm()  # L2 normalize
        centroids.append(centroid)
        total_anchors += len(queries)

        print(f"  {group_name} ({len(queries)} anchors):")
        for q in queries:
            print(f"    • {q}")

    print(f"\n  Total: {total_anchors} anchors → "
          f"{len(centroids)} centroids\n")

    return torch.stack(centroids)  # (K, 512)


# Ranking terms by max cosine similarity across macro-centroids
def rank_terms_by_relevance(
    terms: List[str],
    term_embeddings: torch.Tensor,
    anchor_centroids: torch.Tensor,
) -> List[Tuple[str, float]]:
    """
    Rank terms by their maximum cosine similarity to any of the clinically 
    defined macro-centroids.

    Each term is compared to the K pre-defined centroids. The score is the maximum
    similarity across these macro-centroids: a term is deemed relevant if it
    belongs to at least one of the clinical sub-domains.

    Args:
        terms: list of term strings (same order as embeddings rows).
        term_embeddings: (N, 512) tensor of term embeddings.
        anchor_centroids: (K, 512) tensor of clustered macro-centroids.

    Returns:
        List[Tuple[str, float]]: (term, max_similarity_score) sorted desc.
    """
    # (N, 512) @ (512, K) → (N, K)
    similarity_matrix = term_embeddings @ anchor_centroids.T

    # Max similarity across the K macro-centroids for each term → (N,)
    max_similarities = similarity_matrix.max(dim=1).values

    # Sort descending
    sorted_indices = max_similarities.argsort(descending=True)
    ranked = [
        (terms[idx.item()], max_similarities[idx].item())
        for idx in sorted_indices
    ]
    return ranked


# Build final vocabulary (top-k + NIH seeds)
def build_final_vocabulary(
    ranked_terms: List[Tuple[str, float]],
    config: VocabularyConfig,
    input_terms_set: set,
) -> List[dict]:
    """
    Select the top-k most relevant input terms and ensure the 14 NIH 
    ChestX-ray14 seed terms are always included.

    Args:
        ranked_terms: list of (term, score) sorted by descending relevance.
        config (VocabularyConfig): dataclass containing parameters.
        input_terms_set: set of lowercase terms that strictly came from the input.

    Returns:
        List[dict]: final vocabulary entries with keys 
                    {"term", "similarity_score", "source"}.
    """
    selected = {}
    
    # Collect top-k input terms
    input_count = 0
    for term, score in ranked_terms:
        if input_count >= config.top_k:
            break
        if term.lower() in input_terms_set:
            selected[term.lower()] = {
                "term": term,
                "similarity_score": round(score, 6),
                "source": "input_filtered",
            }
            input_count += 1

    # Ensure NIH seed terms are included
    nih_lookup = {t.lower(): t for t in config.nih_seed_terms}
    ranked_lookup = {t.lower(): s for t, s in ranked_terms}

    added_seeds = 0
    for seed_lower, seed_original in nih_lookup.items():
        if seed_lower not in selected:
            # We are guaranteed to have a score because all missing seeds 
            # were injected before encoding and ranking.
            score = ranked_lookup.get(seed_lower, 0.0)
            selected[seed_lower] = {
                "term": seed_original,
                "similarity_score": round(score, 6),
                "source": "nih_chestxray14_seed",
            }
            added_seeds += 1

    vocabulary = sorted(
        selected.values(),
        key=lambda x: x["similarity_score"],
        reverse=True,
    )

    n_from_input = sum(1 for v in vocabulary if v["source"] == "input_filtered")
    n_from_nih = sum(1 for v in vocabulary if v["source"] == "nih_chestxray14_seed")
    print(f"\nFinal vocabulary: {len(vocabulary)} terms "
          f"({n_from_input} from input top-{config.top_k}, "
          f"{n_from_nih} NIH seeds added).")
    return vocabulary


# Save vocabulary and embeddings
def save_vocabulary(vocabulary: List[dict], config: VocabularyConfig):
    """
    Save the vocabulary list as a JSON file.

    Args:
        vocabulary: list of vocabulary entry dicts.
        config (VocabularyConfig): dataclass containing parameters.
    """
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)

    print(f"Vocabulary saved to {config.output_file}")


def save_vocab_embeddings(
    vocabulary: List[dict],
    all_terms: List[str],
    all_embeddings: torch.Tensor,
    config: VocabularyConfig,
):
    """
    Extract and save the embeddings corresponding to the final 
    vocabulary terms (for downstream concept naming).

    Args:
        vocabulary: list of vocabulary entry dicts.
        all_terms: full list of CSV terms (same order as all_embeddings).
        all_embeddings: (N, 512) embeddings for all CSV terms.
        config (VocabularyConfig): dataclass containing parameters.
    """
    term_to_idx = {t.lower(): i for i, t in enumerate(all_terms)}

    vocab_indices = []
    for entry in vocabulary:
        idx = term_to_idx.get(entry["term"].lower())
        if idx is not None:
            vocab_indices.append(idx)

    vocab_embeddings = all_embeddings[vocab_indices]  # (len(vocab_indices), 512)

    config.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vocab_embeddings, config.embeddings_file)

    print(f"Vocabulary embeddings saved to {config.embeddings_file} "
          f"— shape {tuple(vocab_embeddings.shape)}")


# ──────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────

def build_vocabulary_pipeline(
    model, 
    processor,
    vlm_config: VLMConfig,
    vocab_config: VocabularyConfig,
    all_terms: List[str],
):
    """
    Full vocabulary building pipeline:
        1. Inject missing NIH seed terms into the input term list
        2. Encode all terms with BiomedCLIP text encoder
        3. Compute clinically defined macro-centroids (hard-clustering)
        4. Rank terms by max cosine similarity to any macro-centroid
        5. Select top-k input terms + NIH seed terms
        6. Save vocabulary JSON and embeddings

    Args:
        model: loaded BiomedCLIP model.
        processor: loaded BiomedCLIP processor.
        vlm_config (VLMConfig): model runtime parameters.
        vocab_config (VocabularyConfig): vocabulary pipeline parameters.
        all_terms (List[str]): list of raw terms to filter from.
    """
    input_terms_set = {t.lower() for t in all_terms}
    
    # Inject missing NIH seeds into CSV terms
    missing_seeds = [t for t in vocab_config.nih_seed_terms if t.lower() not in input_terms_set]
    if missing_seeds:
        all_terms.extend(missing_seeds)
    
    # Encode all terms (Input + injected seeds)
    print(f"\nEncoding {len(all_terms)} terms...")
    all_embeddings = encode_texts(
        texts=all_terms,
        model=model,
        processor=processor,
        vlm_config=vlm_config,
    )

    # Compute clinically defined macro-centroids
    anchor_centroids = compute_anchor_centroids(
        model, processor, vlm_config, vocab_config,
    )

    # Rank by max-similarity to any centroid (multi-centroid filtering)
    ranked_terms = rank_terms_by_relevance(all_terms, all_embeddings, anchor_centroids)

    # Preview top-10
    print("\nTop-10 most CXR-relevant terms:")
    for i, (term, score) in enumerate(ranked_terms[:10]):
        print(f"  {i+1:2d}. {term:<40s} (sim={score:.4f})")

    # Build final vocabulary (NIH seeds are always included)
    vocabulary = build_final_vocabulary(ranked_terms, vocab_config, input_terms_set)

    # Save outputs
    save_vocabulary(vocabulary, vocab_config)
    save_vocab_embeddings(vocabulary, all_terms, all_embeddings, vocab_config)

    return vocabulary
