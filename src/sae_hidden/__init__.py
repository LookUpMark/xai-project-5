"""Path A — SAE on BiomedCLIP's 768-d pre-projection CLS hidden state.

Additive pipeline (the 512-d baseline is untouched). Stages, each runnable and
each writing a markdown REPORT under results/sae_hidden/:

    extract_hidden.py            embeddings/standard_hidden/*.pt
    train_hidden.py              models/sae_hidden/sae_seed{N}/
    naming_hidden.py             results/sae_hidden/concept_names.json
    stability_hidden.py          results/sae_hidden/stability_analysis.json
    generate_explanations_hidden.py  results/sae_hidden/sample_explanations.json
"""
