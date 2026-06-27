# Literature

Source PDFs for the SAE-stability / concept-discovery work. The PDFs themselves
are **gitignored** (`.gitignore:232`, `docs/literature/*.pdf` — copyrighted) and
live only on the local working tree; this index is the committed trace.

Regenerated 2026-06-26 (the originals were lost — never committed on any branch).

| Paper | Citation | Source / id | Local file | Role in this project |
|-------|----------|-------------|------------|----------------------|
| Lan et al. 2024 | *Studying the stability of sparse autoencoder features* | arXiv [2410.06981](https://arxiv.org/abs/2410.06981) | `Lan2024_stability_2410.06981.pdf` | Cross-seed SAE stability methodology; validates decoder-cosine feature matching + the matched-stability metric (`SAEManager.compute_stability_matched`). |
| Leask et al. 2025 | *Universality of SAE features* | arXiv [2502.04878](https://arxiv.org/abs/2502.04878) | `Leask2025_2502.04878.pdf` | Weak-vs-strong universality framing — the "weakly universal, not strongly reproducible" verdict in `REPORT_stability_matched.md`. |
| Gao et al. 2024 | *Scaling and evaluating sparse autoencoders* (Top-K / JumpReLU SAE) | arXiv [2406.04093](https://arxiv.org/abs/2406.04093) | `Gao2024_topk_2406.04093.pdf` | The Top-K SAE architecture used throughout (`SAEConfig`, `SAEHiddenConfig`); dead-feature / L0 analysis. |
| Bricken et al. 2023 | *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* | Transformer Circuits Thread ([HTML](https://transformer-circuits.pub/2023/monosemantic-features)) — no official PDF; local copy is a third-party mirror of the freely-published paper | `Bricken2023_monosemanticity.pdf` | Foundational SAE-for-interpretability motivation (dictionary learning → monosemantic features). |

## Notes

- **Bricken 2023** is not on arXiv; Anthropic publishes it only as an interactive
  HTML post, so the PDF here is a mirrored copy. Canonical source:
  https://transformer-circuits.pub/2023/monosemantic-features
- Methodological rationale for each paper is in `docs/design/LITERATURE-SAE-STABILITY.md`.
- To regenerate the arXiv copies:
  ```bash
  for id in 2410.06981 2502.04878 2406.04093; do
    curl -sL -o "${id}.pdf" "https://arxiv.org/pdf/${id}.pdf"
  done
  ```
