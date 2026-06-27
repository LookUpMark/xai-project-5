"""naming_hidden.py — Path A concept naming via the frozen projection bridge.

The 768-d SAE decoder rows live in pre-projection space; RadLex text embeddings
live in the 512-d shared space. We bridge them with BiomedCLIP's FROZEN
``visual_projection`` (Linear 768->512, bias=False — option (a) of audit M-001):
project each decoder direction to 512-d, gap-correct, then cosine-match against
RadLex. This is the naming bridge that ``SAEManager.name_concepts`` cannot do
(it requires vocab dim == activation_dim).

Math (verified against modeling_biomed_clip.py):
    W_proj = model.visual_projection.weight          # (512, 768)
    dec_512 = W_dec_768 @ W_proj.T                   # (dict, 768) @ (768, 512) = (dict, 512)

Run:
    python src/sae_hidden/naming_hidden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import utils
from autoencoder.sae_module import SAEManager
from sae_hidden.reports import md_table, write_report
from sae_hidden.train_hidden import hidden_sae_config

log = utils.setup_logging(__name__)

SEED = config.training.primary_seed
# Baseline reference numbers (ML-AUDIT-2026-06-25, M-005) for the comparison table.
RANDOM_BASELINE = 0.372
BASELINE_SAE_NAMING = 0.395


def dead_feature_mask(W_dec: torch.Tensor, dead_threshold: float = 1e-8) -> torch.Tensor:
    """Boolean mask of dead features, judged on the raw 768-d decoder norms.

    Done pre-projection on purpose: projecting a zero row gives ``-gap`` with
    non-zero norm, which would mask a genuinely dead feature.
    """
    return W_dec.norm(dim=1) < dead_threshold


def project_decoder_to_text(
    W_dec: torch.Tensor, W_proj: torch.Tensor, gap: torch.Tensor | None = None
) -> torch.Tensor:
    """Bridge 768-d decoder rows into the 512-d shared space via the frozen projection.

    Args:
        W_dec: (dict, 768) SAE decoder weight rows.
        W_proj: (512, 768) frozen ``visual_projection.weight`` (bias=False).
        gap: optional (512,) modality gap to subtract from every projected row.

    Returns:
        (dict, 512) projected decoder directions.
    """
    dec_512 = W_dec @ W_proj.T  # (dict, 768) @ (768, 512) = (dict, 512)
    if gap is not None:
        dec_512 = dec_512 - gap.unsqueeze(0)
    return dec_512


def bridge_cosine_sims(
    dec_512: torch.Tensor, vocab_emb: torch.Tensor, dead_mask: torch.Tensor
) -> torch.Tensor:
    """Cosine similarities (dict, V) between projected decoder rows and vocab terms.

    Dead rows get an all-zero similarity row (so they never win an argmax).
    """
    dec_norm = F.normalize(dec_512, dim=1)
    dec_norm[dead_mask] = 0.0
    vocab_norm = F.normalize(vocab_emb, dim=1)
    return dec_norm @ vocab_norm.T


def _vocab_term(label) -> str:
    """vocabulary.json stores {"term", ...} dicts (or bare strings); coerce to str."""
    if isinstance(label, dict):
        return label.get("term") or str(label)
    return label


def _load_visual_projection() -> torch.Tensor:
    """Load BiomedCLIP's frozen visual_projection weight (512, 768)."""
    model, _ = utils.load_vlm(config.vlm)
    w_proj = model.visual_projection.weight.detach().cpu()
    del model
    if tuple(w_proj.shape) != (config.backbone.embedding_dim, config.sae_hidden.activation_dim):
        raise RuntimeError(
            f"visual_projection.weight shape {tuple(w_proj.shape)} != "
            f"expected (512, 768). Has the model changed?"
        )
    return w_proj


def run() -> Path:
    model_dir = config.paths.hidden_models_dir / f"sae_seed{SEED}"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Primary-seed model missing: {model_dir}. Run train_hidden.py first."
        )

    mgr = SAEManager(hidden_sae_config())
    mgr.load(model_dir)
    W_dec = mgr.get_decoder_weights().cpu()  # (dict, 768) — bridge runs on CPU

    # Dead features are a property of the learned 768-d decoder vector, so flag
    # them on the pre-projection norms (projecting a zero row gives -gap, masking it).
    dead_mask = dead_feature_mask(W_dec, config.sae_hidden.dead_threshold)
    n_dead = int(dead_mask.sum().item())

    # ── Frozen-projection bridge: 768-d decoder -> 512-d shared space ──
    W_proj = _load_visual_projection()  # (512, 768)

    # Load RadLex text embeddings + labels. The vocab is a text-space artifact
    # independent of image augmentation, so read the canonical standard/ copy
    # (config.paths.vocab_embeddings_path tracks the augmentation flag and may
    # point at a nonexistent embeddings/augmented/).
    vocab_path = config.paths.project_root / "embeddings" / "standard" / "text_vocab_embeddings.pt"
    vocab_emb = utils.load_tensor(vocab_path)  # (V, 512)
    with open(config.paths.vocab_labels_path) as f:
        vocab_raw = json.load(f)
    vocab_terms = [_vocab_term(v) for v in vocab_raw]
    if len(vocab_terms) != vocab_emb.shape[0]:
        raise ValueError(
            f"vocab labels ({len(vocab_terms)}) != vocab embeddings rows ({vocab_emb.shape[0]})"
        )

    # Modality-gap correction (512-d gap, consistent with the 512-d bridge space)
    gap_path = config.paths.models_dir / "modality_gap.pt"
    gap = utils.load_tensor(gap_path, device="cpu") if gap_path.exists() else None

    dec_512 = project_decoder_to_text(W_dec, W_proj, gap)
    sims = bridge_cosine_sims(dec_512, vocab_emb, dead_mask)  # (dict, V)

    top_n = config.explanation.concept_top_n
    concept_names = {}
    for feat_id in range(W_dec.shape[0]):
        if dead_mask[feat_id]:
            concept_names[feat_id] = {
                "name": "DEAD_FEATURE", "score": 0.0, "candidates": [], "is_dead": True,
            }
            continue
        topk = sims[feat_id].topk(top_n)
        candidates = [
            {"label": vocab_terms[idx.item()], "score": round(val.item(), 4)}
            for val, idx in zip(topk.values, topk.indices)
        ]
        concept_names[feat_id] = {
            "name": candidates[0]["label"],
            "score": candidates[0]["score"],
            "candidates": candidates,
            "is_dead": False,
        }

    # Persist
    out_path = config.paths.hidden_results_dir / "concept_names.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(concept_names, f, indent=2, ensure_ascii=False)

    # ── Report ──
    live = [v for v in concept_names.values() if not v["is_dead"]]
    naming_mean = float(torch.tensor([v["score"] for v in live]).mean().item()) if live else 0.0
    naming_med = float(torch.tensor([v["score"] for v in live]).median().item()) if live else 0.0
    total = len(concept_names)
    dead_pct = n_dead / total * 100

    # Top-scoring live features (the most confidently named concepts)
    live_with_id = [
        (fid, concept_names[fid]) for fid in concept_names if not concept_names[fid]["is_dead"]
    ]
    top_live = sorted(live_with_id, key=lambda kv: kv[1]["score"], reverse=True)[:10]
    summary = (
        f"Named {total - n_dead}/{total} live 768-d features via the frozen projection "
        f"bridge. Mean naming cosine = {naming_mean:.4f} "
        f"(random {RANDOM_BASELINE}, baseline 512-d SAE {BASELINE_SAE_NAMING}). "
        f"Dead features: {n_dead} ({dead_pct:.1f}%)."
    )
    sections = [
        (
            "Naming score vs references",
            md_table(
                ["metric", "value", "reference"],
                [
                    ["mean (live)", f"{naming_mean:.4f}", f"> random {RANDOM_BASELINE}"],
                    ["median (live)", f"{naming_med:.4f}", ""],
                    ["baseline 512-d SAE", f"{BASELINE_SAE_NAMING:.4f}", "ML-AUDIT M-005"],
                    ["random baseline", f"{RANDOM_BASELINE:.4f}", "ML-AUDIT M-005"],
                    ["dead features", f"{dead_pct:.1f}%", "baseline 40-60%"],
                ],
            ),
        ),
        (
            "Bridge check",
            md_table(
                ["item", "value"],
                [
                    ["W_dec shape", str(tuple(W_dec.shape))],
                    ["W_proj shape", str(tuple(W_proj.shape)) + " (bias=False)"],
                    ["dec_512 shape", str(tuple(dec_512.shape))],
                    ["gap applied", str(gap is not None)],
                    ["vocab size", str(vocab_emb.shape[0])],
                ],
            ),
        ),
        (
            "Top-10 most-confidently named live features",
            md_table(
                ["feat_id", "name", "score"],
                [[fid, c["name"], f"{c['score']:.4f}"] for fid, c in top_live],
            ),
        ),
    ]
    report_path = config.paths.hidden_results_dir / "REPORT_naming.md"
    write_report(report_path, "Path A — Concept Naming (frozen-projection bridge)", sections, summary)
    log.info(f"Concept names: {out_path}")
    log.info(f"Report: {report_path}")
    return out_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
