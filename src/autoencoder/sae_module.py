"""
sae_module.py — Sparse Autoencoder Facade

Unified interface for training, loading, and using Top-K SAEs.
Delegates to the `dictionary_learning` library for all SAE internals.

Usage:
    from autoencoder.sae_module import SAEManager
    mgr = SAEManager()
    mgr.train(embeddings_path="embeddings/train_embeddings.pt", seed=42)
    mgr.load("models/sae_seed42")
    sparse = mgr.encode(embeddings)               # (B, 512) → (B, 4096)
    concepts = mgr.get_top_concepts(embeddings, n=5)
    decoder_weights = mgr.get_decoder_weights()   # (4096, 512)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE

logger = logging.getLogger(__name__)

# Default config values — kept in sync with config.py's SAEConfig.
# Change config.py, not here. These are only used when SAEManager
# is constructed without a config (e.g. in tests).
_DEFAULTS = {
    "activation_dim": 512,
    "dict_size": 4096,
    "k": 32,
    "lr": None,
    "steps": 50_000,
    "warmup_steps": 1_000,
    "batch_size": 256,
    "log_steps": 1_000,
    "decay_start_frac": 0.8,
    "lm_name": "BiomedCLIP",
    "layer": 0,
    "device": "cpu",
}


def _extract_sae_config(cfg) -> dict:
    """Convert a frozen SAEConfig dataclass to a plain dict."""
    result = {}
    for f in dataclasses.fields(cfg):
        result[f.name] = getattr(cfg, f.name)
    return result


class SAEManager:
    """
    Unified interface for the SAE lifecycle.

    Inference:
        load, encode, decode, reconstruct, get_decoder_weights, get_top_concepts

    Training:
        train, name_concepts, compute_stability

    Metrics:
        compute_reconstruction_mse, compute_sparsity_metrics
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Dict of overrides, OR a frozen SAEConfig dataclass.
                    When passing SAEConfig, all fields are extracted.
                    Supports legacy dict {"device": "cpu"} for tests.
        """
        if config is not None and not isinstance(config, dict):
            config = _extract_sae_config(config)
        self.config = {**_DEFAULTS, **(config or {})}
        self._ae: Optional[AutoEncoderTopK] = None
        self._model_dir: Optional[Path] = None

    @property
    def is_loaded(self) -> bool:
        return self._ae is not None

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        embeddings_path: str | Path,
        seed: int = 42,
        save_dir: str | Path = "models",
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """
        Train an SAE on saved embeddings and persist to disk.

        Args:
            embeddings_path: Path to .pt file with tensor (N, activation_dim).
            seed: Random seed for full reproducibility.
            save_dir: Base directory for saving the model.
            steps: Override default training steps.
            batch_size: Override default batch size.

        Returns:
            Path to the saved model directory (e.g. models/sae_seed42/).
        """
        steps = steps if steps is not None else self.config["steps"]
        batch_size = batch_size if batch_size is not None else self.config["batch_size"]
        device = self.config["device"]
        lr = self.config.get("lr")  # None = auto-scale

        # Full seed propagation for reproducibility
        _set_global_seed(seed)

        # Load and validate embeddings
        embeddings = torch.load(
            embeddings_path, map_location="cpu", weights_only=True
        )
        if embeddings.dim() != 2 or embeddings.shape[1] != self.config["activation_dim"]:
            raise ValueError(
                f"Expected shape (N, {self.config['activation_dim']}), "
                f"got {embeddings.shape}"
            )
        logger.info(f"Loaded {embeddings.shape[0]} embeddings from {embeddings_path}")

        model_dir = Path(save_dir) / f"sae_seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader with explicit generator for deterministic shuffling
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(embeddings),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device != "cpu"),
            generator=generator,
        )

        # trainSAE is step-based — infinite generator cycles over epochs
        def batch_generator():
            while True:
                for (batch,) in loader:
                    yield batch.to(device)

        decay_start = int(steps * self.config["decay_start_frac"])

        trainer_config = {
            "trainer": TopKTrainer,
            "activation_dim": self.config["activation_dim"],
            "dict_size": self.config["dict_size"],
            "k": self.config["k"],
            "steps": steps,
            "layer": self.config["layer"],
            "lm_name": self.config["lm_name"],
            "lr": lr,
            "warmup_steps": self.config["warmup_steps"],
            "decay_start": decay_start,
            "seed": seed,
            "device": device,
        }

        lr_label = "auto" if lr is None else f"{lr:.1e}"
        logger.info(
            f"Training SAE (seed={seed}, steps={steps}, lr={lr_label}, "
            f"decay_start={decay_start})..."
        )
        trainSAE(
            data=batch_generator(),
            trainer_configs=[trainer_config],
            steps=steps,
            save_dir=str(model_dir),
            log_steps=self.config.get("log_steps", 1000),
            device=device,
            autocast_dtype=torch.bfloat16,
            normalize_activations=True,
            verbose=True,
        )

        # Save training manifest for reproducibility
        self._save_manifest(model_dir, seed, steps, batch_size, embeddings_path, embeddings)

        logger.info(f"Model saved to {model_dir}")
        self.load(model_dir)
        return model_dir

    # ── Loading ───────────────────────────────────────────────────────

    def load(self, model_dir: str | Path) -> None:
        """
        Load a pre-trained SAE from disk.

        Handles the library's trainer_0/ subdirectory convention
        and uses weights_only=True for safe deserialization.

        Args:
            model_dir: Directory containing ae.pt (with or without trainer_0/).
        """
        model_dir = Path(model_dir)

        # Check for ae.pt directly, then in trainer_0/ subdirectory
        ae_path = model_dir / "ae.pt"
        if not ae_path.exists():
            trainer_path = model_dir / "trainer_0" / "ae.pt"
            if trainer_path.exists():
                ae_path = trainer_path
            else:
                raise FileNotFoundError(
                    f"Model not found at {model_dir / 'ae.pt'} or {trainer_path}"
                )

        # Safe load with weights_only=True (bypasses unsafe library default)
        state_dict = torch.load(
            ae_path, map_location=self.config["device"], weights_only=True
        )
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        k = self.config["k"]

        if "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(
                f"Config k={k} != saved k={state_dict['k'].item()}"
            )

        self._ae = AutoEncoderTopK(activation_dim, dict_size, k)
        self._ae.load_state_dict(state_dict)
        self._ae = self._ae.float()  # Ensure float32 for consistent inference
        self._ae.eval()
        self._ae.to(self.config["device"])
        self._model_dir = model_dir

        # Validate config matches model
        if activation_dim != self.config["activation_dim"]:
            raise ValueError(
                f"Config activation_dim={self.config['activation_dim']} != "
                f"model activation_dim={activation_dim}"
            )
        if dict_size != self.config["dict_size"]:
            raise ValueError(
                f"Config dict_size={self.config['dict_size']} != "
                f"model dict_size={dict_size}"
            )

        logger.info(
            f"Loaded SAE from {ae_path} "
            f"(activation_dim={activation_dim}, dict_size={dict_size}, k={k})"
        )

    # ── Encoding / Decoding ───────────────────────────────────────────

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings into a sparse representation.

        Args:
            embeddings: Tensor (B, 512).

        Returns:
            Sparse tensor (B, 4096) with k non-zero values per row.
        """
        self._check_loaded()
        with torch.no_grad():
            return self._ae.encode(embeddings.to(self._device))

    def encode_topk(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode returning top-k values and indices.

        Returns:
            (sparse_full, topk_values [B, k], topk_indices [B, k])
        """
        self._check_loaded()
        with torch.no_grad():
            # 4th return value is pre-topk ReLU activations (discarded)
            encoded, values, indices, _ = self._ae.encode(
                embeddings.to(self._device), return_topk=True
            )
        return encoded, values, indices

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode a sparse representation back to embedding space.

        Args:
            sparse: Tensor (B, 4096), output of encode().

        Returns:
            Reconstructed tensor (B, 512).
        """
        self._check_loaded()
        with torch.no_grad():
            return self._ae.decode(sparse.to(self._device))

    def reconstruct(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encode + decode in a single forward pass."""
        self._check_loaded()
        with torch.no_grad():
            return self._ae(embeddings.to(self._device))

    # ── Concept Analysis ──────────────────────────────────────────────

    def get_decoder_weights(self) -> torch.Tensor:
        """
        Return the decoder weight matrix W_dec.
        Each row is a "concept direction" in embedding space.

        NOTE: Rows may not be unit-norm. Use F.normalize() before
        computing cosine similarity (name_concepts does this internally).

        Returns:
            Tensor (dict_size, activation_dim) = (4096, 512).
        """
        self._check_loaded()
        return self._ae.decoder.weight.data.T.clone()

    def get_top_concepts(
        self, embeddings: torch.Tensor, n: int = 5
    ) -> list[list[tuple[int, float]]]:
        """
        For each sample, return the top-n activated concepts.

        Args:
            embeddings: Tensor (B, 512).
            n: Number of top concepts to return per sample.

        Returns:
            List of B lists, each containing n tuples (feature_id, activation).
        """
        self._check_loaded()
        with torch.no_grad():
            sparse = self._ae.encode(embeddings.to(self._device))

        # Vectorized topk (much faster than row-by-row Python loop)
        topk = sparse.topk(n, dim=1)  # values [B, n], indices [B, n]
        results = []
        for i in range(sparse.shape[0]):
            concepts = [
                (idx.item(), val.item())
                for idx, val in zip(topk.indices[i], topk.values[i])
            ]
            results.append(concepts)
        return results

    def name_concepts(
        self,
        vocab_embeddings: torch.Tensor,
        vocab_labels: list[str],
        top_n: int = 3,
    ) -> dict[int, dict]:
        """
        Assign names to SAE concepts via cosine similarity with vocabulary.

        Args:
            vocab_embeddings: Tensor (V, 512), vocabulary embeddings.
            vocab_labels: List of V strings (term names).
            top_n: Number of candidate names per feature.

        Returns:
            Dict {feature_id: {"name": str, "score": float, "candidates": [...]}}.
        """
        self._check_loaded()

        # Validate inputs
        if vocab_embeddings.dim() != 2:
            raise ValueError(
                f"vocab_embeddings must be 2D, got {vocab_embeddings.dim()}D"
            )
        if vocab_embeddings.shape[1] != self.config["activation_dim"]:
            raise ValueError(
                f"vocab_embeddings dim-1 ({vocab_embeddings.shape[1]}) != "
                f"activation_dim ({self.config['activation_dim']})"
            )
        if len(vocab_labels) != vocab_embeddings.shape[0]:
            raise ValueError(
                f"vocab_labels length ({len(vocab_labels)}) != "
                f"vocab_embeddings rows ({vocab_embeddings.shape[0]})"
            )

        W_dec = self.get_decoder_weights()  # (dict_size, 512)

        # Normalize → dot product equals cosine similarity
        W_norm = F.normalize(W_dec, dim=1)
        V_norm = F.normalize(vocab_embeddings.to(self._device), dim=1)

        similarities = W_norm @ V_norm.T  # (dict_size, V)

        concept_names = {}
        for feat_id in range(self.config["dict_size"]):
            topk = similarities[feat_id].topk(top_n)
            candidates = [
                {"label": vocab_labels[idx.item()], "score": val.item()}
                for val, idx in zip(topk.values, topk.indices)
            ]
            concept_names[feat_id] = {
                "name": candidates[0]["label"],
                "score": candidates[0]["score"],
                "candidates": candidates,
            }

        return concept_names

    # ── Metrics ───────────────────────────────────────────────────────

    def compute_reconstruction_mse(self, embeddings: torch.Tensor) -> float:
        """Compute mean MSE between input and reconstruction."""
        self._check_loaded()
        with torch.no_grad():
            x = embeddings.to(self._device)
            x_hat = self._ae(x)
            return F.mse_loss(x_hat, x).item()

    def compute_cosine_reconstruction(self, embeddings: torch.Tensor) -> float:
        """Compute mean cosine similarity between input and reconstruction."""
        self._check_loaded()
        with torch.no_grad():
            x = embeddings.to(self._device)
            x_hat = self._ae(x)
            return F.cosine_similarity(x_hat, x, dim=-1).mean().item()

    def compute_sparsity_metrics(self, embeddings: torch.Tensor) -> dict:
        """
        Compute sparsity and utilization metrics.

        Returns:
            {"l0_mean", "l0_std", "dead_features_pct",
             "activation_entropy", "dict_utilization_pct"}
        """
        self._check_loaded()
        with torch.no_grad():
            sparse = self._ae.encode(embeddings.to(self._device))

        # L0: non-zero count per sample (should equal k for TopK — kept as sanity check)
        l0 = (sparse != 0).float().sum(dim=1)

        # Dead features and dictionary utilization
        active_per_feature = (sparse != 0).float().sum(dim=0)
        n_total = sparse.shape[1]
        n_dead = (active_per_feature == 0).sum().item()
        dead_pct = n_dead / n_total * 100
        utilization_pct = (n_total - n_dead) / n_total * 100

        # Activation entropy: Shannon entropy of feature activation frequencies
        freq = active_per_feature / (active_per_feature.sum() + 1e-8)
        freq = freq[freq > 0]
        entropy = -(freq * freq.log()).sum().item()

        return {
            "l0_mean": l0.mean().item(),
            "l0_std": l0.std().item(),
            "dead_features_pct": dead_pct,
            "activation_entropy": entropy,
            "dict_utilization_pct": utilization_pct,
        }

    # ── Stability ─────────────────────────────────────────────────────

    @staticmethod
    def compute_stability(
        model_dirs: list[str | Path],
        embeddings: torch.Tensor,
        config: Optional[dict] = None,
        n: Optional[int] = None,
    ) -> dict:
        """
        Compute Jaccard similarity across SAEs trained with different seeds.
        Models are loaded one at a time to avoid GPU OOM.

        Args:
            model_dirs: Paths to model directories (one per seed).
            embeddings: Tensor (B, 512) for testing — should be HELD-OUT test set.
            config: Config override.
            n: Number of top features to compare. Defaults to k from config.

        Returns:
            {"jaccard_matrix", "mean_jaccard", "std_jaccard"}
        """
        if config is not None and not isinstance(config, dict):
            config = _extract_sae_config(config)
        effective_config = {**_DEFAULTS, **(config or {})}
        if n is None:
            n = effective_config["k"]

        # Load one model at a time, extract active sets, release GPU memory
        active_sets: list[list[set[int]]] = []
        for d in model_dirs:
            mgr = SAEManager(effective_config)
            mgr.load(d)

            sample_sets: list[set[int]] = []
            chunk_size = 512
            for start in range(0, embeddings.shape[0], chunk_size):
                chunk = embeddings[start : start + chunk_size]
                _, _, indices = mgr.encode_topk(chunk)
                # Use only top-n for comparison
                if n < indices.shape[1]:
                    indices = indices[:, :n]
                for row in indices.cpu():
                    sample_sets.append(set(row.tolist()))
            active_sets.append(sample_sets)

            # Free GPU memory between models
            del mgr._ae
            mgr._ae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_seeds = len(model_dirs)
        n_samples = embeddings.shape[0]
        jaccard_matrix = torch.zeros(n_seeds, n_seeds)

        # Pairwise Jaccard averaged over samples
        for i in range(n_seeds):
            for j in range(i, n_seeds):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                    continue
                jaccards = []
                for s in range(n_samples):
                    a, b = active_sets[i][s], active_sets[j][s]
                    union = len(a | b)
                    if union > 0:
                        jaccards.append(len(a & b) / union)
                    else:
                        jaccards.append(0.0)
                mean_j = sum(jaccards) / len(jaccards)
                jaccard_matrix[i, j] = mean_j
                jaccard_matrix[j, i] = mean_j

        # Summary from upper triangle (unique pairs only)
        mask = torch.triu(torch.ones(n_seeds, n_seeds), diagonal=1).bool()
        upper_vals = jaccard_matrix[mask]

        return {
            "jaccard_matrix": jaccard_matrix,
            "mean_jaccard": upper_vals.mean().item(),
            "std_jaccard": upper_vals.std().item(),
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _check_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("SAE not loaded. Call .load(model_dir) or .train() first.")

    @property
    def _device(self) -> str:
        return self.config["device"]

    def _save_manifest(
        self,
        model_dir: Path,
        seed: int,
        steps: int,
        batch_size: int,
        embeddings_path: str | Path,
        embeddings: torch.Tensor,
    ) -> None:
        """Save training manifest for exact reproduction."""
        lr_used = self.config.get("lr")
        if lr_used is None:
            scale = self.config["dict_size"] / (2**14)
            lr_used = 2e-4 / scale**0.5

        manifest = {
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "lr_auto_scaled": lr_used,
            "activation_dim": self.config["activation_dim"],
            "dict_size": self.config["dict_size"],
            "k": self.config["k"],
            "warmup_steps": self.config["warmup_steps"],
            "decay_start_frac": self.config["decay_start_frac"],
            "log_steps": self.config.get("log_steps", 1000),
            "autocast_dtype": "bfloat16",
            "normalize_activations": True,
            "device": self.config["device"],
            "embeddings_path": str(embeddings_path),
            "embeddings_shape": list(embeddings.shape),
            "embeddings_hash": hashlib.sha256(
                embeddings[: min(100, len(embeddings))].cpu().numpy().tobytes()
            ).hexdigest()[:16],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)

        with open(model_dir / "training_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def _set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
