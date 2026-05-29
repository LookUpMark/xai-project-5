"""
SAE Module — Sparse Autoencoder Facade

Unified interface for training, loading, and using Top-K SAEs.

Usage:
    from sae_module import SAEManager
    mgr = SAEManager()
    mgr.train(embeddings_path="embeddings/visual_embeddings.pt", seed=42)
    mgr.load("models/sae_seed42")
    sparse = mgr.encode(embeddings)
    concepts = mgr.get_top_concepts(embeddings, n=5)
    decoder_weights = mgr.get_decoder_weights()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "activation_dim": 512,
    "dict_size": 4096,
    "k": 32,
    "lr": 5e-5,
    "steps": 50_000,
    "warmup_steps": 1000,
    "batch_size": 256,
    "lm_name": "BiomedCLIP",
    "layer": 0,
    "device": "cuda",
}


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
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._ae: Optional[AutoEncoderTopK] = None
        self._model_dir: Optional[Path] = None

    @property
    def is_loaded(self) -> bool:
        return self._ae is not None

    def train(
        self,
        embeddings_path: str | Path,
        seed: int = 42,
        save_dir: str | Path = "models",
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """
        Train an SAE on saved embeddings and persist the model to disk.

        Args:
            embeddings_path: Path to .pt file with tensor (N, 512).
            seed: Random seed for reproducibility.
            save_dir: Base directory for saving the model.
            steps: Override default training steps.
            batch_size: Override default batch size.

        Returns:
            Path to the saved model directory (e.g. models/sae_seed42/).
        """
        steps = steps or self.config["steps"]
        batch_size = batch_size or self.config["batch_size"]
        device = self.config["device"]

        embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        assert embeddings.dim() == 2 and embeddings.shape[1] == self.config["activation_dim"], (
            f"Expected shape (N, {self.config['activation_dim']}), got {embeddings.shape}"
        )
        logger.info(f"Loaded {embeddings.shape[0]} embeddings from {embeddings_path}")

        model_dir = Path(save_dir) / f"sae_seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)

        def batch_generator():
            n = embeddings.shape[0]
            while True:
                perm = torch.randperm(n)
                for i in range(0, n, batch_size):
                    batch = embeddings[perm[i : i + batch_size]].to(device)
                    yield batch

        trainer_config = {
            "trainer": TopKTrainer,
            "activation_dim": self.config["activation_dim"],
            "dict_size": self.config["dict_size"],
            "k": self.config["k"],
            "steps": steps,
            "layer": self.config["layer"],
            "lm_name": self.config["lm_name"],
            "lr": self.config["lr"],
            "warmup_steps": self.config["warmup_steps"],
            "seed": seed,
            "device": device,
        }

        logger.info(f"Training SAE (seed={seed}, steps={steps})...")
        trainSAE(
            data=batch_generator(),
            trainer_configs=[trainer_config],
            steps=steps,
            save_dir=str(model_dir),
            device=device,
            autocast_dtype=torch.bfloat16,
            verbose=True,
        )

        logger.info(f"Model saved to {model_dir}")
        self.load(model_dir)
        return model_dir

    def load(self, model_dir: str | Path) -> None:
        """
        Load a pre-trained SAE from disk.

        Args:
            model_dir: Directory containing ae.pt (and optionally config.json).
        """
        model_dir = Path(model_dir)
        ae_path = model_dir / "ae.pt"

        if not ae_path.exists():
            raise FileNotFoundError(f"Model not found: {ae_path}")

        self._ae = AutoEncoderTopK.from_pretrained(
            str(ae_path),
            k=self.config["k"],
            device=self.config["device"],
        )
        self._ae.eval()
        self._model_dir = model_dir
        logger.info(f"Loaded SAE from {model_dir}")

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings into a sparse representation.

        Args:
            embeddings: Tensor (B, 512).

        Returns:
            Sparse tensor (B, 4096) with only k=32 non-zero values per row.
        """
        self._check_loaded()
        with torch.no_grad():
            return self._ae.encode(embeddings.to(self._device))

    def encode_topk(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode returning top-k values and indices as well.

        Returns:
            (sparse_full, topk_values [B,k], topk_indices [B,k])
        """
        self._check_loaded()
        with torch.no_grad():
            encoded, values, indices, _ = self._ae.encode(
                embeddings.to(self._device), return_topk=True
            )
        return encoded, values, indices

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode a sparse representation back to the embedding space.

        Args:
            sparse: Tensor (B, 4096) — output of encode().

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

    def get_decoder_weights(self) -> torch.Tensor:
        """
        Return the decoder weight matrix W_dec.
        Each row is a "concept direction" in embedding space.

        Returns:
            Tensor (dict_size, activation_dim) = (4096, 512).
        """
        self._check_loaded()
        # decoder.weight is (512, 4096), transpose to get (4096, 512)
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
            List of B lists, each containing n tuples (feature_id, activation_value)
            sorted by activation in descending order.
        """
        self._check_loaded()
        with torch.no_grad():
            sparse = self._ae.encode(embeddings.to(self._device))

        results = []
        for row in sparse:
            topk = row.topk(n)
            concepts = [
                (idx.item(), val.item())
                for idx, val in zip(topk.indices, topk.values)
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
        Assign names to SAE concepts via cosine similarity with the vocabulary.

        Args:
            vocab_embeddings: Tensor (V, 512) — medical vocabulary embeddings.
            vocab_labels: List of V strings (term names).
            top_n: Number of candidate names to return per feature.

        Returns:
            Dict {feature_id: {"name": str, "score": float, "candidates": [...]}}.
        """
        self._check_loaded()
        W_dec = self.get_decoder_weights()  # (4096, 512)

        W_norm = F.normalize(W_dec, dim=1)
        V_norm = F.normalize(vocab_embeddings.to(self._device), dim=1)

        # Cosine similarity: (4096, V)
        similarities = W_norm @ V_norm.T

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

    def compute_reconstruction_mse(self, embeddings: torch.Tensor) -> float:
        """Compute mean MSE between input and reconstruction."""
        self._check_loaded()
        with torch.no_grad():
            x = embeddings.to(self._device)
            x_hat = self._ae(x)
            return F.mse_loss(x_hat, x).item()

    def compute_sparsity_metrics(self, embeddings: torch.Tensor) -> dict:
        """
        Compute sparsity metrics on the activations.

        Returns:
            {"l0_mean": float, "l0_std": float, "hoyer_mean": float,
             "dead_features_pct": float}
        """
        self._check_loaded()
        with torch.no_grad():
            sparse = self._ae.encode(embeddings.to(self._device))

        l0 = (sparse != 0).float().sum(dim=1)

        n = sparse.shape[1]
        l1 = sparse.abs().sum(dim=1)
        l2 = sparse.norm(dim=1)
        hoyer = (n**0.5 - l1 / (l2 + 1e-8)) / (n**0.5 - 1)

        active_per_feature = (sparse != 0).float().sum(dim=0)
        dead_pct = (active_per_feature == 0).float().mean().item() * 100

        return {
            "l0_mean": l0.mean().item(),
            "l0_std": l0.std().item(),
            "hoyer_mean": hoyer.mean().item(),
            "dead_features_pct": dead_pct,
        }

    @staticmethod
    def compute_stability(
        model_dirs: list[str | Path],
        embeddings: torch.Tensor,
        config: Optional[dict] = None,
        n: int = 32,
    ) -> dict:
        """
        Compute Jaccard similarity across SAEs trained with different seeds.
        Measures concept robustness.

        Args:
            model_dirs: List of paths to model directories (one per seed).
            embeddings: Tensor (B, 512) for testing.
            config: Config override (optional).
            n: Number of top features to compare per sample.

        Returns:
            {"jaccard_matrix": Tensor (n_seeds, n_seeds),
             "mean_jaccard": float, "std_jaccard": float}
        """
        managers = []
        for d in model_dirs:
            mgr = SAEManager(config)
            mgr.load(d)
            managers.append(mgr)

        active_sets = []
        for mgr in managers:
            _, _, indices = mgr.encode_topk(embeddings)
            sets_per_sample = [set(row.tolist()) for row in indices.cpu()]
            active_sets.append(sets_per_sample)

        n_seeds = len(model_dirs)
        n_samples = embeddings.shape[0]
        jaccard_matrix = torch.zeros(n_seeds, n_seeds)

        for i in range(n_seeds):
            for j in range(i, n_seeds):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                    continue
                jaccards = []
                for s in range(n_samples):
                    a, b = active_sets[i][s], active_sets[j][s]
                    if len(a | b) > 0:
                        jaccards.append(len(a & b) / len(a | b))
                    else:
                        jaccards.append(0.0)
                mean_j = sum(jaccards) / len(jaccards)
                jaccard_matrix[i, j] = mean_j
                jaccard_matrix[j, i] = mean_j

        mask = torch.triu(torch.ones(n_seeds, n_seeds), diagonal=1).bool()
        upper_vals = jaccard_matrix[mask]

        return {
            "jaccard_matrix": jaccard_matrix,
            "mean_jaccard": upper_vals.mean().item(),
            "std_jaccard": upper_vals.std().item(),
        }

    def _check_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("SAE not loaded. Call .load(model_dir) or .train() first.")

    @property
    def _device(self) -> str:
        return self.config["device"]
