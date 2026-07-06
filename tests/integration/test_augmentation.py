import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import VLMConfig, EmbeddingConfig, AugmentationConfig
from embedding_extraction.extract_embeddings import extract_visual_embeddings
from unittest.mock import patch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegrationAugmentationPipeline:
    """
    Integration test: loads the real BiomedCLIP model and extracts embeddings
    from an AugmentedImageDataset to verify that the real model processes 
    augmented images correctly and outputs the correct repeated IDs.
    """

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        from utils import load_vlm
        config = VLMConfig()
        model, processor = load_vlm(config)
        yield model, processor
        del model
        torch.cuda.empty_cache()

    def test_extract_augmented_images(self, model_and_processor, tmp_path):
        """
        End-to-end test:
        1. Find a real image.
        2. Set num_augmentations = 2.
        3. Extract visual embeddings.
        4. Verify that we get 1+2=3 embeddings.
        5. Verify that visual_image_ids.json has 3 identical IDs.
        """
        model, processor = model_and_processor

        # --- Locate a real image ----------------------------------------
        project_root = Path(__file__).parent.parent.parent
        image_dir = project_root / "data" / "iu_xray" / "images" / "images_normalized"
        real_images = sorted(image_dir.glob("*.png"))
        if not real_images:
            pytest.skip(
                f"No PNG images in {image_dir} — IU X-Ray not downloaded locally "
                "(data/iu_xray/* is gitignored). Run python xai_datasets/download_iu_xray.py."
            )

        real_image_path = real_images[0]
        real_image_name = real_image_path.name

        # -- Build single-element dataset --------------------------------
        class SingleImageDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                img = Image.open(real_image_path).convert("RGB")
                return img, real_image_name

        # -- Configs for integration test ---------------------------------
        vlm_config = VLMConfig(
            batch_size=2,
            num_workers=0,
            device="cuda",
        )
        embedding_config = EmbeddingConfig(
            output_base=str(tmp_path / "embeddings"),
        )
        
        # We enforce augmentation for this specific integration test
        aug_config = AugmentationConfig(
            enabled=True, 
            num_augmentations=2, 
            rotation_degrees=10, 
            crop_scale=(0.9, 1.0)
        )

        dataset = SingleImageDataset()

        with patch("embedding_extraction.extract_embeddings.augmentation", aug_config), \
             patch("config.augmentation", aug_config):
            extract_visual_embeddings(
                model, processor, dataset, vlm_config, embedding_config
            )
            visual_emb_path = embedding_config.visual_output_path

        assert visual_emb_path.parent.name == "augmented", "It should have saved into the 'augmented' folder!"
        
        assert visual_emb_path.exists()
        visual_emb = torch.load(visual_emb_path, weights_only=True)

        # 1 base image + 2 augmentations = 3 embeddings
        assert visual_emb.shape == (3, 512), (
            f"Visual embedding shape mismatch: expected (3, 512) got {visual_emb.shape}"
        )

        # Verify they are correctly L2 Normalized by the real model
        norms = visual_emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3).to("cuda" if norms.is_cuda else "cpu"), atol=1e-4)

        # Verify Sidecar IDs
        sidecar_path = visual_emb_path.with_name("visual_image_ids.json")
        assert sidecar_path.exists(), "Sidecar ID file was not saved!"
        
        with open(sidecar_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
            
        assert len(ids) == 3, f"Expected 3 IDs, got {len(ids)}"
        
        # All 3 IDs must be absolutely identical to map back to the same report
        assert ids[0] == real_image_name
        assert ids[1] == real_image_name
        assert ids[2] == real_image_name
