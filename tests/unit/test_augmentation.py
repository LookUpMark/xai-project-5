import pytest
from unittest.mock import MagicMock
from PIL import Image

from src.config import AugmentationConfig
from xai_datasets.augmentation import AugmentedImageDataset
from src.augmentation.transforms import get_safe_cxr_transforms
import torchvision.transforms as T


class FakeBaseDataset:
    """A minimal fake dataset for testing the wrapper."""
    def __init__(self, n=2):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return a simple solid color image and a dummy id
        img = Image.new("RGB", (300, 300), color=(idx * 50, 100, 200))
        return img, f"fake_report_id_{idx}"


class TestAugmentedImageDataset:
    """Unit tests for the AugmentedImageDataset wrapper."""

    def test_dataset_length(self):
        """Dataset length should be base_len * (1 + num_augmentations)."""
        base_ds = FakeBaseDataset(n=3)
        config = AugmentationConfig(enabled=True, num_augmentations=4)
        aug_ds = AugmentedImageDataset(base_ds, config)
        
        assert len(aug_ds) == 3 * (1 + 4)
        assert len(aug_ds) == 15

    def test_returns_original_first(self):
        """The first element of every group (idx % (1+N) == 0) must be the original image."""
        base_ds = FakeBaseDataset(n=2)
        config = AugmentationConfig(enabled=True, num_augmentations=2)
        aug_ds = AugmentedImageDataset(base_ds, config)

        orig_img, orig_id = base_ds[0]
        aug_img, aug_id = aug_ds[0]

        # The IDs must be identical
        assert orig_id == aug_id
        # The images must be identically the same object/content for idx=0
        assert list(orig_img.getdata()) == list(aug_img.getdata())

    def test_multiple_images_share_same_id(self):
        """
        1 original + N augmented images must all map to the exact same report ID.
        """
        base_ds = FakeBaseDataset(n=2)
        config = AugmentationConfig(enabled=True, num_augmentations=3)
        aug_ds = AugmentedImageDataset(base_ds, config)

        # For base element 0, indices 0, 1, 2, 3 should all share the same ID
        expected_id_0 = "fake_report_id_0"
        for i in range(4):
            _, img_id = aug_ds[i]
            assert img_id == expected_id_0, f"Image {i} did not share the base ID!"

        # For base element 1, indices 4, 5, 6, 7 should all share the same ID
        expected_id_1 = "fake_report_id_1"
        for i in range(4, 8):
            _, img_id = aug_ds[i]
            assert img_id == expected_id_1, f"Image {i} did not share the base ID!"


class TestSafeCXRTransforms:
    """Unit tests for the CXR data augmentation pipeline."""

    def test_transform_pipeline_components(self):
        """Verify the safe transforms are correctly assembled."""
        config = AugmentationConfig(enabled=True, rotation_degrees=15, crop_scale=(0.8, 1.0))
        transforms = get_safe_cxr_transforms(config)

        assert isinstance(transforms, T.Compose)
        assert len(transforms.transforms) == 2
        
        # Verify RandomRotation is first
        assert isinstance(transforms.transforms[0], T.RandomRotation)
        assert transforms.transforms[0].degrees == [-15.0, 15.0]

        # Verify RandomResizedCrop is second
        assert isinstance(transforms.transforms[1], T.RandomResizedCrop)
        assert transforms.transforms[1].size == (224, 224)
        assert transforms.transforms[1].scale == (0.8, 1.0)

    def test_transform_output_type_and_size(self):
        """Verify the transform pipeline processes PIL images without crashing and resizes to 224x224."""
        config = AugmentationConfig(enabled=True)
        transforms = get_safe_cxr_transforms(config)
        
        orig_img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
        aug_img = transforms(orig_img)

        # Should remain a PIL Image (no ToTensor applied here yet, as BiomedCLIP processor does that later)
        assert isinstance(aug_img, Image.Image)
        # RandomResizedCrop enforces the size
        assert aug_img.size == (224, 224)
