from typing import Any, Tuple
from torch.utils.data import Dataset
from src.config import AugmentationConfig
from src.augmentation.transforms import get_safe_cxr_transforms


class AugmentedImageDataset(Dataset):
    """
    A generic wrapper that applies safe data augmentations to any underlying
    image dataset (e.g., IUXrayImageDataset).

    For an underlying dataset of size M and `num_augmentations=N`, 
    this dataset will have size M * (1 + N).
    It dynamically returns the original image, followed by N augmented versions,
    keeping the IDENTICAL identifier for all versions so that downstream pipelines
    (like LLM Judges) can still link them back to the original textual reports.
    """

    def __init__(self, base_dataset: Dataset, config: AugmentationConfig):
        self.base_dataset = base_dataset
        self.config = config
        self.n_aug = config.num_augmentations
        self.transforms = get_safe_cxr_transforms(config)

    def __len__(self) -> int:
        return len(self.base_dataset) * (1 + self.n_aug)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        # Determine which original sample this corresponds to
        base_idx = idx // (1 + self.n_aug)
        aug_idx = idx % (1 + self.n_aug)

        # Get original image and identifier (usually a path or ID)
        img, base_id = self.base_dataset[base_idx]

        if aug_idx == 0:
            # Return original unmodified image
            return img, base_id
        else:
            # Return augmented image with the SAME exact base_id.
            # This ensures that downstream tasks (like LLM Judges) can still
            # join this embedding back to the original textual report.
            aug_img = self.transforms(img)
            return aug_img, base_id
