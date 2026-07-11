import torchvision.transforms as T
from src.config import AugmentationConfig


def get_safe_cxr_transforms(config: AugmentationConfig) -> T.Compose:
    """
    Returns a torchvision.transforms.Compose pipeline with safe data augmentations
    for Chest X-Rays.

    Safe augmentations included:
    - Small-angle random rotations.
    - Light random resized crop to prevent center-bias without losing anatomy.
    """
    return T.Compose([
        T.RandomRotation(degrees=config.rotation_degrees),
        T.RandomResizedCrop(
            size=(224, 224),  # Standard input size, might be resized again by BiomedCLIP's transform later
            scale=config.crop_scale,
            ratio=(0.95, 1.05)
        )
    ])
