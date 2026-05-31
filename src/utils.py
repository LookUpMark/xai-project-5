from transformers import AutoModel, AutoProcessor

from config import VLMConfig


def load_vlm(config: VLMConfig):
    """Load BiomedCLIP model and processor.

    Args:
        config (VLMConfig): dataclass containing parameters.

    Returns:
        tuple: (model, processor) loaded.
    """
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.processor_name,
        trust_remote_code=True,
    )

    model.eval().to(config.device)

    return model, processor