import torch
from transformers import AutoModel, AutoProcessor
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import VLMConfig

def load_vlm(config: VLMConfig):
    """    
    Args:
        config (VLMConfig): dataclass containing model parameters.
        
    Returns:
        tuple: (model, processor) loaded.
    """    
    model = AutoModel.from_pretrained(
        config.model_name, 
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        config.processor_name, 
        trust_remote_code=True
    )
    
    model.eval().to("cuda")
    
    return model, processor


def extract_embeddings(model, processor, config: VLMConfig):
    """
    It extracts and save embeddings by processing samples from the specified folder.

    Args:
        model: loaded model
        processor: loaded processor
        config (VLMConfig): dataclass containing model parameters.
    """

    # Visual Embeddings
    img_dir = Path(config.image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Error: Folder {img_dir} not found!")
    
    image_paths = sorted(list(img_dir.glob(config.image_ext)))
    
    if not image_paths:
        raise ValueError(f"No image found having extension '{config.image_ext}' in {img_dir}")
        
    print(f"Found {len(image_paths)} images. Starting extraction...")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), config.batch_size), desc="Images Feature Extraction"):
            batch_paths = image_paths[i : i + config.batch_size]
            
            # Loading images
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Input preparation
            inputs = processor(images=images, return_tensors="pt", padding=True).to(config.device)
            
            # Inference
            outputs = model.get_image_features(**inputs)  # (B, 512)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 Normalization
            
            # Moving on CPU to avoid VRAM problems
            all_embeddings.append(outputs.cpu())

    visual_embeddings = torch.cat(all_embeddings)  # (N, 512)
    torch.save(visual_embeddings, config.output_path)