import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import VLMConfig, EmbeddingConfig


def _unzip_collate(batch: list) -> tuple:
    """Collate that unzips (item, label) pairs into separate tuples.

    Defined at module level so it can be pickled by multiprocessing workers.
    """
    return tuple(zip(*batch))


def extract_visual_embeddings(
    model, processor, dataset: Dataset, 
    vlm_config: VLMConfig, embedding_config: EmbeddingConfig
):
    """
    It extracts and save embeddings by processing images from the specified folder.

    Args:
        model: loaded model
        processor: loaded processor
        dataset: the dataset you want to extract features on
        vlm_config (VLMConfig): model runtime parameters.
        embedding_config (EmbeddingConfig): I/O paths configuration.
    """

    print(f"\nFound {len(dataset)} images. Starting extraction...")

    dataloader = DataLoader(
        dataset,
        batch_size=vlm_config.batch_size,
        shuffle=False,
        num_workers=vlm_config.num_workers,
        collate_fn=_unzip_collate,
    )

    embedding_config.visual_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch_images, _ in tqdm(dataloader, desc="Images Processing"):
            # Input preparation
            inputs = processor.image_processor(
                images=list(batch_images), 
                return_tensors="pt"
            ).to(vlm_config.device)
            
            # Inference
            outputs = model.get_image_features(**inputs)  # (B, 512)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 Normalization

            # Moving on CPU to avoid VRAM problems
            all_embeddings.append(outputs.cpu())

    visual_embeddings = torch.cat(all_embeddings)  # (N, 512)
    torch.save(visual_embeddings, embedding_config.visual_output_path)

    print(f"Images Embedding Extraction completed. Saving on {embedding_config.visual_output_path}.")


def extract_text_embeddings(
    model, processor, dataset: Dataset, 
    vlm_config: VLMConfig, embedding_config: EmbeddingConfig
):
    """
    It extracts and save embeddings by processing textual reports from the specified folder.

    Args:
        model: loaded model
        processor: loaded processor
        dataset: the dataset you want to extract features on
        vlm_config (VLMConfig): model runtime parameters.
        embedding_config (EmbeddingConfig): I/O paths configuration.
    """
    print(f"\nFound {len(dataset)} reports. Starting extraction...")

    dataloader = DataLoader(
        dataset,
        batch_size=vlm_config.batch_size,
        shuffle=False,
        num_workers=vlm_config.num_workers,
        collate_fn=_unzip_collate,
    )
    
    embedding_config.text_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch_texts, _ in tqdm(dataloader, desc="Reports Processing"):
            # Input preparation
            inputs = processor.tokenizer(
                text=list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(vlm_config.device)

            # Inference
            outputs = model.get_text_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

            # Moving on CPU to avoid VRAM problems
            all_embeddings.append(outputs.cpu())

    text_embeddings = torch.cat(all_embeddings)
    torch.save(text_embeddings, embedding_config.text_output_path)
    print(f"Reports Embedding Extraction completed. Saving on {embedding_config.text_output_path}.")