import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import VLMConfig, EmbeddingConfig, augmentation
from xai_datasets.augmentation import AugmentedImageDataset


def _unzip_collate(batch: list) -> tuple:
    """Collate that unzips (item, label) pairs into separate tuples.

    Defined at module level so it can be pickled by multiprocessing workers.
    """
    return tuple(zip(*batch))


def _autocast_ctx(vlm_config: VLMConfig):
    """fp16 autocast context on CUDA only (no-op on MPS/CPU or when use_half is off).

    Outputs are cast back to fp32 by the caller before L2-norm + save, so the
    saved tensors are fp32 regardless — downstream stages see no dtype change.
    """
    if vlm_config.use_half and vlm_config.device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False) if vlm_config.device.startswith("cuda") \
        else _nullcontext()


class _nullcontext:
    """Lightweight no-op context manager (avoids a contextlib import)."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _dataloader_kwargs(vlm_config: VLMConfig) -> dict:
    """DataLoader kwargs tuned for GPU throughput without blowing host RAM.

    - pin_memory + persistent_workers keep the GPU fed when image decode (PNG)
      is the bottleneck rather than forward compute.
    - prefetch_factor stays at 2 (torch default): workers buffer PIL images, so
      in-flight memory ~= num_workers * prefetch_factor * batch_size * (PIL img).
      prefetch_factor=4 + batch=256 + 4 workers = ~4k buffered images, which the
      OOM killer will SIGKILL on host-RAM-constrained boxes (e.g. Lightning
      studios). If you raise batch_size a lot, lower num_workers or use --no-half
      is NOT the lever — host RAM is.
    """
    is_cuda = vlm_config.device.startswith("cuda")
    kwargs = {
        "batch_size": vlm_config.batch_size,
        "shuffle": False,
        "num_workers": vlm_config.num_workers,
        "collate_fn": _unzip_collate,
        "pin_memory": is_cuda,
    }
    if vlm_config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


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

    if augmentation.enabled:
        dataset = AugmentedImageDataset(dataset, augmentation)
        print(f"\nAugmentation enabled: size expanded to {len(dataset)} samples. Starting extraction...")
    else:
        print(f"\nFound {len(dataset)} images. Starting extraction...")

    dataloader = DataLoader(dataset, **_dataloader_kwargs(vlm_config))

    embedding_config.visual_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = []
    all_image_ids = []  # basename per row, kept in lockstep with the embeddings

    if vlm_config.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True  # fixed image shape post-processor
    model.eval()
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, desc="Images Processing"):
            # Input preparation
            inputs = processor.image_processor(
                images=list(batch_images),
                return_tensors="pt"
            ).to(vlm_config.device)

            # Inference (fp16 autocast on CUDA; cast to fp32 before norm + save)
            with _autocast_ctx(vlm_config):
                outputs = model.get_image_features(**inputs)  # (B, 512)
            outputs = outputs.float()
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 Normalization

            # Moving on CPU to avoid VRAM problems
            all_embeddings.append(outputs.cpu())
            # Preserve the image identity (the tensor itself is bare (N, 512))
            all_image_ids.extend(Path(p).name for p in batch_paths)

    visual_embeddings = torch.cat(all_embeddings)  # (N, 512)
    torch.save(visual_embeddings, embedding_config.visual_output_path)

    # Sidecar JSON of image ids (basename) aligned row-for-row with the tensor.
    # Consumed downstream by the train/test split and generate_explanations so
    # the LLM judge can join concepts back to reports.csv on image_id.
    image_ids_path = embedding_config.visual_output_path.with_name("visual_image_ids.json")
    with open(image_ids_path, "w") as f:
        json.dump(all_image_ids, f)

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

    dataloader = DataLoader(dataset, **_dataloader_kwargs(vlm_config))

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

            # Inference (fp16 autocast on CUDA; cast to fp32 before norm + save)
            with _autocast_ctx(vlm_config):
                outputs = model.get_text_features(**inputs)
            outputs = outputs.float()
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

            # Moving on CPU to avoid VRAM problems
            all_embeddings.append(outputs.cpu())

    text_embeddings = torch.cat(all_embeddings)
    torch.save(text_embeddings, embedding_config.text_output_path)
    print(f"Reports Embedding Extraction completed. Saving on {embedding_config.text_output_path}.")