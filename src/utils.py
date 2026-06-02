from transformers import AutoModel, AutoProcessor

from config import VLMConfig

def load_vlm(config: VLMConfig):
    """    
    Args:
        config (VLMConfig): dataclass containing parameters.
        
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
    
    model.eval().to(config.device)
    
    return model, processor


def load_radlex_terms(csv_path: str) -> list[str]:
    """
    Load RadLex CSV and return a deduplicated list of non-obsolete 
    preferred labels.

    Args:
        csv_path (str): path to the RadLex CSV file.

    Returns:
        List[str]: cleaned RadLex terms.
    """
    import csv
    
    terms = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip obsolete
            if row.get("Obsolete", "").strip().upper() == "TRUE":
                continue
            
            label = row.get("Preferred Label")
            if label and label.strip():
                terms.append(label.strip())

    # Deduplicate while preserving order, drop very short labels
    seen = set()
    unique_terms = []
    for t in terms:
        t_lower = t.lower()
        if t_lower not in seen and len(t) > 1:
            seen.add(t_lower)
            unique_terms.append(t)

    print(f"Loaded {len(unique_terms)} unique non-obsolete RadLex terms "
          f"(from {len(terms)} raw rows).")
    return unique_terms