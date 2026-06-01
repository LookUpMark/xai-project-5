# xai_datasets/iu_xray.py
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class IUXrayImageDataset(Dataset):
    """Dataset handling images from IU Xray."""
    def __init__(self, image_dir: Path, image_ext: str = "*.png"):
        self.image_paths = sorted(list(image_dir.glob(image_ext)))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, str(img_path)


class IUXrayTextDataset(Dataset):
    """Dataset handling reports from IU Xray."""
    def __init__(self, reports_dir: Path):
        self.report_paths = sorted(list(reports_dir.glob("*.xml")))
        
    def __len__(self):
        return len(self.report_paths)
        
    def _parse_xml(self, xml_path: Path) -> str:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            findings = ""
            impression = ""
            
            for abstract in root.findall(".//AbstractText"):
                label = abstract.get("Label")
                if label == "FINDINGS" and abstract.text:
                    findings = abstract.text
                elif label == "IMPRESSION" and abstract.text:
                    impression = abstract.text
                    
            full_text = f"Findings: {findings} Impression: {impression}".strip()
            return full_text if len(full_text) > 30 else "No clinical report available."
        except Exception:
            return "Error parsing report."

    def __getitem__(self, idx):
        xml_path = self.report_paths[idx]
        text_content = self._parse_xml(xml_path)
        return text_content, str(xml_path)