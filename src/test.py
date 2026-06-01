import torch
from transformers import pipeline
from PIL import Image

# 1. Inizializza la pipeline multimodale (Image-to-Text)
# Sfrutta il modello ottimizzato da Unsloth
pipe = pipeline(
    "image-text-to-text", 
    model="unsloth/medgemma-4b-it", 
    torch_dtype=torch.bfloat16,  # Consuma meno VRAM mantenendo la precisione
    device_map="auto"            # Assegna automaticamente il modello alla GPU
)

# 2. Carica l'immagine medica (es. una radiografia locale)
image_path = "./radiografia_torace.png"
medical_image = Image.open(image_path)

# 3. Struttura il messaggio secondo il template ufficiale del modello
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an expert radiologist. Analyze the medical image accurately."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the findings in this chest X-ray and check for abnormalities."},
            {"type": "image", "image": medical_image}
        ]
    }
]

# 4. Genera il testo (il referto o l'analisi)
outputs = pipe(text=messages, max_new_tokens=256)

# 5. Stampa il risultato generato da MedGemma
print('ciao')
print(outputs[0]["generated_text"][-1]["content"])