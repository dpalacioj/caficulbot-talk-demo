#Script to download the Gemma-3N model from Hugging Face and save it locally.
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

model_id = "sergioq2/gemma-3N-finetune-coffe_q4_off"
local_dir = "./models"

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(f"Descargando {model_id}...")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=HUGGINGFACEHUB_API_TOKEN
)

print(f"Modelo descargado en: {local_dir}")
