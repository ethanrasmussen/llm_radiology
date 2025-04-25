from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# NOTE: Code requires `huggingface-cli login` via token

LLM_MODELS_DIR = ""

#### PRAGMATIC LLAMA ####
# Load fused model
model = AutoModelForCausalLM.from_pretrained(
    "dangnguyen0420/pragmatic-llama",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("dangnguyen0420/pragmatic-llama")
# Save model files
SAVE_DIR = f"{LLM_MODELS_DIR}/pragmatic-llama"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

#### RADIOLOGY LLAMA 2 ####
# Load base & adapter:
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "allen-eric/radiology-llama2")
# Merge adapter weights into base
model = model.merge_and_unload()
# Save model files
SAVE_DIR = f"{LLM_MODELS_DIR}/radiology-llama2"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf").save_pretrained(SAVE_DIR)