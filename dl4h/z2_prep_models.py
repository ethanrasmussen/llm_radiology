from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
import os

# NOTE: Code requires `huggingface-cli login` via token

LLM_MODELS_DIR = ""

#### PRAGMATIC LLAMA ####
# NOTE: Use if loading checkpoints from paper/original model. Otherwise, follow the training/finetuning pipeline notebook with subset & LLaMA-2-7B.
# FROM: https://huggingface.co/dangnguyen0420/pragmatic-llama
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
# FROM: https://huggingface.co/allen-eric/radiology-llama2
# PAPER: https://arxiv.org/pdf/2309.06419
# RELATED: https://arxiv.org/pdf/2306.08666
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

#### CLINICAL-GPT ####
# FROM: https://huggingface.co/medicalai/ClinicalGPT-base-zh
# PAPER: https://arxiv.org/abs/2306.09968
# Load model directly
model = AutoModelForCausalLM.from_pretrained(
    "medicalai/ClinicalGPT-base-zh",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalGPT-base-zh")
# Save model files
SAVE_DIR = f"{LLM_MODELS_DIR}/clinical-gpt"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)