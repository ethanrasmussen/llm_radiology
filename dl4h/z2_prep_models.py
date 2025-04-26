from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
import os

# NOTE: Code requires `huggingface-cli login` via token

LLM_MODELS_DIR = ""

#### PRAGMATIC LLAMA ####
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

# TODO: Fix prep/import/save script below:
#### RADIALOG ####
# FROM: https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation
# PAPER: https://openreview.net/pdf?id=trUvr1gSNI
# Load model directly
processor = AutoProcessor.from_pretrained("ChantalPellegrini/RaDialog-interactive-radiology-report-generation")
model = AutoModelForCausalLM.from_pretrained(
    "ChantalPellegrini/RaDialog-interactive-radiology-report-generation",
    torch_dtype=torch.float16,
    device_map="auto",
)
# Save model files
SAVE_DIR = f"{LLM_MODELS_DIR}/radialog"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

#### SELFSYNTHX ####
# FROM: https://huggingface.co/YuchengShi/llava-med-v1.5-mistral-7b-chest-xray
# PAPER: https://arxiv.org/abs/2502.14044
# Load fused model
model = AutoModelForImageTextToText.from_pretrained(
    "YuchengShi/llava-med-v1.5-mistral-7b-chest-xray",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,  # Needed because ImageTextToText models are often custom
)
processor = AutoProcessor.from_pretrained(
    "YuchengShi/llava-med-v1.5-mistral-7b-chest-xray",
    trust_remote_code=True,
)
# Save model files
SAVE_DIR = f"{LLM_MODELS_DIR}/selfsynthx"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)