from z0_utils import runcmd

# Input items:
LLM_MODEL_DIR_PATH = ""
VISION_MODEL_DIR_PATH = ""
INDICATION_CSV_PATH = ""
IMAGE_TENSOR_PATH = ""

# Output locations:
PREDICTED_LABELS_PATH = ""
REPORTS_PATH = ""

runcmd(f"python pragmatic_llama_inference.py --llama-path {LLM_MODEL_DIR_PATH} --vision_path {VISION_MODEL_DIR_PATH} --indication_path {INDICATION_CSV_PATH} --image_path {IMAGE_TENSOR_PATH} --vision_out_path {PREDICTED_LABELS_PATH} --outpath {REPORTS_PATH}")
