from z0_utils import runcmd
import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

PHYSIONET_USER = ""
PHYSIONET_PASS = ""

# Download IMAGE_FILENAMES for MIMIC-CXR-JPG
IMAGE_FILENAMES_PATH = ""
runcmd(f"wget -r -N -c -np --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -P {IMAGE_FILENAMES_PATH} https://physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES")

# Download subset of MIMIC-CXR-JPG
MIMIC_CXR_SUBSET_SIZE = 100
MIMIC_CXR_DATASET_PATH = ""
runcmd(f"head -n {MIMIC_CXR_SUBSET_SIZE} {IMAGE_FILENAMES_PATH}/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES |wget -r -N -c -np -nH --cut-dirs=1 --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -i - -P {MIMIC_CXR_DATASET_PATH} --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/")

# Download MIMIC-CXR reports CSV
MIMIC_CXR_REPORTS_PATH = ""
runcmd(f"wget -r -N -c -np --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -P {MIMIC_CXR_REPORTS_PATH} https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip")
runcmd(f"unzip {MIMIC_CXR_REPORTS_PATH}/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip -d {MIMIC_CXR_REPORTS_PATH}")

# Reorganize dataset, moving JPG files, and creating CSV files to track indications & ground truth reports
PARSED_DATASET_PATH = ""
runcmd(f"python z0_parse_dataset.py {MIMIC_CXR_REPORTS_PATH}/files {MIMIC_CXR_DATASET_PATH}/files {PARSED_DATASET_PATH}")

# Process raw JPG images into tensor file
df = pd.read_csv(f"{PARSED_DATASET_PATH}/indications.csv").fillna("")
# assume df['study_id'] matches your image filenames (e.g. "12345.jpg")
study_ids = df['study_id'].astype(str).tolist()
# Build the exact transforms used by DenseChexpertModel at training time
prep = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),      # if model uses 3‑channel input
    transforms.Resize((320, 320)),                    # 320×320 CheXpert default
    transforms.ToTensor(),                            # scales to [0,1]
    transforms.Normalize(                             # imagenet‐style norm
        mean=[0.485, 0.485, 0.485],
        std =[0.229, 0.229, 0.229],
    ),
])
# Load each JPEG in the exact order of your CSV
imgs = []
for sid in study_ids:
    path = os.path.join(PARSED_DATASET_PATH, f"{sid}.jpg")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image for study_id={sid} not found at {path}")
    img = Image.open(path).convert("RGB")
    imgs.append(prep(img))
# Stack into one big Tensor [N,3,320,320] and save
images = torch.stack(imgs)
torch.save(images, f"{PARSED_DATASET_PATH}/image_tensor.pt")
print(f"Saved {len(imgs)} preprocessed images -> {PARSED_DATASET_PATH}/image_tensor.pt")
