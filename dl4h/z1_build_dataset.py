from z0_utils import runcmd

PHYSIONET_USER = ""
PHYSIONET_PASS = ""

IMAGE_FILENAMES_PATH = ""

MIMIC_CXR_SUBSET_SIZE = 100
MIMIC_CXR_DATASET_PATH = ""

MIMIC_CXR_REPORTS_PATH = ""

# Download IMAGE_FILENAMES for MIMIC-CXR-JPG
runcmd(f"wget -r -N -c -np --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -P {IMAGE_FILENAMES_PATH} https://physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES")

# Download subset of MIMIC-CXR-JPG
runcmd(f"head -n {MIMIC_CXR_SUBSET_SIZE} {IMAGE_FILENAMES_PATH} |wget -r -N -c -np -nH --cut-dirs=1 --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -i - -P {MIMIC_CXR_DATASET_PATH} --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/")

# Download MIMIC-CXR reports CSV
runcmd(f"wget -r -N -c -np --user {PHYSIONET_USER} --password {PHYSIONET_PASS} -P {MIMIC_CXR_REPORTS_PATH} https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip")
runcmd(f"unzip {MIMIC_CXR_REPORTS_PATH}/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip -d {MIMIC_CXR_REPORTS_PATH}")

# TODO: Generate Chexbert labels

# TODO: Process raw JPG images into tensor file