from z0_utils import runcmd

GROUND_TRUTH_REPORTS_CSV_PATH = ""
GENERATED_REPORTS_CSV_PATH = ""

EVAL_OUTPUTS_PATH = ""

runcmd(f"python evaluate.py --gt_path {GROUND_TRUTH_REPORTS_CSV_PATH} --gen_path {GENERATED_REPORTS_CSV_PATH} --out_path {EVAL_OUTPUTS_PATH}")
