#!/usr/bin/env python3
import os
import re
import csv
import shutil
import argparse

def parse_study_report(path):
    """
    Read the .txt file at `path` and extract the INDICATION and IMPRESSION sections.
    Returns (indication, impression) as stripped strings, or (None, None) if either is missing.
    """
    text = open(path, 'r', encoding='utf-8', errors='ignore').read()
    # match from "INDICATION:" up to the next all-caps header (e.g. COMPARISON:, NOTIFICATION:, etc.) or EOF
    ind_pat = re.compile(r'INDICATION:(.*?)(?=\n\s*[A-Z ]{2,}:|\Z)', re.S)
    imp_pat = re.compile(r'IMPRESSION:(.*?)(?=\n\s*[A-Z ]{2,}:|\Z)', re.S)

    ind_m = ind_pat.search(text)
    imp_m = imp_pat.search(text)
    if not ind_m or not imp_m:
        return None, None

    indication = ind_m.group(1).strip().replace('\n', ' ')
    impression = imp_m.group(1).strip().replace('\n', ' ')
    return indication, impression

def main(text_root, image_root, output_dir):
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # paths for our two CSVs
    indications_csv = os.path.join(output_dir, 'indications.csv')
    reports_csv     = os.path.join(output_dir, 'ground_truth_reports.csv')

    # open both CSVs for writing
    with open(indications_csv, 'w', newline='', encoding='utf-8') as ind_f, \
         open(reports_csv,     'w', newline='', encoding='utf-8') as rep_f:

        ind_writer = csv.writer(ind_f)
        rep_writer = csv.writer(rep_f)

        # write headers
        ind_writer.writerow(['study_id', 'indication'])
        rep_writer.writerow(['study_id', 'report'])

        # traverse all .txt files under the text_root
        for root, _, files in os.walk(text_root):
            for fname in files:
                if not fname.lower().endswith('.txt'):
                    continue

                study_id = os.path.splitext(fname)[0]
                txt_path = os.path.join(root, fname)

                # parse out the two sections
                indication, impression = parse_study_report(txt_path)
                if not indication or not impression:
                    # skip if either section was missing
                    continue

                # compute the matching image folder:
                # e.g. if txt_path is TEXT/files/p12/p1201675/s56699142.txt
                # then rel = p12/p1201675/s56699142, so image_dir = image_root/p12/p1201675/s56699142
                rel = os.path.relpath(txt_path, text_root)
                rel_dir = os.path.splitext(rel)[0]  # strips .txt
                image_dir = os.path.join(image_root, rel_dir)

                if not os.path.isdir(image_dir):
                    print(f"[WARN] no image folder for study {study_id}: {image_dir}")
                    continue

                print(f"[!] Image folder located for {study_id}: {image_dir}")

                # pick the first JPG in that folder
                jpgs = sorted(f for f in os.listdir(image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg')))
                if not jpgs:
                    print(f"[WARN] no JPG in {image_dir} for study {study_id}")
                    continue

                src_jpg = os.path.join(image_dir, jpgs[0])
                dst_jpg = os.path.join(output_dir, f"{study_id}.jpg")
                try:
                    # copy jpg to destination
                    shutil.copy2(src_jpg, dst_jpg)
                    # write rows to each CSV
                    ind_writer.writerow([study_id, indication])
                    rep_writer.writerow([study_id, impression])
                except Exception as e:
                    print(f"[ERROR] copying {src_jpg} -> {dst_jpg}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract indications/impressions and gather one JPG per study."
    )
    p.add_argument("text_root",
                   help="Root of your text-report tree (e.g. /path/to/TEXT/files)")
    p.add_argument("image_root",
                   help="Root of your image tree (e.g. /path/to/IMAGES/files)")
    p.add_argument("output_dir",
                   help="Directory where indications.csv, ground_truth_reports.csv, and JPGs will be placed")
    args = p.parse_args()

    main(args.text_root, args.image_root, args.output_dir)
