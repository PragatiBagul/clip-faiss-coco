#!/usr/bin/env python3
"""
Creates metadata CSV from COCO caption annotations.
Output CSV columns:
    file_path, caption, image_id

Usage:
python scripts/prepare_coco.py \
  --annotations data/coco/annotations/captions_train2017.json \
  --images_dir data/coco/images/train2017 \
  --output_csv data/coco/metadata_train.csv
"""

import json
import argparse
import pandas as pd
import os
import sys


def main(args):
    annotations_path = args.annotations
    images_dir = args.images_dir
    output_csv = args.output_csv

    if not os.path.exists(annotations_path):
        print(f"ERROR: annotations JSON not found: {annotations_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading COCO annotations from {annotations_path}")
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Expect typical COCO structure: keys "images" and "annotations"
    if isinstance(coco, dict) and "images" in coco and "annotations" in coco:
        images_list = coco["images"]
        annotations_list = coco["annotations"]
    else:
        # try common alternative shapes (e.g., some dumps use train2014/val2014)
        # find images list
        images_list = None
        annotations_list = None
        for k, v in coco.items() if isinstance(coco, dict) else []:
            if isinstance(v, list) and len(v) > 0:
                first = v[0]
                if isinstance(first, dict) and "file_name" in first and "id" in first:
                    images_list = v
                if isinstance(first, dict) and "image_id" in first and "caption" in first:
                    annotations_list = v
        if images_list is None or annotations_list is None:
            print("ERROR: Could not find 'images' and 'annotations' lists in the JSON.", file=sys.stderr)
            print("Top-level keys:", list(coco.keys()) if isinstance(coco, dict) else "JSON root not a dict", file=sys.stderr)
            sys.exit(3)

    # Build image_id -> file_name map
    images = {}
    for img in images_list:
        if "id" in img and ("file_name" in img or "file_name" in img):
            images[int(img["id"])] = img.get("file_name")
        elif "id" in img and "file_name" in img:
            images[int(img["id"])] = img.get("file_name")
        else:
            # skip entries that don't match expected shape
            continue

    rows = []
    skipped = 0
    for ann in annotations_list:
        if "image_id" not in ann or "caption" not in ann:
            skipped += 1
            continue

        image_id = int(ann["image_id"])
        caption = ann["caption"]
        file_name = images.get(image_id)
        if file_name is None:
            skipped += 1
            continue

        file_path = os.path.join(images_dir, file_name)

        rows.append({
            "image_id": image_id,
            "file_path": file_path,
            "caption": caption
        })

    if len(rows) == 0:
        print("ERROR: No rows produced. Possible mismatch between annotations and images.", file=sys.stderr)
        sys.exit(4)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"✔️ Saved metadata CSV: {output_csv}")
    print(f"Total rows: {len(df)} (skipped {skipped} annotations that couldn't be mapped)")
    print("Sample rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="COCO captions JSON file")
    parser.add_argument("--images_dir", required=True, help="Folder containing COCO images")
    parser.add_argument("--output_csv", required=True, help="Where to save metadata CSV")
    args = parser.parse_args()
    main(args)
