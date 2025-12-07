#!/usr/bin/env python3
"""
Create train.jsonl and val.jsonl for LoRA fine-tuning.

Each output line is a JSON object:
{
  "context": "<retrieved captions ...>",   # concatenated evidence lines (used as input context)
  "image_caption": "<gt caption>",         # short ground-truth caption (one of COCO captions)
  "expanded": "<long caption>",            # longer human-like caption (we create by joining multiple GT captions)
  "why": "<rationales>"                    # optional: synthetic rationale (concatenated key phrases)
}

Strategies:
- If FAISS index + embeddings are available, for each target image we find top-(k) neighbors and use their captions as context.
- Else: sample random captions from the dataset.
- We split 95/5 by default into train / val.
"""

import os
import json
import random
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# optional import of faiss/numpy for neighbor search
try:
    import numpy as np
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def read_metadata(metadata_csv: str):
    df = pd.read_csv(metadata_csv)
    # expect columns: image_id, file_path, caption
    return df

def build_caption_pool(df):
    # group captions by image_id
    by_img = {}
    for _, row in df.iterrows():
        img_id = int(row["image_id"])
        by_img.setdefault(img_id, []).append(row["caption"])
    return by_img

def sample_context_from_neighbors(
    target_idx,
    neighbor_indices,
    idx_to_imageids,
    imageid_to_captions,
    k=5
):
    # neighbor_indices: list of integer positions into the original df
    captions = []
    for ni in neighbor_indices[:k]:
        imgid = idx_to_imageids.get(ni)
        if imgid is None:
            continue
        caps = imageid_to_captions.get(imgid, [])
        if caps:
            # take first caption (or random)
            captions.append(random.choice(caps))
    return captions

def main(args):
    random.seed(args.seed or 42)
    os.makedirs(args.output_dir, exist_ok=True)

    df = read_metadata(args.metadata)
    print("Loaded metadata rows:", len(df))

    # Build mapping from row index -> image_id and image_id -> captions
    idx_to_imageid = {}
    imageid_to_captions = {}
    for i, row in df.reset_index().iterrows():
        idx_to_imageid[i] = int(row["image_id"])
        imageid_to_captions.setdefault(int(row["image_id"]), []).append(row["caption"])

    # If embeddings+faiss provided, load and query neighbors
    use_faiss = False
    if args.faiss_index and args.embeddings and _HAS_FAISS:
        if os.path.exists(args.faiss_index) and os.path.exists(args.embeddings):
            print("Loading embeddings and FAISS index for neighbor queries...")
            emb = np.load(args.embeddings).astype("float32")
            index = faiss.read_index(args.faiss_index)
            use_faiss = True
        else:
            print("FAISS index or embeddings not found; falling back to random sampling.")
            use_faiss = False

    rows = []
    for i in tqdm(range(len(df))):
        # target image's captions (may be multiple rows)
        imgid = idx_to_imageid.get(i)
        target_caps = imageid_to_captions.get(imgid, [])
        if not target_caps:
            continue
        # choose a gt caption to be the short caption
        short_caption = random.choice(target_caps)
        # expanded caption: join all captions for the image into 1-2 sentences
        expanded = " ".join(target_caps[:3])
        # rationale: join nouns/phrases (here we just join first words as synthetic)
        why = " ; ".join([c.split(",")[0] for c in target_caps[:3]])

        # Build context/evidence
        evidence_captions = []
        if use_faiss:
            # query the index with this image's embedding
            # find rows in df that belong to this image; pick first one to locate embedding row
            # NOTE: this assumes the embeddings rows align with df row order
            D, I = index.search(emb[i:i+1], args.k_neighbors+1)
            neighbors = I[0].tolist()
            # remove self if present
            neighbors = [n for n in neighbors if n != i]
            evidence_captions = sample_context_from_neighbors(i, neighbors, idx_to_imageid, imageid_to_captions, k=args.k_neighbors)
        else:
            # sample random captions from other images
            all_img_ids = list(imageid_to_captions.keys())
            sampled = random.sample(all_img_ids, min(len(all_img_ids), args.k_neighbors+2))
            for sid in sampled:
                if sid == imgid:
                    continue
                caps = imageid_to_captions.get(sid, [])
                if caps:
                    evidence_captions.append(random.choice(caps))
                if len(evidence_captions) >= args.k_neighbors:
                    break

        # format context
        context = " ||| ".join(evidence_captions)

        example = {
            "context": context,
            "image_caption": short_caption,
            "expanded": expanded,
            "why": why
        }
        rows.append(example)

    # Shuffle and split
    random.shuffle(rows)
    n = len(rows)
    n_val = int(n * args.val_frac)
    val = rows[:n_val]
    train = rows[n_val:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    print(f"Writing train {len(train)} -> {train_path}")
    with open(train_path, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Writing val {len(val)} -> {val_path}")
    with open(val_path, "w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=str, default="data/coco/metadata_train.csv", help="COCO metadata CSV")
    p.add_argument("--faiss_index", type=str, default="indexes/faiss_index.idx", help="FAISS index path (optional)")
    p.add_argument("--embeddings", type=str, default="data/embeddings/images.npy", help="Embeddings file (optional)")
    p.add_argument("--output_dir", type=str, default="llama/data", help="Where to write train/val jsonl")
    p.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbor captions to include as context")
    p.add_argument("--val_frac", type=float, default=0.05, help="Fraction for validation set")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
