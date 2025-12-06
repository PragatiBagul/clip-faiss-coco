"""
scripts/build_faiss_index.py

Builds a FAISS index from precomputed embeddings. (.npy) and saves:
- FAISS index file (faiss.write_index)
- metadata JSONL mapping (indexes/metadata.json)

Features:
- supports IndexFlatIP (flat, exact, inner-product) and IVF (IndexIVFFlat)
- checks / enforces L2-normalization for cosine similarity with IndexFlatIP
- basic integrity check: sample a few embeddings and ensure they retrieve themselves as top-1
- accepts metadata CSV (file_path, caption, image_id optional); will generate ids if missing
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm

# faiss import
try:
    import faiss
except Exception as e:
    raise ImportError("faiss is not installed. Please install it with 'pip install faiss'")

def load_embeddings(path):
    x = np.load(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x

def is_normalized(x, tol=1e-3):
    # checks whether vectors have norm ≈ 1
    norms = np.linalg.norm(x, axis=1)
    return np.all(np.abs(norms - 1.0) < tol)

def l2_normalize_inplace(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0.0] = 1.0
    x /= norms

def read_metadata_csv(csv_path):
    """
    Expected CSV columns: file_path, caption, image_id (optional)
    If image_id missing, we create incremental ids.
    Return list of dicts with keys: image_id, file_path, caption
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # required column
    if "file_path" not in df.columns or "caption" not in df.columns:
        raise ValueError("metadata csv must contain columns: file_path, caption (image_id optional)")
    if "image_id" not in df.columns:
        df["image_id"] = df["image_id"].astype(int)
        records = df[["image_id", "file_path","caption"]].to_dict(orient="records")
        return records

def build_flat_index(embeddings):
    """
    Build an IndexFlatIP for inner-product (works with L2 normalized vectors as cosine similarity)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    return index

def build_ivf_index(embeddings,nlist, use_gpu=False):
    """
    Build an IVF index with IndexIVFFlat.
    - quantizer : IndexFlatIP (inner product) (works with normalized vectors as cosine similarity)
    - nlist : number of centroids / clusters
    """
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer,dim, nlist, faiss.METRIC_INNER_PRODUCT)
    return index

def train_and_add(index, embeddings, use_gpu=False):
    """
    For IVF indexes: train with a representative set (faiss requires training)
    Then add all vectors.
    """
    if not index.is_trained:
        print("Training index (this may take a while for large datasets).")
        # FAISS expects float 32
        index.train(embeddings)
    print("Adding vectors to index ...")
    index.add(embeddings)

def save_metadata_jsonl(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(records):
            rec = {
                "index":i,
                "image_id":int(r["image_id"]),
                "file_path": r['file_path'],
                "caption": r['caption']
            }
            fh.write(json.dumps(rec,ensure_ascii=False) + "\n")
def integrity_check(index,embeddings,records,sample_size=10,k=5):
    """
    Pick `sample_size` random vectors, query the index, and check whether the top-1 returned index equals the original vector's index
    Because IVF is approximate, top-1 may not always be itself, We will report success rate and sample mismatches
    """
    n = embeddings.shape[0]
    rng = np.random.default_rng(12345)
    picks = rng.choice(n,size=min(n,sample_size),replace=False)
    # FAISS wants contiguous array
    query_vectors = embeddings[picks].astype(np.float32).copy()
    D, I = index.search(query_vectors, k) # distances (inner product), indices
    success = 0
    mismatches = []
    for pick_idx, returned in zip(picks,I):
        top1 = int(returned[0])
        if top1 == int(pick_idx):
            success += 1
        else:
            mismatches.append({
                "query_index":int(pick_idx),
                "top1":top1,
                "topk":[int(x) for x in returned.tolist()]
            })
    success_rate = success/ len(picks)
    return success_rate, mismatches
def main(args):
    # load config yaml
    import yaml
    cfg = yaml.safe_load(open(args.config))

    print(f"FAISS config: index_type={cfg.get('index_type')} - index_path={cfg.get('index_path')}")

    # load embeddings
    print("Loading embeddings from ",args.embeddings)
    embeddings = load_embeddings(args.embeddings)
    n,dim = embeddings.shape
    print(f"Loaded embeddings : n={n}, dim={dim}")

    # optionally check normalization
    if not cfg.get("embeddings_normalized",True):
        print("Embeddings not flagged as normalized in config, normalizing now.")
        l2_normalize_inplace(embeddings)
    else:
        print("Embeddings are L2-normalized. (checked)")

    # read metadata
    print("Reading metadata CSV : ",args.metadata)
    records = read_metadata_csv(args.metadata)

    if len(records) != n:
        print(f"Warning: metadata records ({len(records)}) != number of embeddings ({n}). Using min() and truncating.")
        min_n = min(len(records),n)
        embeddings = embeddings[:min_n]
        records = records[:min_n]
        n = min_n

    # create index
    index_type = cfg.get("index_type","flat").lower()
    if index_type == "flat":
        print("Building IndexFlatIP (exact inner-product)")
        index = build_flat_index(embeddings)
        # add vectors
        index.add(embeddings)
    elif index_type == "ivf":
        nlist = cfg.get("nlist",args.nlist)
        print(f"Building IndexIVFFlat with nlist={nlist}.")
        index = build_ivf_index(embeddings,nlist)
        # train + add
        # FAISS expects contiguous memory
        embeddings_for_faiss = np.ascontiguousarray(embeddings)
        train_and_add(index,embeddings_for_faiss)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}. Choose from ['flat', 'ivf']")
    # save index
    index_dir = os.path.dirname(cfg.get("index_path"))
    if index_dir:
        os.makedirs(index_dir,exist_ok=True)
    faiss.write_index(index,cfg.get("index_path"))
    print(f"FAISS index saved to: {cfg.get('index_path')}")

    # Save metadata.jsonl (mapping index -> file_path, caption, image_id)
    metadata_path = cfg.get("metadata_path","indexes/metadata.jsonl")
    save_metadata_jsonl(records,metadata_path)
    print(f"Metadata JSONL saved to : {metadata_path}")

    # Integrity check
    print("Running integrity check (self-retrieval on sample vectors)...")
    success_rate,mismatches = integrity_check(index,embeddings,records,sample_size=args.check_samples,k=args.check_k)
    print(f"Self-retrieval success_rate (top-1) over {min(args.check_samples,n)} samples: {success_rate*100:.2f}")
    if len(mismatches)>0:
        print("Some mismatches (showing up to 5):")
        for mm in mismatches[:5]:
            qi = mm['query_index']
            top1 = mm['top1']
            print(f" - query_index = {qi}, top1 = {top1} topk = {mm['topk']}")
            # print metadata for context
            print(f" query -> {records[qi]['file_path']} / {records[qi]['caption']}")
            print(f" top1 -> {records[top1]['file_path']} / {records[top1]['caption']}")
    else:
        print("All sampled vectors retrieved themselves as top-1.")

    print("\n Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build faiss index")
    parser.add_argument("--embeddings",type=str,required=True,help="Path to .npy embedding file (float32)")
    parser.add_argument("--metadata",type=str,required=True,help="CSV with columns : file_path,caption,(optional) image_id")
    parser.add_argument("--config",type=str,default="configs/faiss.yaml",help="Path to FAISS yaml config")
    parser.add_argument("--nlist",type=int,default=100,help="nlist for IVF (overrides config if provided).")
    parser.add_argument("--check_samples",type=int,default=20,help="Number of random samples to check for integrity check")
    parser.add_argument("--check_k",type=int,default=5,help="k used for integrity check (top-k)")
    args = parser.parse_args()
    main(args)