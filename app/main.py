"""
FASTAPI Backend for CLIP + FAISS bi-model retrieval.

Endpoints:
- GET /health
- POST /search/text (JSON body : {"query":"a dog on a skateboard","k":10}
- POST /search/image (mutlipart file or JSON base64: {"image":"<base64>","k":10})

Startup loads:
- CLIP model (same model used to create embeddings)
- FAISS index (indexes/faiss_index.idx)
- metadata JSONL (indexes/metadata.jsonl) mapping index -> file_path, caption, image_id

Behavior:
- Query embeddings are L2-normalized when configs/clip.yaml -> normalize_embeddings is true
- For IVF indices, index.nprobe is set from configs/faiss.yaml before search
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import io
import os
import json
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip
import faiss

from src.preprocess import CLIPPreprocessor

# Config / Constants

CLIP_CONFIG_PATH = "configs/clip.yaml"
FAISS_CONFIG_PATH = "configs/faiss.yaml"
METADATA_JSON_DEFAULT = "indexes/metadata.jsonl"
FAISS_INDEX_DEFAULT = "indexes/faiss_index.idx"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Pydantic request/response models
class TextQuery(BaseModel):
    query: str
    k: Optional[int] = 10
class ImageQuery(BaseModel):
    image: str
    k: Optional[int] = 10

# APP Initialization
app = FastAPI(title="CLIP+FAISS Retrieval API",version="0.1")

class AppState:
    """
    Container for global objects loaded at startup
    """
    model = None
    device = "cpu"
    preprocess = None
    clip_config = {}
    faiss_config = {}
    index = None
    metadata = []
    index_is_ivf = False

state = AppState()

# Utility helpers
def load_yaml(path):
    with open(path, "r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_metadata_jsonl(path):
    meta = []
    with open(path,"r",encoding="utf-8") as fh:
        for line in fh:
            meta.append(json.loads(line))
    return meta

def l2_normalize(x:np.ndarray):
    norms = np.linalg.norm(x,axis=1,keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms

def prepare_image_tensor_from_bytes(bytestr:bytes, preprocessor: CLIPPreprocessor,device:str):
    """
    Convert image bytes to preprocessed torch tensor ready for model.encode_image().
    """
    image = Image.open(io.BytesIO(bytestr)).convert("RGB")
    tensor = preprocessor.image_transform(image).unsqueeze(0).to(device)
    return tensor

def metadata_for_indices(indices:List[int],metadata:List[Dict[str, Any]]):
    """
    Map list of indices returned by FAISS to metadata entries
    Returns list of dicts with (rank, image_path, caption, score placeholder).
    """
    res = []
    for rank, idx in enumerate(indices,start=1):
        if idx < 0  or idx >= len(metadata):
            # invalid index (FAISS returns -1 or empty)
            res.append({
                "rank": rank,
                "image_path": None,
                "caption": None,
                "score": None,
            })
        else:
            m = metadata[idx]
            res.append({
                "rank": rank,
                "image_path": m.get("image_path"),
                "caption": m.get("caption"),
                "score": None,
            })
    return res

# Startup / Shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Load CLIP model, FAISS index and metadata into global state.
    This runs once when the FastAPI app is initialized.
    """
    #Load configs
    clip_config = load_yaml(CLIP_CONFIG_PATH)
    faiss_config = load_yaml(FAISS_CONFIG_PATH)

    state.clip_config = clip_config
    state.faiss_config = faiss_config

    # Device
    device = clip_config.get("device","cpu")
    state.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

    # Load CLIP model
    model_name = clip_config.get("model_name","ViT-B/32")
    print(f"Loading CLIP model from {model_name} on {state.device}")
    model, clip_preproc = clip.load(model_name, device=state.device)
    model.eval()
    state.model = model

    # Use our deterministic preprocessor so its identical to the embedding script
    state.preprocess = CLIPPreprocessor(image_size=clip_config.get("image_size", 224))

    #Load FAISS index
    index_path = faiss_config.get("index_path",FAISS_INDEX_DEFAULT)
    if not os.path.exists(index_path):
        raise RuntimeError(f"Index path {index_path} does not exist")
    print(f"Loading FAISS index from {index_path} on {state.device}")
    idx = faiss.read_index(index_path)
    state.index = idx

    # If index is IVF, set nprobe from config for search-time
    if "IndexIVF" in type(idx).__name__:
        state.index_ivf = True
        nprobe = faiss_config.get("nprobe",10)
        try:
            idx.nprobe = nprobe
        except Exception as e:
            # some FAISS versions require setting vis faiss.ParameterSpace
            try:
                faiss.ParameterSpace().set_index_parameter(idx,f"nprobe={nprobe}")
            except Exception as e:
                print("Warning: could not set nprobe on index")
    # Load metadata
    metadata_path = faiss_config.get("metadata_path",METADATA_JSON_DEFAULT)
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata path {metadata_path} does not exist")
    print(f"Loading FAISS index from {metadata_path} on {state.device}")
    state.metadata = load_metadata_jsonl(metadata_path)

    print("Startup completed: model, index and metadata loaded")

# API endpoints
@app.get("/health")
async def health():
    return {"status":"ok","model_loaded":state.model is not None,"index_loaded":state.index is not None}

@app.post("/search/text")
async def search_text(q:TextQuery):
    """
    Accepts JSON {"query":"a dog on a skateboard",k:10}
    Returns top-k image results {image_path, caption, score}
    """
    if state.model is None or state.index is None:
        raise HTTPException(status_code=503, detail="Model or index not loaded")

    if not q.query or q.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query not provided. Empty query string")

    k = int(q.k or 10)
    if k <= 0 or k > 100:
        raise HTTPException(status_code=400, detail="Query k must be between 0 and 100")

    # tokenize + encode
    with torch.no_grad():
        tokens = clip.tokenize([q.query]).to(state.device)
        text_emb = state.model.encode_text(tokens)
        text_emb = text_emb.cpu().numpy().astype(np.float32)

    # normalize if the pipeline used normalization
    if state.clip_config.get('normalize_embeddings', True):
        text_emb = l2_normalize(text_emb)

    # search
    D, I = state.index.search(text_emb, k)
    I = I[0].tolist()
    D = D[0].tolist()

    results = metadata_for_indices(I, state.metadata)
    # attach scores
    for r, score in zip(results, D):
        r["score"] = float(score)

    response = {
        "query_type":"text",
        "query":q.query,
        "results": results
    }
    return JSONResponse(response)

@app.post("/search/image")
async def search_image(file:Optional[UploadFile] = File(None),payload:Optional[ImageQuery] = None):
    """
    Two modes:
    - multipart form: upload file under 'file'
    - JSON base64: POST JSON {"image":"<base64>","k":10}
    """
    if state.model is None or state.index is None:
        raise HTTPException(status_code=503, detail="Model or index not loaded")

    #Load bytes either from multipart upload or JSON base64
    image_bytes = None
    k = 10
    if file is not None:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Max {MAX_UPLOAD_BYTES} bytes")
        image_bytes = contents
    elif payload is not None:
        try:
            image_bytes = base64.b64decode(payload.image)
            k = int(payload.k or 10)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base 64 image {e}")

        if image_bytes is None or len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="No image provided. Use multipart `file` or JSON base64")

        # preprocess -> tensor
        try:
            img_tensor = prepare_image_tensor_from_bytes(image_bytes,state.preprocess,state.device)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode image {e}")

        # search
        k = int(k or 10)
        if k <= 0 or k > 100:
            raise HTTPException(status_code=400, detail="Query k must be between 0 and 100")

        D,I =state.index.search(img_tensor, k)
        I = I[0].tolist()
        D = D[0].tolist()

        results = metadata_for_indices(I, state.metadata)
        for r, score in zip(results, D):
            r["score"] = float(score)

        response = {
            "query_type":"image",
            "query":"uploaded_image",
            "results": results
        }
        return JSONResponse(response)