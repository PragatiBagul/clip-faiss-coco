# place under app/routers/
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi import Body
from typing import List, Optional
from pydantic import BaseModel

from ..schemas.generation_schemas import (
    CaptionFromImageRequest, CaptionFromImageResponse,
    ExplanationFromTextRequest, ExplanationFromTextResponse, RetrievedItem
)

# local imports
from llama.inference.inference import LlamaGenerator
from llama.inference.utils import build_prompt

router = APIRouter(prefix="/generate", tags=["generation"])

# Create a global generator instance at import-time for simplicity; in production do startup event.
GENERATOR = None

def get_generator():
    global GENERATOR
    if GENERATOR is None:
        # default base model — change to your trusted model
        GENERATOR = LlamaGenerator(base_model="meta-llama/Llama-2-7b-chat-hf", use_8bit=False, lora_adapter_path=None)
    return GENERATOR

# Mock function: replace with your actual CLIP+FAISS search call
def run_clip_faiss_search_by_image(image_bytes: bytes, k: int) -> List[dict]:
    """
    Return list of dicts: {'rank','image_id','image_path','caption','score'}
    Replace this stub to call your search router or internal function.
    """
    # Simple stub: return empty list
    return []

def run_clip_faiss_search_by_text(query: str, k: int) -> List[dict]:
    return []

@router.post("/caption_from_image", response_model=CaptionFromImageResponse)
async def caption_from_image(request: CaptionFromImageRequest = Body(...), file: Optional[UploadFile] = File(None)):
    """
    Accept either image upload (multipart) or an existing image_id.
    Flow:
      1. Get image bytes or image_id
      2. Run CLIP->FAISS search (top-k)
      3. Build prompt (caption_prompt.txt)
      4. Call LLaMA generate
    """
    gen = get_generator()

    if file:
        image_bytes = await file.read()
        retrieved = run_clip_faiss_search_by_image(image_bytes, request.k)
        image_id = None
    elif request.image_id:
        # if the v1 code supports search by id, call it
        retrieved = run_clip_faiss_search_by_text(request.image_id, request.k)
        image_id = request.image_id
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded file or an image_id")

    # Build prompt
    prompt, metadata = build_prompt("image", retrieved, "caption_prompt.txt", user_query="", max_chars_context=3000)

    # Generate
    outputs = gen.generate_json(prompt, max_new_tokens=request.max_tokens or 128, temperature=request.temperature or 0.7, num_return_sequences=1)
    result_output = outputs["parsed"] if outputs["success"] else {"raw": outputs["raw"]}

    # Build response retrieved items into pydantic form
    retrieved_resp = []
    for i, r in enumerate(retrieved, start=1):
        retrieved_resp.append(RetrievedItem(rank=i, image_id=r.get("image_id"), image_path=r.get("image_path"), caption=r.get("caption"), score=r.get("score")))

    return CaptionFromImageResponse(
        query_type="image",
        image_id=image_id,
        retrieved=retrieved_resp,
        prompt=prompt,
        output=result_output
    )

@router.post("/explanation_from_text_query", response_model=ExplanationFromTextResponse)
async def explanation_from_text(req: ExplanationFromTextRequest = Body(...)):
    gen = get_generator()
    retrieved = run_clip_faiss_search_by_text(req.query, req.k)

    # build prompt
    prompt, meta = build_prompt("text", retrieved, "explanation_prompt.txt", user_query=req.query, max_chars_context=3000)

    outputs = gen.generate_from_prompt(prompt, max_new_tokens=256, temperature=0.7, num_return_sequences=1)
    raw = outputs[0] if outputs else ""

    retrieved_resp = []
    for i, r in enumerate(retrieved, start=1):
        retrieved_resp.append(RetrievedItem(rank=i, image_id=r.get("image_id"), image_path=r.get("image_path"), caption=r.get("caption"), score=r.get("score")))

    return ExplanationFromTextResponse(
        query_type="text",
        query=req.query,
        retrieved=retrieved_resp,
        prompt=prompt,
        output={"explanation_raw": raw}
    )
