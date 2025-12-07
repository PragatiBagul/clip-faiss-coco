# app/schemas/generation_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any


class CaptionFromImageRequest(BaseModel):
    # Either file upload or image_id can be used. File handled in router as UploadFile.
    image_id: Optional[int] = None
    k: Optional[int] = Field(default=8, ge=1, le=100)
    max_new_tokens: Optional[int] = Field(default=128, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    seed: Optional[int] = None


class ExplanationRequest(BaseModel):
    query: str
    k: Optional[int] = Field(default=8, ge=1, le=100)
    mode: Optional[str] = Field(default="why")  # "why" or "commonality"
    max_new_tokens: Optional[int] = Field(default=128, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    seed: Optional[int] = None


class RetrievedItem(BaseModel):
    rank: int
    image_path: Optional[str]
    caption: Optional[str]
    score: Optional[float]
    visual: Optional[str] = None


class CaptionFromImageResponse(BaseModel):
    query_type: str
    image_id: Optional[int]
    retrieved: List[RetrievedItem]
    prompt: str
    output: Any  # structured JSON or raw text


class ExplanationResponse(BaseModel):
    query_type: str
    query: str
    retrieved: List[RetrievedItem]
    prompt: str
    output: Any
