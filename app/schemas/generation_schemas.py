# place under app/schemas/
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class RetrievedItem(BaseModel):
    rank: int
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None
    score: Optional[float] = None

class CaptionFromImageRequest(BaseModel):
    image_id: Optional[str] = None
    k: int = Field(10, ge=1, le=100)
    max_tokens: Optional[int] = Field(128, ge=16, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)

class CaptionFromImageResponse(BaseModel):
    query_type: str
    image_id: Optional[str]
    retrieved: List[RetrievedItem]
    prompt: Optional[str]
    output: Dict[str, Any]

class ExplanationFromTextRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=100)
    mode: str = Field("why", description="why or commonality")

class ExplanationFromTextResponse(BaseModel):
    query_type: str
    query: str
    retrieved: List[RetrievedItem]
    prompt: Optional[str]
    output: Dict[str, Any]
