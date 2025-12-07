# llama/inference/context_builder.py
"""
Context Builder for retrieval → prompt pipeline.

Responsibilities:
- Accept CLIP results (list[dict] with keys: "image_path", "caption", "score", optional "image_id", "width", "height")
- Deduplicate captions (preserve highest-score occurrence)
- Sort by score (descending)
- Truncate to max items OR to a maximum token budget (if tokenizer provided)
- Optionally attach a minimal visual descriptor per item (width x height or placeholder)
- Return:
    {
      "items": [ { "rank": int, "caption": str, "score": float, "image_path": str, "visual": str } ... ],
      "skipped": int,            # number of items skipped (due to missing caption etc.)
      "truncated": int           # number removed to meet limits
    }

Design notes:
- This module is intentionally dependency-light. Tokenization for token-budget truncation is optional:
    - If `tokenizer` (HuggingFace tokenizer) is provided and `max_tokens` supplied,
      the function will estimate token counts per caption using tokenizer.encode().
    - Otherwise, we fall back to max_items truncation.
- Use this module before building prompts to keep prompt length bounded.
"""

from typing import List, Dict, Optional, Any
import os

def _safe_get_caption(rec: Dict[str, Any]) -> Optional[str]:
    c = rec.get("caption")
    if c is None:
        c = rec.get("text")  # fallback key
    if c is None:
        return None
    c = str(c).strip()
    return c if c != "" else None


def attach_visual_descriptor(rec: Dict[str, Any]) -> str:
    """
    Create a tiny visual descriptor if width/height are present; else return empty string.
    Example: "800x600"
    """
    w = rec.get("width")
    h = rec.get("height")
    if w and h:
        return f"{w}x{h}"
    # fallback: maybe file exists and we can read image size (optional, lightweight)
    fp = rec.get("image_path")
    if fp and os.path.exists(fp):
        try:
            from PIL import Image
            with Image.open(fp) as im:
                return f"{im.width}x{im.height}"
        except Exception:
            return ""
    return ""


def build_context_from_clip_results(
    retrieved: List[Dict[str, Any]],
    max_items: int = 8,
    tokenizer = None,
    max_tokens: Optional[int] = None,
    include_visual: bool = False
) -> Dict[str, Any]:
    """
    Build a sanitized, truncated context from CLIP results.

    Args:
        retrieved: list of dicts with keys: "image_path", "caption", "score", optional "image_id", "width", "height".
                   Expected to be in arbitrary order (we sort inside).
        max_items: fallback cap on the number of items to return.
        tokenizer: optional HF tokenizer (for token-counting based truncation).
        max_tokens: if provided and tokenizer available, ensures total tokens (approx) <= max_tokens.
        include_visual: if True, attach 'visual' descriptor string per item.

    Returns:
        {
          "items": [ { "rank": int, "caption": str, "score": float, "image_path": str, "visual": str }, ... ],
          "skipped": int,
          "truncated": int
        }
    """
    if not isinstance(retrieved, list):
        raise ValueError("retrieved must be a list of dicts")

    # 1) Normalize & filter out records without captions
    normalized = []
    skipped = 0
    for rec in retrieved:
        caption = _safe_get_caption(rec)
        if caption is None:
            skipped += 1
            continue
        score = rec.get("score", 0.0)
        image_path = rec.get("image_path") or rec.get("file_path") or rec.get("path") or None
        normalized.append({
            "caption": caption,
            "score": float(score) if score is not None else 0.0,
            "image_path": image_path,
            "raw": rec
        })

    if len(normalized) == 0:
        return {"items": [], "skipped": skipped, "truncated": 0}

    # 2) Deduplicate captions (keep highest-score instance)
    seen = {}
    for rec in normalized:
        c = rec["caption"]
        if c not in seen or rec["score"] > seen[c]["score"]:
            seen[c] = rec
    deduped = list(seen.values())

    # 3) Sort by score descending
    deduped.sort(key=lambda x: x["score"], reverse=True)

    # 4) Truncate either by tokens (if tokenizer+max_tokens) or by max_items
    truncated = 0
    final_items = []

    if tokenizer is not None and max_tokens is not None:
        # estimate tokens per caption using tokenizer
        total_tokens = 0
        for rec in deduped:
            # tokenizer may be fast; we call encode (no special settings)
            try:
                toks = tokenizer.encode(rec["caption"], add_special_tokens=False)
                toklen = len(toks)
            except Exception:
                # fallback heuristic: words -> tokens
                toklen = max(1, len(rec["caption"].split()))
            if total_tokens + toklen > max_tokens and len(final_items) >= 1:
                # stop if adding this caption would exceed budget
                break
            total_tokens += toklen
            final_items.append(rec)
        truncated = len(deduped) - len(final_items)
    else:
        final_items = deduped[:max_items]
        truncated = max(0, len(deduped) - len(final_items))

    # 5) Format items with rank and optional visual descriptor
    out_items = []
    for i, rec in enumerate(final_items, start=1):
        visual = ""
        if include_visual:
            visual = attach_visual_descriptor(rec["raw"])
        out_items.append({
            "rank": i,
            "caption": rec["caption"],
            "score": rec["score"],
            "image_path": rec["image_path"],
            "visual": visual
        })

    return {"items": out_items, "skipped": skipped, "truncated": truncated}
