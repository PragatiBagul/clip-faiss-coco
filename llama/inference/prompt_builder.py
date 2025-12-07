# llama/inference/prompt_builder.py
"""
Prompt builder for LLaMA inference.

Key functions:
- build_caption_prompt(retrieved: list[dict], max_items=10)
- build_explanation_prompt(query: str, retrieved: list[dict], max_items=10)

`retrieved` is expected to be a list of dicts each containing:
  {"rank": int, "image_path": str, "caption": str, "score": float}
"""

from typing import List, Dict
import json
import os

PROMPTS_DIR = os.path.join("llama", "prompts")


def _truncate_evidence(evidence_lines: List[str], max_items: int):
    """Return top max_items from evidence list (already sorted by score)."""
    return evidence_lines[:max_items]


def build_caption_prompt(retrieved: List[Dict], max_items: int = 8) -> str:
    """
    Build prompt for caption generation using retrieved captions.
    Returns a string prompt (ready to pass to LLaMA).
    """
    # Extract captions and make evidence block
    captions = [r.get("caption", "") for r in retrieved if r.get("caption")]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for c in captions:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    evidence = _truncate_evidence(deduped, max_items)
    evidence_block = "\n".join(f"- {c}" for c in evidence)

    # load template
    template_path = os.path.join(PROMPTS_DIR, "caption_prompt.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(evidence=evidence_block)
    return prompt


def build_explanation_prompt(query: str, retrieved: List[Dict], max_items: int = 8) -> str:
    """
    Build prompt for explanation generation given a text query and retrieved captions.
    """
    # Build evidence with scores
    lines = []
    for r in retrieved[:max_items]:
        score = r.get("score", None)
        if score is not None:
            lines.append(f"- {r.get('caption','')} (score={score:.3f})")
        else:
            lines.append(f"- {r.get('caption','')}")
    evidence_block = "\n".join(lines)

    template_path = os.path.join(PROMPTS_DIR, "explanation_prompt.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(user_query=query, evidence_with_scores=evidence_block)
    return prompt
