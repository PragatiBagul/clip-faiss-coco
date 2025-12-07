"""
llama/inference/utils.py

Helpers to build prompts safely from retrieved captions/metadata.
Place under llama/inference/
"""

import json
import os
import textwrap
from typing import List, Dict, Any, Tuple

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt_template(name: str) -> str:
    path = os.path.join(PROMPT_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars-3] + "..."

def format_retrieved(retrieved: List[Dict[str, Any]], include_scores: bool = False) -> str:
    """
    Turn list of retrieved items into a single text block.
    Each item: {'image_id','caption','score'}
    """
    lines = []
    for i, r in enumerate(retrieved, start=1):
        cap = r.get("caption", "")
        if include_scores:
            lines.append(f"{i}. score={r.get('score', None):.4f} | {cap}")
        else:
            lines.append(f"{i}. {cap}")
    return "\n".join(lines)

def build_prompt(
    query_type: str,
    retrieved: List[Dict[str, Any]],
    template_name: str,
    user_query: str = "",
    max_chars_context: int = 3000
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a prompt by loading a template and filling placeholders.

    Returns (prompt_text, metadata)
    """
    template = load_prompt_template(template_name)
    # prepare retrieved captions
    retrieved_text = format_retrieved(retrieved, include_scores=("with_scores" in template_name or "explanation" in template_name))
    # truncate
    retrieved_text = safe_truncate(retrieved_text, max_chars_context)

    prompt = template.replace("{retrieved_captions}", retrieved_text).replace("{retrieved_captions_with_scores}", retrieved_text).replace("{user_query}", user_query)
    metadata = {
        "query_type": query_type,
        "k": len(retrieved),
        "retrieved_preview": retrieved_text[:400]
    }
    return prompt, metadata
