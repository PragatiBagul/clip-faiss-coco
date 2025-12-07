# llama/inference/inference.py
"""
LLaMA inference wrapper with optional LoRA (PEFT) adapter loading and deterministic seeding.

Public API:
- load_model(config_path="llama/configs/llama.yaml", adapter_path=None)
- generate_from_prompt(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9, num_return_sequences=1, seed=None)

Notes:
- If seed is provided, we set torch, numpy, and python random seeds for reproducible generation (best-effort).
- Generation uses transformers' generation API and returns the completion (attempts to strip prompt).
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import yaml
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_model = None
_tokenizer = None
_config = None
_peft_applied = False


def load_config(path: str = "llama/configs/llama.yaml") -> Dict[str, Any]:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config_path: str = "llama/configs/llama.yaml", adapter_path: Optional[str] = None):
    global _model, _tokenizer, _config, _peft_applied

    if _model is not None and _tokenizer is not None:
        logger.info("Model already loaded.")
        return _model, _tokenizer

    _config = load_config(config_path)
    model_name = _config.get("model_name")
    device_pref = _config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_8bit = bool(_config.get("use_8bit", False))

    logger.info(f"Loading tokenizer for {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    logger.info(f"Loading model {model_name} use_8bit={use_8bit}")
    if use_8bit:
        # 8-bit load
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # FP16/FP32 load depending on device
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device_pref != "cpu" else None,
            torch_dtype=torch.float16 if device_pref != "cpu" else torch.float32,
            trust_remote_code=True
        )

    if adapter_path:
        logger.info(f"Attempting to attach PEFT adapter from {adapter_path}")
        try:
            _model = PeftModel.from_pretrained(_model, adapter_path)
            _peft_applied = True
        except Exception as e:
            logger.warning(f"Failed to attach adapter: {e}")
            _peft_applied = False

    logger.info("Model loaded.")
    return _model, _tokenizer


def generate_from_prompt(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    seed: Optional[int] = None,
    config_path: str = "llama/configs/llama.yaml"
) -> str:
    """
    Synchronous generation. Returns the generated text (completion).
    If seed is provided, generate deterministically (best-effort).
    """
    global _model, _tokenizer, _config

    if _model is None or _tokenizer is None:
        load_model(config_path=config_path, adapter_path=_config.get("adapter_path") if _config else None)

    cfg = _config or load_config(config_path)
    # apply seed if given
    _set_seed(seed or cfg.get("seed"))

    gen_cfg = GenerationConfig(
        temperature=temperature or cfg.get("temperature", 0.7),
        top_p=top_p or cfg.get("top_p", 0.9),
        max_new_tokens=max_new_tokens or cfg.get("max_length", 256),
        do_sample=True,
        num_return_sequences=num_return_sequences
    )

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True).to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(**inputs, generation_config=gen_cfg)
    texts = [_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # strip prompt where possible
    def strip_prompt(full_text: str, prompt_text: str) -> str:
        if full_text.startswith(prompt_text):
            return full_text[len(prompt_text):].strip()
        return full_text.strip()

    results = [strip_prompt(t, prompt) for t in texts]
    return results[0] if len(results) == 1 else results
