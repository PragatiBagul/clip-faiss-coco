"""
llama/inference/inference.py

Lightweight inference wrapper for HF causal LMs (LLaMA-like) with optional LoRA/PEFT adapter support.

Usage:
    from llama.inference.inference import LlamaGenerator

    gen = LlamaGenerator(
        base_model="meta-llama/Llama-2-7b-chat-hf",
        lora_adapter_path=None,
        use_8bit=True,
        device=None
    )
    out = gen.generate_from_prompt("Write a JSON caption for an image: ...", max_new_tokens=128)
"""

from typing import Optional, Dict, Any, List
import torch
import json
import logging
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

try:
    from peft import PeftModel
    _has_peft = True
except Exception:
    _has_peft = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LlamaGenerator:
    def __init__(
        self,
        base_model: str,
        lora_adapter_path: Optional[str] = None,
        use_8bit: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.base_model = base_model
        self.lora_adapter_path = lora_adapter_path
        self.use_8bit = use_8bit
        self.trust_remote_code = trust_remote_code

        # auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model {base_model} on {self.device} (8bit={use_8bit})")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.use_8bit:
            model_kwargs["load_in_8bit"] = True

        try:
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", **model_kwargs)
            else:
                # cpu
                self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": "cpu"}, **model_kwargs)
        except Exception as e:
            logger.warning(f"Model load with device_map failed: {e}. Falling back to simple load.")
            self.model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True, **({"trust_remote_code": self.trust_remote_code}))

        # attach adapter if present
        if lora_adapter_path:
            if not _has_peft:
                raise ImportError("PEFT not installed. Install peft to load LoRA adapters.")
            logger.info(f"Loading PEFT adapter from {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path, device_map="auto" if self.device == "cuda" else {"": "cpu"})

        self.model.eval()

    def _set_seed(self, seed: Optional[int]):
        if seed is None:
            return
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        seed: Optional[int] = None,
        return_full_text: bool = False,
        **generate_kwargs
    ) -> List[str]:
        self._set_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = []
        for i in range(num_return_sequences):
            seq = outputs[i] if num_return_sequences > 1 else outputs
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            if not return_full_text and text.startswith(prompt):
                text = text[len(prompt):].strip()
            decoded.append(text)
        return decoded

    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        outs = self.generate_from_prompt(prompt, **kwargs)
        raw = outs[0].strip()
        try:
            parsed = json.loads(raw)
            return {"success": True, "parsed": parsed, "raw": raw}
        except Exception:
            return {"success": False, "parsed": None, "raw": raw}

    def close(self):
        try:
            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass
