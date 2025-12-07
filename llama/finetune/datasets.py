# place under llama/finetune/
import json
from typing import Dict, Any, List
from torch.utils.data import Dataset

class LoraJsonlDataset(Dataset):
    """
    Expects jsonl lines with keys: context, caption, expanded, why (optional)
    We will build input-target pairs by concatenation of context + instruction.
    """
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        context = item.get("context", "")
        caption = item.get("caption", "")
        # instruction -> we ask model to generate JSON structure
        prompt = f"Context:\n{context}\n\nInstruction: Generate a JSON object with Title, ShortCaption, DetailedCaption, Tags.\nOutput:"
        full = prompt + " " + caption
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # label strategy: shift input_ids for causal LM - handled in collator/training loop
        }
