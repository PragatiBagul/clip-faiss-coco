# tests/test_prompt_builder.py
import pytest
from llama.inference.context_builder import build_context_from_clip_results

class DummyTokenizer:
    # a very small tokenizer-like object for tests: tokenizes by whitespace
    def encode(self, text, add_special_tokens=False):
        return text.split()

def make_item(caption, score, path=None, width=None, height=None):
    rec = {"caption": caption, "score": score}
    if path:
        rec["image_path"] = path
    if width:
        rec["width"] = width
    if height:
        rec["height"] = height
    return rec

def test_dedup_and_sort_and_truncate_by_count():
    retrieved = [
        make_item("a cat on a mat", 0.5),
        make_item("a dog in a park", 0.9),
        make_item("a cat on a mat", 0.8),  # duplicate caption, higher score should win
        make_item("a bird flying", 0.4),
        make_item("a car on road", 0.3),
    ]
    out = build_context_from_clip_results(retrieved, max_items=3, tokenizer=None)
    items = out["items"]
    captions = [it["caption"] for it in items]
    assert len(items) == 3
    # highest scores first: "a dog in a park", "a cat on a mat" (0.8), "a bird flying"
    assert captions[0] == "a dog in a park"
    assert captions[1] == "a cat on a mat"
    assert captions[2] == "a bird flying"
    assert out["truncated"] == 2  # 5 deduped -> 3 returned

def test_truncate_by_tokens_with_tokenizer():
    retrieved = [
        make_item("short", 0.9),
        make_item("this is a slightly longer caption", 0.8),
        make_item("an extremely long caption that will likely overflow the budget if tokens counted", 0.7),
        make_item("another short", 0.6)
    ]
    # max tokens set so that only first two captions fit according to DummyTokenizer
    tok = DummyTokenizer()
    out = build_context_from_clip_results(retrieved, max_items=10, tokenizer=tok, max_tokens=6)
    items = out["items"]
    # "short" -> 1 token, "this is a slightly longer caption" -> 6 tokens -> might fill budget
    assert len(items) >= 1
    assert out["truncated"] >= 1

def test_include_visual_descriptor():
    # use a fake path that doesn't exist; should still produce 'visual' key but possibly empty
    retrieved = [
        {"caption": "img one", "score": 0.9, "image_path": "/tmp/nonexistent.jpg", "width": 800, "height": 600},
        {"caption": "img two", "score": 0.8, "image_path": "/tmp/nonexistent2.jpg"}
    ]
    out = build_context_from_clip_results(retrieved, include_visual=True)
    items = out["items"]
    assert "visual" in items[0]
    # first item had width/height
    assert items[0]["visual"] == "800x600"
