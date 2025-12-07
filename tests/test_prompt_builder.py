# simple unit test for build_prompt
from llama.inference.utils import build_prompt

def test_build_prompt_truncation():
    retrieved = [{"caption": "a " + "very "*200 + "long caption", "score":0.9} for _ in range(5)]
    prompt, meta = build_prompt("image", retrieved, "caption_prompt.txt", user_query="", max_chars_context=200)
    assert isinstance(prompt, str)
    assert "Title" in prompt or "Context" in prompt  # template text inserted
    assert meta["k"] == 5
