import pytest

from qwen_rsi.ollama_client import OllamaClient, OllamaError


def test_client_endpoints(fake_ollama):
    state, url = fake_ollama
    c = OllamaClient(url)
    assert c.version() == "0.99.0-fake"
    assert c.has_model("qwen3.8:27b")
    assert "thinking" in c.capabilities("qwen3.8:27b")
    assert c.context_length("qwen3.8:27b") == 262144
    toks = []
    r = c.chat("qwen3.8:27b", [{"role": "user", "content": "hi"}], think="high",
               on_token=lambda k, t: toks.append(k))
    assert r.content == "Hello from the fake model." and r.thinking == "Let me think."
    assert "thinking" in toks and "content" in toks
    assert state.requests[-1]["think"] == "high"
    assert r.output_tokens == 50


def test_client_errors(fake_ollama):
    _, url = fake_ollama
    with pytest.raises(OllamaError, match="not found"):
        OllamaClient(url).chat("nope:1b", [{"role": "user", "content": "x"}])
    with pytest.raises(OllamaError, match="Cannot reach"):
        OllamaClient("http://127.0.0.1:9", timeout=2).version()
