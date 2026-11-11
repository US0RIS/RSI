import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def fake_ollama():
    import fake_ollama as fo
    srv, state, url = fo.start()
    yield state, url
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_RSI_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
