import importlib.util

from qwen_rsi import guard
from qwen_rsi.evolve import SEED_DIR


def load_seed():
    spec = importlib.util.spec_from_file_location("seed_agent", SEED_DIR / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_passes_guard():
    assert guard.check_genome_code((SEED_DIR / "agent.py").read_text())


def test_apply_edits():
    a = load_seed()
    files = {"agent.py": "def solve(p, l, t):\n    return 1\n\ndef improve(p, l, t):\n    pass\n",
             "config.json": "{}"}
    new, errs = a.apply_edits(files, [{"file": "agent.py", "search": "return 1", "replace": "return 2"}])
    assert not errs and "return 2" in new["agent.py"]
    _, errs = a.apply_edits(files, [{"file": "agent.py", "search": "nope", "replace": "x"}])
    assert errs
    _, errs = a.apply_edits(files, [{"file": "agent.py", "search": "return 1", "replace": "return ("}])
    assert any("parses" in e for e in errs)
    new, errs = a.apply_edits(files, [{"file": "NOTES.md", "content": "hi"}])
    assert not errs and new["NOTES.md"] == "hi"
    _, errs = a.apply_edits(files, [{"file": "../evil.py", "content": "x"}])
    assert errs


def test_extract_helpers():
    a = load_seed()
    assert a.extract_code("x\n```python\ndef f():\n  pass\n```\n", "f") == "def f():\n  pass"
    assert a.extract_json('blah {"a": 1} blah') == {"a": 1}
