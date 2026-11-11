from qwen_rsi import guard
from qwen_rsi.sandbox import run_code


def test_genome_guard_blocks_escapes():
    assert guard.check_genome_code("import json\nx = 1\n")
    for bad in ["import os", "import subprocess", "from socket import socket", "open('x')",
                "eval('1')", "().__class__.__bases__", "getattr(llm, 'x')", "llm._send({})",
                "__import__('os')", "from . import x"]:
        assert not guard.check_genome_code(bad), bad


def test_solution_guard():
    assert guard.check_solution_code("from collections import Counter\nimport heapq\n")
    assert not guard.check_solution_code("import os\n")
    assert not guard.check_solution_code("import shutil\n")
    assert not guard.check_solution_code("def f(:\n")


def test_run_code_ok_and_timeout():
    r = run_code("print(input()[::-1])", stdin="abc", timeout=10)
    assert r.returncode == 0 and r.stdout.strip() == "cba"
    r = run_code("while True: pass", timeout=1)
    assert r.timed_out


def test_run_code_memory_limit():
    r = run_code("x = bytearray(2_000_000_000)\nprint('allocated')", timeout=10, memory_mb=256)
    assert "allocated" not in r.stdout


def test_run_code_scrubbed_env(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    r = run_code("import os\nprint(os.environ.get('SECRET_TOKEN'))", timeout=10)
    assert r.stdout.strip() == "None"
