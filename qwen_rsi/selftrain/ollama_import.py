"""Turn a trained adapter into an Ollama model and check that it actually works.

Path 1 (preferred): ``FROM <base>`` + ``ADAPTER <adapter dir>``. Inherits the base
model's template, renderer, parser, parameters and vision projector; the adapter
is small.
Path 2 (fallback): merged 16-bit safetensors imported with ``--quantize q4_K_M``,
with the base model's renderer/parser/template/parameters copied over.

Either way the new model must pass a probe (answers correctly, thinking is
separated from the answer) before it is evaluated.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..config import Settings
from ..ollama_client import OllamaClient, OllamaError


class ImportError_(RuntimeError):
    pass


class Importer:
    def __init__(self, settings: Settings, client: OllamaClient):
        self.s = settings
        self.client = client

    def _run(self, args: list[str], cwd: Path, timeout: float = 4 * 3600) -> tuple[int, str]:
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000
        try:
            p = subprocess.run([self.s.ollama_exe, *args], cwd=str(cwd), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout, **kw)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 1, str(e)
        return p.returncode, (p.stdout + p.stderr)[-3000:]

    def create_from_adapter(self, tag: str, base: str, adapter_dir: Path, workdir: Path) -> None:
        mf = workdir / "Modelfile.adapter"
        mf.write_text(f"FROM {base}\nADAPTER {adapter_dir.resolve()}\n", encoding="utf-8")
        rc, out = self._run(["create", tag, "-f", str(mf)], workdir)
        if rc != 0:
            raise ImportError_(f"ollama create (adapter) failed: {out[-800:]}")

    def create_from_merged(self, tag: str, base: str, merged_dir: Path, workdir: Path) -> None:
        info = self.client.show(base)
        lines = [f"FROM {merged_dir.resolve()}"]
        if info.get("renderer"):
            lines.append(f"RENDERER {info['renderer']}")
        if info.get("parser"):
            lines.append(f"PARSER {info['parser']}")
        if not info.get("renderer") and info.get("template"):
            lines.append('TEMPLATE """' + info["template"] + '"""')
        for raw in (info.get("parameters") or "").splitlines():
            parts = raw.split(None, 1)
            if len(parts) == 2:
                lines.append(f"PARAMETER {parts[0]} {parts[1]}")
        mf = workdir / "Modelfile.merged"
        mf.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rc, out = self._run(["create", tag, "--quantize", "q4_K_M", "-f", str(mf)], workdir)
        if rc != 0:
            raise ImportError_(f"ollama create (merged) failed: {out[-800:]}")

    def remove(self, tag: str, workdir: Path) -> None:
        self._run(["rm", tag], workdir, timeout=300)

    def probe(self, tag: str) -> tuple[bool, str]:
        """A broken import usually shows up as gibberish, a wrong answer, or raw <think> tags."""
        try:
            r = self.client.chat(tag, [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}],
                                 think="low", options={"num_ctx": 4096, "num_predict": 2048, "temperature": 0})
        except OllamaError as e:
            return False, f"chat failed: {e}"
        if "391" not in r.content:
            return False, f"wrong answer to probe: {r.content[:200]!r}"
        if "<think>" in r.content or "</think>" in r.content:
            return False, "thinking leaked into the answer (template/parser mismatch)"
        return True, "ok"
