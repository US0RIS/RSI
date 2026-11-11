"""Runs train_lora.py in the separate trainer environment and streams its progress."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from ..config import Settings

TRAIN_SCRIPT = Path(__file__).with_name("train_lora.py")


class TrainerError(RuntimeError):
    pass


class TrainerBridge:
    def __init__(self, settings: Settings, script: Path | None = None):
        self.s = settings
        self.script = Path(script or TRAIN_SCRIPT)

    def command(self, *args: str) -> list[str]:
        return [self.s.trainer_python_path(), "-u", str(self.script), *args]

    def _run(self, args: list[str], *, timeout_s: float, on_event: Callable[[dict], None],
             stop: threading.Event | None) -> tuple[int, list[dict], list[str]]:
        py = self.s.trainer_python_path()
        if not Path(py).exists():
            raise TrainerError(f"trainer Python not found at {py}. Run scripts\\setup_trainer.ps1 or set "
                               f"'Trainer Python' in Settings.")
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(self.command(*args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env, **kw)
        events: list[dict] = []
        tail: list[str] = []
        start = time.monotonic()
        killed = threading.Event()

        def watchdog() -> None:
            while proc.poll() is None:
                if time.monotonic() - start > timeout_s or (stop is not None and stop.is_set()):
                    killed.set()
                    proc.kill()
                    return
                time.sleep(0.5)
        threading.Thread(target=watchdog, daemon=True).start()
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("QRSI "):
                try:
                    ev = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                events.append(ev)
                on_event(ev)
            elif line.strip():
                tail.append(line)
                del tail[:-200]
        rc = proc.wait()
        if killed.is_set():
            raise TrainerError("trainer stopped" if stop is not None and stop.is_set() else "trainer timed out")
        return rc, events, tail

    def check(self) -> tuple[bool, dict]:
        try:
            rc, events, tail = self._run(["check"], timeout_s=900, on_event=lambda e: None, stop=None)
        except (TrainerError, OSError) as e:
            return False, {"error": str(e)}
        info = next((e for e in reversed(events) if e.get("event") == "check"), None)
        if info is None:
            return False, {"error": "no response from trainer", "output": "\n".join(tail[-20:])}
        return bool(info.get("ok")) and rc == 0, info

    def run(self, mode: str, job: dict, job_dir: Path, *, on_event: Callable[[dict], None],
            stop: threading.Event | None, timeout_s: float) -> dict:
        job_dir.mkdir(parents=True, exist_ok=True)
        job = dict(job, out_dir=str(job_dir))
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        rc, events, tail = self._run([mode, str(job_path)], timeout_s=timeout_s, on_event=on_event, stop=stop)
        (job_dir / f"{mode}.log").write_text("\n".join(tail), encoding="utf-8")
        err = next((e for e in reversed(events) if e.get("event") == "error"), None)
        if rc != 0 or err:
            msg = err.get("message") if err else f"exit code {rc}"
            raise TrainerError(f"{mode} failed: {msg}\n" + "\n".join(tail[-15:]))
        res_path = job_dir / "result.json"
        if not res_path.exists():
            raise TrainerError(f"{mode} produced no result.json")
        return json.loads(res_path.read_text(encoding="utf-8"))
