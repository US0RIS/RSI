"""Background threads so the UI never blocks on Ollama or the sandbox."""
from __future__ import annotations

import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from ..archive import Archive
from ..config import Settings
from ..evolve import Evolver, InfraError
from ..ollama_client import OllamaClient, OllamaError


class EvolutionWorker(QThread):
    event = Signal(str, str)
    stopped = Signal(str)

    def __init__(self, settings: Settings, archive: Archive, generations: int | None):
        super().__init__()
        self.generations = generations
        self.evolver = Evolver(settings, archive=archive, on_event=lambda k, m: self.event.emit(k, m))

    def run(self) -> None:
        reason = "finished"
        try:
            self.evolver.run(self.generations)
            if self.evolver.stop.is_set():
                reason = "stopped by user"
        except InfraError as e:
            reason = f"Ollama problem: {e}"
        except Exception as e:  # surface, never crash the UI thread
            reason = f"{type(e).__name__}: {e}"
            self.event.emit("error", traceback.format_exc()[-2000:])
        self.stopped.emit(reason)

    def request_stop(self) -> None:
        self.evolver.stop.set()


class ChatWorker(QThread):
    token = Signal(str, str)        # kind, text
    finished_ok = Signal(str, str)  # thinking, content
    failed = Signal(str)

    def __init__(self, settings: Settings, messages: list[dict], think, champion_archive: Archive | None = None):
        super().__init__()
        self.s = settings
        self.messages = messages
        self.think = think
        self.champion_archive = champion_archive
        self.stop_event = threading.Event()
        self._evolver: Evolver | None = None

    def run(self) -> None:
        try:
            if self.champion_archive is not None:
                self._evolver = Evolver(self.s, archive=self.champion_archive)
                self._evolver.stop = self.stop_event
                content = self._evolver.chat_with_champion(self.messages)
                self.token.emit("content", content)
                self.finished_ok.emit("", content)
                return
            client = OllamaClient(self.s.ollama_url, self.s.request_timeout_s)
            r = client.chat(self.s.model, self.messages, think=self.think,
                            options={"num_ctx": self.s.num_ctx}, keep_alive=self.s.keep_alive,
                            stop=self.stop_event, on_token=lambda k, t: self.token.emit(k, t))
            self.finished_ok.emit(r.thinking, r.content)
        except (OllamaError, OSError, RuntimeError) as e:
            self.failed.emit(str(e))

    def request_stop(self) -> None:
        self.stop_event.set()


class PullWorker(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            OllamaClient(self.s.ollama_url, 3600).pull(
                self.s.model, progress=lambda st, c, t: self.progress.emit(st, c, t), stop=self.stop_event)
            self.done.emit("" if not self.stop_event.is_set() else "cancelled")
        except (OllamaError, OSError) as e:
            self.done.emit(str(e))


class StatusPoller(QThread):
    status = Signal(bool, str)

    def __init__(self, settings: Settings, interval_s: float = 15.0):
        super().__init__()
        self.s = settings
        self.interval_s = interval_s
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                c = OllamaClient(self.s.ollama_url, 5)
                ver = c.version()
                if not c.has_model(self.s.model):
                    self.status.emit(False, f"Ollama {ver} · {self.s.model} not pulled")
                else:
                    loaded = [m for m in c.running() if m.get("name") == self.s.model]
                    if loaded:
                        vram = loaded[0].get("size_vram", 0) / 1e9
                        state = f"loaded ({vram:.1f} GB VRAM)"
                    else:
                        state = "ready (not loaded)"
                    self.status.emit(True, f"Ollama {ver} · {self.s.model} {state}")
            except (OllamaError, OSError) as e:
                self.status.emit(False, str(e)[:120])
            end = time.monotonic() + self.interval_s
            while time.monotonic() < end and not self._halt.is_set():
                self._halt.wait(0.5)

    def halt(self) -> None:
        self._halt.set()
