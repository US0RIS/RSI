"""Settings and on-disk locations.

All mutable state lives under the per-user data directory
(``%LOCALAPPDATA%\\QwenRSI`` on Windows), never inside the installed package,
so the immutable supervisor code and the evolving genomes stay separate.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "QwenRSI"
DEFAULT_MODEL = "qwen3.8:27b"


def data_dir() -> Path:
    override = os.environ.get("QWEN_RSI_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class Settings:
    # --- Ollama connection -------------------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    num_ctx: int = 32768            # Qwen3.8 supports far more; VRAM is the limit.
    keep_alive: str = "30m"
    request_timeout_s: float = 900.0

    # --- Chat tab ------------------------------------------------------------
    chat_think: str = "medium"      # "off" | "low" | "medium" | "high" | "max"

    # --- Evolution loop ------------------------------------------------------
    smoke_tasks: int = 4            # stage 1: quick filter
    smoke_min_frac: float = 0.5     # stage 1 must reach this fraction of parent's smoke score
    holdout_margin: float = 0.05    # child must beat champion on hold-out by this much
    require_approval: bool = True   # human must approve a new champion
    selection_lambda: float = 10.0  # DGM sigmoid sharpness
    selection_midpoint: float = 0.5
    max_children_per_gen: int = 1

    # --- Budgets enforced by the supervisor (not by the genome) --------------
    llm_calls_per_task: int = 12
    llm_calls_per_improve: int = 16
    exec_calls_per_task: int = 12
    task_timeout_s: float = 600.0
    improve_timeout_s: float = 1500.0
    exec_timeout_s: float = 10.0
    sandbox_memory_mb: int = 1024

    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------
    @classmethod
    def path(cls) -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def think_value(self, level: str | None = None):
        """Translate a UI think level into Ollama's ``think`` field."""
        level = (level or self.chat_think).lower()
        if level in ("off", "false", "none", ""):
            return False
        if level in ("on", "true"):
            return True
        return level
