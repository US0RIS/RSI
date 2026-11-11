"""Archive of genome variants (DGM-style open-ended archive) in SQLite.

Genome files live in ``<data>/genomes/<id>/``; scores and lineage in
``<data>/archive.db``. Genome directories are written once and never
modified afterwards, so any variant can be inspected, diffed or restored.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GENOME_FILES = ("agent.py", "system_prompt.md", "improve_prompt.md", "config.json", "NOTES.md")
MAX_GENOME_FILE_BYTES = 200_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    generation INTEGER NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,              -- created | evaluated | rejected | failed
    rationale TEXT DEFAULT '',
    smoke REAL, dev REAL, holdout REAL,
    is_champion INTEGER DEFAULT 0,
    awaiting_approval INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    seconds REAL DEFAULT 0,
    error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS evals (
    variant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    split TEXT NOT NULL,
    stage TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER, total INTEGER,
    failures TEXT, error TEXT,
    seconds REAL, llm_calls INTEGER, tokens INTEGER,
    PRIMARY KEY (variant_id, task_id)
);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL
);
"""


@dataclass
class Variant:
    id: str
    parent_id: str | None
    generation: int
    created_at: float
    status: str
    rationale: str
    smoke: float | None
    dev: float | None
    holdout: float | None
    is_champion: bool
    awaiting_approval: bool
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    seconds: float
    error: str


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.genomes = self.root / "genomes"
        self.genomes.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "archive.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------------ files
    def genome_dir(self, vid: str) -> Path:
        return self.genomes / vid

    def read_files(self, vid: str) -> dict[str, str]:
        d = self.genome_dir(vid)
        return {name: (d / name).read_text(encoding="utf-8") for name in GENOME_FILES if (d / name).exists()}

    def _next_id(self) -> str:
        row = self.db.execute("SELECT COUNT(*) FROM variants").fetchone()
        n = row[0]
        while self.genome_dir(f"g{n:04d}").exists():
            n += 1
        return f"g{n:04d}"

    def add(self, files: dict[str, str], parent_id: str | None, rationale: str = "") -> str:
        with self._lock:
            vid = self._next_id()
            d = self.genome_dir(vid)
            d.mkdir(parents=True)
            for name, text in files.items():
                if name in GENOME_FILES:
                    (d / name).write_text(text, encoding="utf-8")
            gen = 0 if parent_id is None else (self.get(parent_id).generation + 1)
            self.db.execute(
                "INSERT INTO variants (id, parent_id, generation, created_at, status, rationale) VALUES (?,?,?,?,?,?)",
                (vid, parent_id, gen, time.time(), "created", rationale[:4000]))
            self.db.commit()
            return vid

    def import_seed(self, seed_dir: Path) -> str:
        files = {p.name: p.read_text(encoding="utf-8") for p in seed_dir.iterdir() if p.name in GENOME_FILES}
        return self.add(files, None, "seed genome")

    # --------------------------------------------------------------- records
    def update(self, vid: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.db.execute(f"UPDATE variants SET {cols} WHERE id = ?", (*fields.values(), vid))
            self.db.commit()

    def add_usage(self, vid: str, llm_calls: int, prompt_tokens: int, output_tokens: int, seconds: float) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE variants SET llm_calls = llm_calls + ?, prompt_tokens = prompt_tokens + ?, "
                "output_tokens = output_tokens + ?, seconds = seconds + ? WHERE id = ?",
                (llm_calls, prompt_tokens, output_tokens, seconds, vid))
            self.db.commit()

    def record_eval(self, vid: str, task_id: str, split: str, stage: str, score: float, passed: int, total: int,
                    failures: list[str], error: str, seconds: float, llm_calls: int, tokens: int) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO evals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (vid, task_id, split, stage, score, passed, total, json.dumps(failures), error, seconds,
                 llm_calls, tokens))
            self.db.commit()

    def evals(self, vid: str, split: str | None = None) -> list[dict]:
        q = "SELECT * FROM evals WHERE variant_id = ?" + (" AND split = ?" if split else "")
        args = (vid, split) if split else (vid,)
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["failures"] = json.loads(d["failures"] or "[]")
            out.append(d)
        return out

    def log(self, kind: str, message: str) -> None:
        with self._lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), kind, message[:4000]))
            self.db.commit()

    def recent_events(self, limit: int = 200) -> list[tuple[float, str, str]]:
        with self._lock:
            rows = self.db.execute("SELECT ts, kind, message FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [tuple(r) for r in rows][::-1]

    # --------------------------------------------------------------- queries
    @staticmethod
    def _variant(row: sqlite3.Row) -> Variant:
        d = dict(row)
        d["is_champion"] = bool(d["is_champion"])
        d["awaiting_approval"] = bool(d["awaiting_approval"])
        return Variant(**d)

    def get(self, vid: str) -> Variant:
        with self._lock:
            row = self.db.execute("SELECT * FROM variants WHERE id = ?", (vid,)).fetchone()
        if row is None:
            raise KeyError(vid)
        return self._variant(row)

    def all(self) -> list[Variant]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM variants ORDER BY created_at").fetchall()
        return [self._variant(r) for r in rows]

    def evaluated(self) -> list[Variant]:
        return [v for v in self.all() if v.status == "evaluated" and v.dev is not None]

    def children(self, vid: str) -> list[Variant]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM variants WHERE parent_id = ?", (vid,)).fetchall()
        return [self._variant(r) for r in rows]

    def productivity(self, vid: str) -> float | None:
        """Fraction of this variant's children that beat it on dev: a direct
        measure of how good its *improve()* procedure is."""
        parent = self.get(vid)
        kids = [c for c in self.children(vid) if c.status in ("evaluated", "rejected", "failed")]
        if not kids or parent.dev is None:
            return None
        better = sum(1 for c in kids if c.dev is not None and c.dev > parent.dev)
        return better / len(kids)

    def champion(self) -> Variant | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM variants WHERE is_champion = 1").fetchone()
        return self._variant(row) if row else None

    def set_champion(self, vid: str) -> None:
        with self._lock:
            self.db.execute("UPDATE variants SET is_champion = 0")
            self.db.execute("UPDATE variants SET is_champion = 1, awaiting_approval = 0 WHERE id = ?", (vid,))
            self.db.commit()

    def lineage(self, vid: str) -> list[Variant]:
        chain = []
        cur: str | None = vid
        while cur:
            v = self.get(cur)
            chain.append(v)
            cur = v.parent_id
        return chain[::-1]

    def reset(self) -> None:
        with self._lock:
            self.db.executescript("DELETE FROM variants; DELETE FROM evals; DELETE FROM events;")
            self.db.commit()
            shutil.rmtree(self.genomes, ignore_errors=True)
            self.genomes.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.db.close()
