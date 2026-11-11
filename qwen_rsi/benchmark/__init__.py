"""Benchmark loading and grading (supervisor-only; never imported by genomes)."""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from .. import guard
from ..sandbox import run_code
from .builtin_tasks import TASKS as BUILTIN_TASKS


@dataclass
class Task:
    id: str
    split: str
    entry_point: str
    prompt: str
    checks: list[str]
    source: str = "builtin"

    def public(self) -> dict:
        """What the agent is allowed to see."""
        return {"id": self.id, "kind": "code", "entry_point": self.entry_point, "prompt": self.prompt}


def load_tasks(user_dir: Path | None = None) -> list[Task]:
    tasks = [Task(**t) for t in BUILTIN_TASKS]
    if user_dir and user_dir.is_dir():
        for p in sorted(user_dir.glob("*.json")):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for t in raw if isinstance(raw, list) else [raw]:
                    tasks.append(Task(id=t["id"], split=t.get("split", "dev"), entry_point=t["entry_point"],
                                      prompt=t["prompt"], checks=list(t["checks"]), source=p.name))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
    seen: set[str] = set()
    unique = []
    for t in tasks:
        if t.id not in seen:
            seen.add(t.id)
            unique.append(t)
    return unique


@dataclass
class GradeResult:
    task_id: str
    score: float                  # fraction of checks passed
    passed: int
    total: int
    failures: list[str] = field(default_factory=list)   # human-readable, dev split only
    error: str = ""

    @property
    def solved(self) -> bool:
        return self.total > 0 and self.passed == self.total


_HARNESS = r'''
import json as _json
_SRC = {src!r}
_CHECKS = {checks!r}
_MARK = {mark!r}
_print = print

def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False

_ns = {{"__name__": "solution"}}
_out = {{"passed": 0, "total": len(_CHECKS), "failures": [], "error": ""}}
try:
    exec(compile(_SRC, "solution.py", "exec"), _ns)
except BaseException as e:
    _out["error"] = "solution failed to load: %s: %s" % (type(e).__name__, e)
else:
    _ns["_raises"] = _raises
    for _c in _CHECKS:
        try:
            if eval(_c, _ns) is True:
                _out["passed"] += 1
            else:
                _out["failures"].append("check failed: " + _c)
        except BaseException as e:
            _out["failures"].append("%s raised %s: %s" % (_c, type(e).__name__, str(e)[:200]))
_print(_MARK + _json.dumps(_out))
'''


def grade(task: Task, solution: str, *, timeout: float = 20.0, memory_mb: int = 768) -> GradeResult:
    if not isinstance(solution, str) or not solution.strip():
        return GradeResult(task.id, 0.0, 0, len(task.checks), error="empty solution")
    g = guard.check_solution_code(solution)
    if not g.ok:
        return GradeResult(task.id, 0.0, 0, len(task.checks), error="rejected by guard: " + "; ".join(g.problems[:5]))
    mark = "@@GRADE-" + secrets.token_hex(8) + "@@"
    script = _HARNESS.format(src=solution, checks=task.checks, mark=mark)
    rr = run_code(script, timeout=timeout, memory_mb=memory_mb)
    line = next((ln for ln in rr.stdout.splitlines()[::-1] if ln.startswith(mark)), None)
    if line is None:
        why = "timed out" if rr.timed_out else f"crashed (exit {rr.returncode}): {rr.stderr[-500:]}"
        return GradeResult(task.id, 0.0, 0, len(task.checks), error=why)
    out = json.loads(line[len(mark):])
    total = max(1, out["total"])
    return GradeResult(task.id, out["passed"] / total, out["passed"], out["total"],
                       failures=out["failures"][:6], error=out["error"])
