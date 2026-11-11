"""Self-generated training tasks, verified before use.

The model writes problems with a reference solution and checks. A problem is
kept only if:
* the reference solution passes the guard and ALL of its own checks in the sandbox,
* a do-nothing stub fails most of the checks (the checks actually test something),
* it does not resemble any benchmark task (dev, hold-out, audit, user tasks), so
  training can't leak the evaluation sets,
* it is not a near-duplicate of a task already in the pool.
"""
from __future__ import annotations

import difflib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..benchmark import Task, grade
from ..ollama_client import OllamaClient, OllamaError

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry_point": {"type": "string"},
                    "prompt": {"type": "string"},
                    "reference_solution": {"type": "string"},
                    "checks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entry_point", "prompt", "reference_solution", "checks"],
            },
        }
    },
    "required": ["tasks"],
}

GEN_PROMPT = """Create {n} new, original Python programming problems for training a coding model.

Topic: {topic}
Difficulty: {difficulty}

For each problem give:
- "entry_point": the function name (snake_case),
- "prompt": a precise specification starting with "Write `name(args) -> type`" that fully determines the
  output for every input, including edge cases and which exceptions to raise,
- "reference_solution": a correct, self-contained module using only the Python standard library,
- "checks": 6 to 12 Python boolean expressions that call the function, e.g. "name([1, 2]) == 3".
  Cover edge cases. To assert an exception use "_raises(ValueError, name, arg1, arg2)".

Rules: no file, network or randomness; deterministic results; no well-known textbook problem copied verbatim.
Do not reuse these function names: {avoid}

Return JSON: {{"tasks": [...]}}"""


@dataclass
class GenTask:
    id: str
    entry_point: str
    prompt: str
    checks: list[str]
    reference: str
    topic: str = ""
    round: int = 0
    pass_rate: float | None = None
    meta: dict = field(default_factory=dict)

    def as_task(self) -> Task:
        return Task(id=self.id, split="train", entry_point=self.entry_point, prompt=self.prompt,
                    checks=self.checks, source="generated")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        return json.loads(text[a:b + 1])
    raise ValueError("no JSON in model output")


def _similar(a: str, b: str) -> float:
    sm = difflib.SequenceMatcher(None, a.lower(), b.lower())
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return 0.0
    return sm.ratio()


class TaskValidator:
    def __init__(self, protected: list[Task], pool: list[GenTask], similarity: float = 0.6):
        self.protected = protected
        self.protected_names = {t.entry_point.lower() for t in protected}
        self.pool = list(pool)
        self.similarity = similarity

    def validate(self, raw: dict) -> tuple[GenTask | None, str]:
        try:
            ep = str(raw["entry_point"]).strip()
            prompt = str(raw["prompt"]).strip()
            ref = str(raw["reference_solution"])
            checks = [str(c).strip() for c in raw["checks"] if str(c).strip()]
        except (KeyError, TypeError):
            return None, "missing fields"
        if not re.fullmatch(r"[a-z_][a-z0-9_]{2,60}", ep):
            return None, f"bad entry point {ep!r}"
        if not 60 <= len(prompt) <= 3000:
            return None, "prompt length out of range"
        if not 5 <= len(checks) <= 25 or any(len(c) > 600 for c in checks):
            return None, "need 5-25 checks"
        if not all(ep in c for c in checks):
            return None, "every check must call the entry point"
        if ep.lower() in self.protected_names:
            return None, "name collides with a benchmark task"
        for t in self.protected:
            if _similar(prompt, t.prompt) >= self.similarity:
                return None, f"too similar to benchmark task {t.id}"
        for g in self.pool:
            if g.entry_point == ep and _similar(prompt, g.prompt) >= 0.8:
                return None, "duplicate of an existing generated task"
        task = GenTask(id="", entry_point=ep, prompt=prompt, checks=checks, reference=ref.strip())
        r = grade(task.as_task(), task.reference)
        if not r.solved:
            return None, f"reference fails its own checks ({r.passed}/{r.total}) {r.error[:80]}"
        stub = grade(task.as_task(), f"def {ep}(*args, **kwargs):\n    return None\n")
        if stub.passed > 0.4 * stub.total:
            return None, "checks too weak (a stub passes many)"
        return task, ""

    def accept(self, task: GenTask) -> None:
        self.pool.append(task)


def generate_tasks(client: OllamaClient, model: str, recipe: dict, *, round_no: int, protected: list[Task],
                   pool: list[GenTask], num_ctx: int, on_event: Callable[[str, str], None],
                   stop: threading.Event, max_calls: int | None = None) -> list[GenTask]:
    cfg = recipe["task_gen"]
    target, per_call = cfg["tasks_per_round"], cfg["tasks_per_call"]
    topics = cfg["topics"]
    validator = TaskValidator(protected, pool)
    out: list[GenTask] = []
    calls = 0
    max_calls = max_calls or max(4, 3 * target // per_call)
    rejected: dict[str, int] = {}
    while len(out) < target and calls < max_calls and not stop.is_set():
        topic = topics[calls % len(topics)]
        avoid = ", ".join(sorted({g.entry_point for g in validator.pool[-80:]} |
                                 {t.entry_point for t in protected}))[:3000]
        prompt = GEN_PROMPT.format(n=per_call, topic=topic, difficulty=cfg["difficulty"], avoid=avoid or "(none)")
        calls += 1
        try:
            r = client.chat(model, [{"role": "user", "content": prompt}], think=cfg["think"],
                            format=TASK_SCHEMA, options={"temperature": cfg["temperature"], "num_ctx": num_ctx},
                            stop=stop)
            items = _extract_json(r.content).get("tasks", [])
        except (OllamaError, ValueError, AttributeError) as e:
            on_event("taskgen", f"generation call {calls} failed: {str(e)[:150]}")
            if isinstance(e, OllamaError) and "Cannot reach" in str(e):
                raise
            continue
        for raw in items if isinstance(items, list) else []:
            if not isinstance(raw, dict):
                continue
            task, why = validator.validate(raw)
            if task is None:
                rejected[why.split(" (")[0]] = rejected.get(why.split(" (")[0], 0) + 1
                continue
            task.id = f"gen_r{round_no:03d}_{len(pool) + len(out):05d}"
            task.topic, task.round = topic, round_no
            validator.accept(task)
            out.append(task)
        on_event("taskgen", f"call {calls}: {len(out)}/{target} verified tasks")
    if rejected:
        top = ", ".join(f"{k}: {v}" for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])[:5])
        on_event("taskgen", f"rejected proposals by reason: {top}")
    return out
