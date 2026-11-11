"""Turn verified tasks into training examples (rejection sampling / expert iteration).

For each task the current model makes several attempts; only attempts that pass
ALL hidden checks become training data. Tasks the model already solves almost
every time are dropped (nothing to learn); tasks it never solves can contribute
the verified reference solution instead.
"""
from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..benchmark import grade
from ..ollama_client import OllamaClient, OllamaError
from .taskgen import GenTask

USER_SUFFIX = ("\n\nReturn a complete, self-contained Python module in a single ```python block that defines `{ep}`. "
               "Only standard-library imports. No example usage, no tests, no input().")


def user_message(task: GenTask) -> str:
    return task.prompt + USER_SUFFIX.format(ep=task.entry_point)


def extract_code(text: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return text.strip()
    for b in reversed(blocks):
        if re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(", b):
            return b.strip()
    return blocks[-1].strip()


@dataclass
class Attempt:
    passed: bool
    score: float
    content: str
    thinking: str
    code: str
    tokens: int = 0


@dataclass
class CollectStats:
    tasks: int = 0
    attempts: int = 0
    solved_attempts: int = 0
    kept_tasks: int = 0
    too_easy: int = 0
    unsolved: int = 0
    examples: int = 0
    reference_examples: int = 0
    replay_examples: int = 0
    mean_pass_rate: float = 0.0
    pass_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("pass_rates")
        return d


def _attempt_direct(client: OllamaClient, model: str, system: str, task: GenTask, cfg: dict, num_ctx: int,
                    stop: threading.Event) -> Attempt:
    think = None if cfg["think"] == "off" else cfg["think"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_message(task)}]
    r = client.chat(model, msgs, think=think if think else False,
                    options={"temperature": cfg["temperature"], "num_ctx": num_ctx}, stop=stop)
    code = extract_code(r.content, task.entry_point)
    g = grade(task.as_task(), code)
    return Attempt(g.solved, g.score, r.content, r.thinking, code, r.prompt_tokens + r.output_tokens)


def collect_examples(tasks: list[GenTask], recipe: dict, *, client: OllamaClient, model: str, system_prompt: str,
                     num_ctx: int, round_no: int, on_event: Callable[[str, str], None], stop: threading.Event,
                     champion_solve: Callable[[GenTask], str] | None = None,
                     replay_pool: list[dict] | None = None, rng: random.Random | None = None
                     ) -> tuple[list[dict], CollectStats]:
    """Returns (examples, stats). Each example is {"messages": [...], "thinking": str, ...}."""
    rng = rng or random.Random(round_no)
    s_cfg, f_cfg, d_cfg = recipe["sampling"], recipe["filter"], recipe["data"]
    stats = CollectStats(tasks=len(tasks))
    examples: list[dict] = []
    for i, task in enumerate(tasks):
        if stop.is_set():
            break
        attempts: list[Attempt] = []
        for _ in range(s_cfg["attempts_per_task"]):
            if stop.is_set():
                break
            try:
                if s_cfg["mode"] == "champion" and champion_solve is not None:
                    code = champion_solve(task)
                    g = grade(task.as_task(), code)
                    attempts.append(Attempt(g.solved, g.score, f"```python\n{code}\n```", "", code))
                else:
                    attempts.append(_attempt_direct(client, model, system_prompt, task, s_cfg, num_ctx, stop))
            except OllamaError as e:
                if "Cannot reach" in str(e):
                    raise
                on_event("collect", f"{task.id}: attempt failed: {str(e)[:120]}")
        if not attempts:
            continue
        stats.attempts += len(attempts)
        good = [a for a in attempts if a.passed]
        stats.solved_attempts += len(good)
        rate = len(good) / len(attempts)
        task.pass_rate = rate
        stats.pass_rates.append(rate)
        if rate > f_cfg["max_pass_rate"]:
            stats.too_easy += 1
            continue
        base = {"task_id": task.id, "round": round_no, "entry_point": task.entry_point}
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message(task)}]
        if good:
            stats.kept_tasks += 1
            if f_cfg["prefer_shortest"]:
                good.sort(key=lambda a: len(a.code))
            for a in good[: f_cfg["keep_per_task"]]:
                answer = f"```python\n{a.code}\n```"
                examples.append({**base, "source": "attempt", "pass_rate": rate,
                                 "messages": msgs + [{"role": "assistant", "content": answer}],
                                 "thinking": a.thinking if d_cfg["include_thinking"] else ""})
        else:
            stats.unsolved += 1
            if f_cfg["use_reference_when_unsolved"]:
                stats.kept_tasks += 1
                stats.reference_examples += 1
                examples.append({**base, "source": "reference", "pass_rate": 0.0,
                                 "messages": msgs + [{"role": "assistant",
                                                      "content": f"```python\n{task.reference}\n```"}],
                                 "thinking": ""})
        if (i + 1) % 5 == 0 or i + 1 == len(tasks):
            on_event("collect", f"{i + 1}/{len(tasks)} tasks sampled, {len(examples)} verified examples")

    # Replay earlier rounds' data to limit forgetting.
    frac = d_cfg["replay_fraction"]
    if replay_pool and frac > 0 and examples:
        n_replay = min(len(replay_pool), int(round(len(examples) * frac / max(1e-9, 1 - frac))))
        replay = rng.sample(replay_pool, n_replay)
        stats.replay_examples = len(replay)
        examples = examples + [dict(r, source="replay") for r in replay]
    rng.shuffle(examples)
    examples = examples[: d_cfg["max_examples"]]
    stats.examples = len(examples)
    stats.mean_pass_rate = sum(stats.pass_rates) / len(stats.pass_rates) if stats.pass_rates else 0.0
    return examples, stats
