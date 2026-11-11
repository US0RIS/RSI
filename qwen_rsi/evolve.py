"""The recursive self-improvement loop (Darwin-Gödel-Machine style).

One generation:
  1. pick a parent from the archive (score-weighted, novelty- and
     productivity-aware),
  2. run the PARENT's own improve() on its own source plus its evaluation
     report -> child genome,
  3. statically validate the child (guard, required entry points),
  4. staged evaluation: smoke subset -> full dev set -> hold-out set,
  5. archive the child whatever the result (open-ended exploration),
  6. if it beats the champion on the hold-out set by a margin, promote it
     (or queue it for human approval).

Everything in this module is supervisor code: genomes cannot modify it, read
the hidden checks, or see hold-out tasks.
"""
from __future__ import annotations

import ast
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import guard
from .archive import GENOME_FILES, MAX_GENOME_FILE_BYTES, Archive
from .benchmark import GradeResult, Task, grade, load_tasks
from .config import Settings, data_dir
from .ollama_client import OllamaClient, OllamaError
from .sandbox import GenomeSession, SessionOutcome

SEED_DIR = Path(__file__).with_name("seed_genome")

ALLOWED_OPTIONS = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty", "repeat_last_n",
                   "presence_penalty", "frequency_penalty", "seed", "num_predict", "stop"}
ALLOWED_THINK = {None, True, False, "low", "medium", "high", "max"}
MAX_NUM_PREDICT = 32768


class InfraError(RuntimeError):
    """Ollama unreachable or similar: not the genome's fault, abort the run."""


@dataclass
class EvalSummary:
    score: float
    results: list[GradeResult]
    llm_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0


def contract_text(s: Settings) -> str:
    return f"""- Files: {', '.join(GENOME_FILES)} (max {MAX_GENOME_FILE_BYTES // 1000} kB each). agent.py must keep
  `def solve(payload, llm, tools)` and `def improve(payload, llm, tools)`.
- solve() receives payload = {{"task": {{"id", "kind": "code"|"chat", "entry_point", "prompt"}} or
  {{"kind": "chat", "messages": [...]}}, "genome": {{filename: text}}}} and must return {{"code": str}} for code
  tasks or {{"content": str}} for chat tasks.
- improve() receives payload = {{"genome": {{filename: text}}, "report": str, "history": str, "contract": str}}
  and must return {{"files": {{filename: text}}, "rationale": str}}.
- agent.py may import only: {', '.join(sorted(guard.GENOME_IMPORTS))}. It may not use open, exec, eval,
  compile, __import__, getattr/setattr, globals, or dunder attributes. Files are given to you in the payload.
- llm.chat(messages, think=None|bool|"low"|"medium"|"high"|"max", format=None|"json"|JSON-schema,
  options=dict, tools=list) -> {{"content", "thinking", "tool_calls", "prompt_tokens", "output_tokens"}}.
  Allowed options: {', '.join(sorted(ALLOWED_OPTIONS))}; num_predict <= {MAX_NUM_PREDICT}. Context size is fixed by
  the supervisor at {s.num_ctx} tokens.
- tools.run_python(code, stdin="", timeout=None) runs code in a fresh sandbox (max {s.exec_timeout_s:g}s; same
  import allowlist as task solutions) and returns {{"returncode", "stdout", "stderr", "timed_out", "seconds"}}.
  tools.log(msg) writes to the app log.
- Budgets per solve(): {s.llm_calls_per_task} LLM calls, {s.exec_calls_per_task} run_python calls,
  {s.task_timeout_s:g}s wall clock. Per improve(): {s.llm_calls_per_improve} LLM calls, {s.improve_timeout_s:g}s.
  Exceeding the LLM budget raises an exception inside llm.chat; exceeding wall clock kills the process
  and scores the task 0. Faster, cheaper agents are preferred when scores tie.
- Hidden tests grade solve() outputs. Hold-out tasks are never shown to you."""


class Evolver:
    def __init__(self, settings: Settings, archive: Archive | None = None,
                 client: OllamaClient | None = None, tasks: list[Task] | None = None,
                 on_event: Callable[[str, str], None] | None = None, rng: random.Random | None = None):
        self.s = settings
        self.archive = archive or Archive(data_dir())
        self.client = client or OllamaClient(settings.ollama_url, settings.request_timeout_s)
        self.tasks = tasks if tasks is not None else load_tasks(data_dir() / "tasks")
        self.dev = [t for t in self.tasks if t.split == "dev"]
        self.holdout = [t for t in self.tasks if t.split == "holdout"]
        self.on_event = on_event or (lambda kind, msg: None)
        self.rng = rng or random.Random()
        self.stop = threading.Event()
        self._infra_error: str | None = None

    # ------------------------------------------------------------------ util
    def emit(self, kind: str, msg: str) -> None:
        self.archive.log(kind, msg)
        self.on_event(kind, msg)

    def _chat_fn(self, req: dict) -> dict:
        """Broker for genome LLM calls: clamps everything the genome controls."""
        opts = {k: v for k, v in (req.get("options") or {}).items() if k in ALLOWED_OPTIONS}
        if "num_predict" in opts:
            try:
                opts["num_predict"] = max(1, min(int(opts["num_predict"]), MAX_NUM_PREDICT))
            except (TypeError, ValueError):
                opts.pop("num_predict")
        opts["num_ctx"] = self.s.num_ctx
        think = req.get("think")
        if think not in ALLOWED_THINK:
            think = None
        messages = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
        try:
            r = self.client.chat(self.s.model, messages, think=think, format=req.get("format"),
                                 options=opts, tools=req.get("tools") or None,
                                 keep_alive=self.s.keep_alive, stop=self.stop)
        except OllamaError as e:
            text = str(e)
            if "Cannot reach Ollama" in text or "not found" in text.lower():
                self._infra_error = text
            return {"error": text}
        except OSError as e:  # connection reset, timeout
            self._infra_error = f"{type(e).__name__}: {e}"
            return {"error": self._infra_error}
        return {"content": r.content, "thinking": r.thinking, "tool_calls": r.tool_calls,
                "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens, "done_reason": r.done_reason}

    def _session(self, vid: str, entry: str, payload: dict) -> SessionOutcome:
        is_improve = entry == "improve"
        sess = GenomeSession(
            self.archive.genome_dir(vid), chat_fn=self._chat_fn,
            max_llm_calls=self.s.llm_calls_per_improve if is_improve else self.s.llm_calls_per_task,
            max_exec_calls=self.s.exec_calls_per_task,
            timeout_s=self.s.improve_timeout_s if is_improve else self.s.task_timeout_s,
            exec_timeout_s=self.s.exec_timeout_s, memory_mb=self.s.sandbox_memory_mb,
            on_event=lambda kind, text: self.on_event("genome", f"[{vid}/{entry}] {text}") if kind == "log" else None,
            stop=self.stop)
        out = sess.run(entry, payload)
        if self._infra_error:
            err, self._infra_error = self._infra_error, None
            raise InfraError(err)
        return out

    # ------------------------------------------------------------ evaluation
    def evaluate_tasks(self, vid: str, tasks: list[Task], stage: str) -> EvalSummary:
        genome = self.archive.read_files(vid)
        results: list[GradeResult] = []
        calls = toks = 0
        secs = 0.0
        for t in tasks:
            if self.stop.is_set():
                break
            out = self._session(vid, "solve", {"task": t.public(), "genome": genome})
            if out.ok and isinstance(out.result, dict):
                g = grade(t, out.result.get("code", ""), memory_mb=min(self.s.sandbox_memory_mb, 1024))
            else:
                g = GradeResult(t.id, 0.0, 0, len(t.checks), error=(out.error or "no result")[-1500:])
            st = out.stats
            calls += st.llm_calls
            toks += st.prompt_tokens + st.output_tokens
            secs += st.seconds
            self.archive.record_eval(vid, t.id, t.split, stage, g.score, g.passed, g.total, g.failures, g.error,
                                     st.seconds, st.llm_calls, st.prompt_tokens + st.output_tokens)
            self.archive.add_usage(vid, st.llm_calls, st.prompt_tokens, st.output_tokens, st.seconds)
            self.emit("eval", f"{vid} {stage} {t.id}: {g.passed}/{g.total}" + (f" ({g.error[:120]})" if g.error else ""))
            results.append(g)
        score = sum(r.score for r in results) / len(tasks) if tasks else 0.0
        return EvalSummary(score, results, calls, toks, secs)

    def smoke_tasks(self) -> list[Task]:
        return self.dev[: max(1, min(self.s.smoke_tasks, len(self.dev)))]

    def evaluate(self, vid: str, parent_id: str | None) -> None:
        """Staged evaluation + promotion decision."""
        smoke = self.smoke_tasks()
        s1 = self.evaluate_tasks(vid, smoke, "smoke")
        if self.stop.is_set():
            return
        self.archive.update(vid, smoke=s1.score)
        if parent_id is not None:
            p = self.archive.get(parent_id)
            if p.smoke is not None and s1.score < self.s.smoke_min_frac * p.smoke:
                self.archive.update(vid, status="rejected",
                                    error=f"smoke {s1.score:.2f} < {self.s.smoke_min_frac:.0%} of parent {p.smoke:.2f}")
                self.emit("reject", f"{vid} rejected at smoke stage ({s1.score:.2f})")
                return
        rest = [t for t in self.dev if t not in smoke]
        s2 = self.evaluate_tasks(vid, rest, "dev")
        if self.stop.is_set():
            return
        dev = (s1.score * len(smoke) + s2.score * len(rest)) / max(1, len(self.dev))
        self.archive.update(vid, dev=dev, status="evaluated")
        self.emit("score", f"{vid} dev={dev:.3f}")

        champ = self.archive.champion()
        if champ is None:
            h = self.evaluate_tasks(vid, self.holdout, "holdout")
            self.archive.update(vid, holdout=h.score)
            self.archive.set_champion(vid)
            self.emit("champion", f"{vid} is the initial champion (holdout={h.score:.3f})")
            return
        if champ.dev is not None and dev + 1e-9 < champ.dev:
            return  # not a promotion candidate; stays in the archive for exploration
        h = self.evaluate_tasks(vid, self.holdout, "holdout")
        if self.stop.is_set():
            return
        self.archive.update(vid, holdout=h.score)
        self.emit("score", f"{vid} holdout={h.score:.3f} (champion {champ.id} holdout={champ.holdout or 0:.3f})")
        if h.score >= (champ.holdout or 0.0) + self.s.holdout_margin:
            if self.s.require_approval:
                # Only the best pending candidate stays queued for the human.
                for other in self.archive.all():
                    if other.awaiting_approval and (other.holdout or 0.0) <= h.score:
                        self.archive.update(other.id, awaiting_approval=0)
                self.archive.update(vid, awaiting_approval=1)
                self.emit("approval", f"{vid} beats champion {champ.id} on hold-out; awaiting your approval")
            else:
                self.archive.set_champion(vid)
                self.emit("champion", f"{vid} promoted to champion")

    # -------------------------------------------------------------- selection
    def select_parent(self) -> str:
        """DGM parent selection: sigmoid-scaled score x novelty bonus, plus a
        bonus for variants whose children tend to improve (good improvers)."""
        cands = self.archive.evaluated()
        if not cands:
            raise RuntimeError("archive has no evaluated variants")
        weights = []
        for v in cands:
            perf = 1.0 / (1.0 + math.exp(-self.s.selection_lambda * ((v.dev or 0.0) - self.s.selection_midpoint)))
            novelty = 1.0 / (1.0 + len(self.archive.children(v.id)))
            prod = self.archive.productivity(v.id)
            weights.append(perf * novelty * (1.0 + (prod if prod is not None else 0.5)))
        return self.rng.choices([v.id for v in cands], weights=weights, k=1)[0]

    # ------------------------------------------------------------ reporting
    def build_report(self, vid: str) -> str:
        v = self.archive.get(vid)
        dev_by_id = {t.id: t for t in self.dev}
        lines = [f"Dev score: {v.dev:.3f} over {len(self.dev)} tasks (fraction of hidden checks passed)."
                 if v.dev is not None else "Dev score: not yet measured.",
                 f"Usage so far: {v.llm_calls} LLM calls, {v.prompt_tokens + v.output_tokens} tokens, "
                 f"{v.seconds:.0f}s total."]
        if v.holdout is not None:
            lines.append(f"Hold-out score: {v.holdout:.3f} (details withheld).")
        failing = [e for e in self.archive.evals(vid, "dev") if e["score"] < 1.0]
        if failing:
            lines.append(f"\n{len(failing)} dev tasks not fully solved:")
        for e in failing[:10]:
            t = dev_by_id.get(e["task_id"])
            lines.append(f"\n### {e['task_id']} — {e['passed']}/{e['total']} checks, {e['llm_calls']} LLM calls, "
                         f"{e['seconds']:.0f}s")
            if t:
                lines.append("Task: " + t.prompt[:700])
            if e["error"]:
                lines.append("Error: " + e["error"][:700])
            for f in e["failures"][:4]:
                lines.append("- " + f[:300])
        return "\n".join(lines)

    def build_history(self, vid: str) -> str:
        out = []
        for v in self.archive.lineage(vid)[-6:]:
            score = f"dev={v.dev:.3f}" if v.dev is not None else v.status
            out.append(f"- {v.id} ({score}): {v.rationale[:400] or '(no rationale)'}")
        tried = [c for c in self.archive.children(vid)][-5:]
        if tried:
            out.append("\nPrevious attempts to improve THIS version:")
            for c in tried:
                res = f"dev={c.dev:.3f}" if c.dev is not None else f"{c.status}: {c.error[:200]}"
                out.append(f"- {c.id} -> {res}: {c.rationale[:300]}")
        return "\n".join(out) or "(no history)"

    # -------------------------------------------------------------- mutation
    def validate_files(self, files: object) -> tuple[dict[str, str] | None, str]:
        if not isinstance(files, dict):
            return None, "improve() did not return a 'files' dict"
        clean = {}
        for name, text in files.items():
            if name not in GENOME_FILES:
                continue
            if not isinstance(text, str):
                return None, f"{name} is not text"
            if len(text.encode("utf-8")) > MAX_GENOME_FILE_BYTES:
                return None, f"{name} exceeds {MAX_GENOME_FILE_BYTES} bytes"
            clean[name] = text
        src = clean.get("agent.py")
        if not src:
            return None, "agent.py missing"
        g = guard.check_genome_code(src)
        if not g.ok:
            return None, "guard: " + "; ".join(g.problems[:6])
        defs = {n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
        missing = {"solve", "improve"} - defs
        if missing:
            return None, "agent.py is missing " + ", ".join(sorted(missing))
        return clean, ""

    def ensure_seed(self) -> str:
        if not self.archive.all():
            vid = self.archive.import_seed(SEED_DIR)
            self.emit("seed", f"imported seed genome as {vid}")
        champ = self.archive.champion()
        if champ is None:
            roots = [v for v in self.archive.all() if v.parent_id is None]
            if roots:
                self.emit("eval", f"evaluating seed {roots[0].id}")
                self.evaluate(roots[0].id, None)
        champ = self.archive.champion()
        return champ.id if champ else self.archive.all()[0].id

    def step(self) -> str | None:
        """Run one generation. Returns the child id (or None if no child)."""
        parent_id = self.select_parent()
        parent = self.archive.get(parent_id)
        self.emit("select", f"parent {parent_id} (dev={parent.dev:.3f}, children={len(self.archive.children(parent_id))})")
        payload = {"genome": self.archive.read_files(parent_id), "report": self.build_report(parent_id),
                   "history": self.build_history(parent_id), "contract": contract_text(self.s)}
        t0 = time.monotonic()
        out = self._session(parent_id, "improve", payload)
        self.archive.add_usage(parent_id, out.stats.llm_calls, out.stats.prompt_tokens, out.stats.output_tokens,
                               out.stats.seconds)
        if self.stop.is_set():
            return None
        if not out.ok or not isinstance(out.result, dict):
            child = self.archive.add(self.archive.read_files(parent_id), parent_id, "improve() failed")
            self.archive.update(child, status="failed", error=(out.error or "no result")[-1500:])
            self.emit("fail", f"{parent_id}.improve() failed: {(out.error or '')[:200]}")
            return child
        files, problem = self.validate_files(out.result.get("files"))
        rationale = str(out.result.get("rationale", ""))
        if files is None:
            child = self.archive.add(self.archive.read_files(parent_id), parent_id, rationale)
            self.archive.update(child, status="failed", error=problem)
            self.emit("fail", f"child of {parent_id} invalid: {problem}")
            return child
        child = self.archive.add(files, parent_id, rationale)
        self.emit("child", f"{child} created from {parent_id} in {time.monotonic() - t0:.0f}s: {rationale[:200]}")
        self.evaluate(child, parent_id)
        return child

    def run(self, generations: int | None = None) -> None:
        self.stop.clear()
        try:
            self.ensure_seed()
            n = 0
            while not self.stop.is_set() and (generations is None or n < generations):
                self.step()
                n += 1
            self.emit("done", f"stopped after {n} generation(s)")
        except InfraError as e:
            self.emit("error", f"infrastructure problem, loop paused: {e}")
            raise

    # ------------------------------------------------------------ champion use
    def chat_with_champion(self, messages: list[dict]) -> str:
        champ = self.archive.champion()
        if champ is None:
            raise RuntimeError("no champion yet")
        out = self._session(champ.id, "solve", {"task": {"kind": "chat", "messages": messages},
                                                "genome": self.archive.read_files(champ.id)})
        if not out.ok or not isinstance(out.result, dict):
            raise RuntimeError(out.error or "champion returned nothing")
        return str(out.result.get("content", ""))
