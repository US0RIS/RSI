"""The full RSI loop behind the one button.

Each round:
  1. scaffold:  a few generations of code/prompt self-modification (evolve.py)
  2. recipe:    the CURRENT model proposes the next training recipe from the
                history of past rounds (what data, how filtered, how trained)
  3. tasks:     the current model writes new problems; only verified ones are kept
  4. data:      the current model attempts them; only answers that pass every
                hidden check become training examples (+ replay of older rounds)
  5. train:     QLoRA fine-tune in the trainer environment
  6. import:    adapter -> Ollama model, probed for sanity
  7. gate:      champion genome re-evaluated under the new weights; promoted only if
                hold-out improves by the margin and dev does not regress
After a promotion, every later step (scaffold search, task generation, recipe
proposals, data collection) runs on the improved weights. That feedback is the
recursive part: better weights write better recipes and better training data.
"""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Callable

from ..archive import Archive, ModelRecord
from ..benchmark import Task
from ..config import Settings
from ..evolve import Evolver, InfraError
from ..ollama_client import OllamaClient, OllamaError
from . import recipe as R
from .collect import collect_examples
from .ollama_import import Importer, ImportError_
from .taskgen import GenTask, _extract_json, generate_tasks
from .trainer import TrainerBridge, TrainerError

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {"analysis": {"type": "string"}, "recipe": {"type": "object"}},
    "required": ["analysis", "recipe"],
}

RECIPE_PROMPT = """You are designing the next round of your OWN post-training. Each round, you (the current model)
generate coding problems, attempt them, and only answers that pass hidden tests become training data for a LoRA
fine-tune of your weights. The new weights are kept only if they score higher on held-out tasks you never see.
If kept, the improved model designs the following round, so good recipes compound.

## Current recipe (used to train the current model)
{recipe}

## Allowed fields and bounds (anything else is ignored; values are clamped)
{bounds}

## History of rounds (most recent last)
{history}

## Where the current model + champion agent still fails (dev set)
{report}

Think about what limited the last rounds: too little data, data too easy or too hard (see pass rates), overfitting
(dev up, hold-out flat), forgetting (scores dropped), learning rate or LoRA capacity, topic coverage. Propose ONE
coherent recipe for the next round. Prefer changes the history suggests will help; avoid repeating failures.

Respond with JSON: {{"analysis": "...", "recipe": {{...full recipe...}}}}"""


class RSIEngine:
    def __init__(self, settings: Settings, archive: Archive | None = None, client: OllamaClient | None = None,
                 tasks: list[Task] | None = None, on_event: Callable[[str, str], None] | None = None,
                 trainer: TrainerBridge | None = None, importer: Importer | None = None,
                 rng: random.Random | None = None):
        self.s = settings
        self.evolver = Evolver(settings, archive=archive, client=client, tasks=tasks, on_event=on_event, rng=rng)
        self.archive = self.evolver.archive
        self.client = self.evolver.client
        self.stop: threading.Event = self.evolver.stop
        self.trainer = trainer or TrainerBridge(settings)
        self.importer = importer or Importer(settings, self.client)
        self.rng = rng or random.Random()
        self.root = self.archive.root / "selftrain"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage = "idle"
        self.trainer_ok: bool | None = None

    # ----------------------------------------------------------------- utils
    def emit(self, kind: str, msg: str) -> None:
        self.evolver.emit(kind, msg)

    def set_stage(self, stage: str, detail: str = "") -> None:
        self.stage = stage
        self.emit("stage", f"{stage}: {detail}" if detail else stage)

    def _jsonl_read(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _jsonl_append(self, path: Path, rows: list[dict]) -> None:
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    @property
    def pool_path(self) -> Path:
        return self.root / "task_pool.jsonl"

    @property
    def examples_path(self) -> Path:
        return self.root / "examples_all.jsonl"

    def model_tag(self, round_no: int) -> str:
        return f"{self.s.base_model.split(':')[0]}-rsi:r{round_no}"

    # --------------------------------------------------------------- models
    def ensure_models(self) -> ModelRecord:
        if self.archive.get_model(self.s.model) is None:
            self.archive.add_model(self.s.model, None, 0, "base", recipe=R.DEFAULT_RECIPE)
        rec = self.archive.get_model(self.s.model)
        if not rec.is_current:
            self.archive.set_current_model(self.s.model)
        return self.archive.get_model(self.s.model)

    def next_round(self) -> int:
        return max([m.round for m in self.archive.models()] + [0]) + 1

    # --------------------------------------------------------------- recipe
    def history_text(self) -> str:
        rows = [m for m in self.archive.models() if m.round >= 1]
        if not rows:
            return "(no previous rounds)"
        lines = []
        for m in rows[-8:]:
            parent = self.archive.get_model(m.parent_tag) if m.parent_tag else None
            changes = R.diff(R.clamp(parent.recipe) if parent else R.DEFAULT_RECIPE, R.clamp(m.recipe))
            st = m.stats or {}
            data = (f"{st.get('new_tasks', 0)} new tasks, mean pass rate {st.get('mean_pass_rate', 0):.2f}, "
                    f"{st.get('too_easy', 0)} too easy, {st.get('unsolved', 0)} unsolved, "
                    f"{st.get('examples', 0)} examples ({st.get('reference_examples', 0)} reference, "
                    f"{st.get('replay_examples', 0)} replay), loss {st.get('first_loss')}->{st.get('final_loss')}")

            def delta(a, b):
                return "n/a" if a is None or b is None else f"{a:.3f} ({a - b:+.3f})"
            scores = (f"dev {delta(m.dev, parent.dev if parent else None)}, "
                      f"hold-out {delta(m.holdout, parent.holdout if parent else None)}, "
                      f"audit {delta(m.audit, parent.audit if parent else None)}")
            lines.append(f"- {m.tag} round {m.round} from {m.parent_tag}: {m.status.upper()}"
                         f"{' (' + m.error[:150] + ')' if m.error else ''}\n"
                         f"  changes: {'; '.join(changes) or 'none'}\n  data: {data}\n  scores: {scores}")
        return "\n".join(lines)

    def propose_recipe(self) -> tuple[dict, str]:
        cur = self.archive.get_model(self.s.model)
        parent = R.clamp(cur.recipe if cur and cur.recipe else None)
        rounds = [m for m in self.archive.models() if m.round >= 1]
        if not rounds:
            return parent, "first round: default recipe"
        champ = self.archive.champion()
        report = self.evolver.build_report(champ.id)[:3000] if champ else ""
        prompt = RECIPE_PROMPT.format(recipe=json.dumps(parent, indent=2), bounds=R.describe_bounds(),
                                      history=self.history_text(), report=report)
        try:
            r = self.client.chat(self.s.model, [{"role": "user", "content": prompt}], think="high",
                                 format=RECIPE_SCHEMA, options={"num_ctx": self.s.num_ctx, "temperature": 0.7},
                                 stop=self.stop)
            proposal = _extract_json(r.content)
            new = R.clamp(proposal.get("recipe"), base=parent)
            why = str(proposal.get("analysis", ""))[:1500]
        except (OllamaError, ValueError, AttributeError) as e:
            if isinstance(e, OllamaError) and "Cannot reach" in str(e):
                raise InfraError(str(e))
            new, why = parent, f"recipe proposal failed ({str(e)[:120]})"
        last = rounds[-1]
        if new == parent and last.status != "promoted":
            new, changes = R.mutate(parent, self.rng)
            why += f"\n[exploration] proposal repeated a recipe that did not improve; mutated: {'; '.join(changes)}"
        return new, why

    # ------------------------------------------------------------------ round
    def _champion_solver(self, champ_id: str):
        files = self.archive.read_files(champ_id)

        def solve(task: GenTask) -> str:
            out = self.evolver._session(champ_id, "solve", {"task": task.as_task().public(), "genome": files})
            return str(out.result.get("code", "")) if out.ok and isinstance(out.result, dict) else ""
        return solve

    def _audit_tasks(self) -> list[Task]:
        return [t for t in self.evolver.tasks if t.split == "audit"]

    def _baseline_audit(self, champ_id: str) -> float | None:
        audit = self._audit_tasks()
        if not audit:
            return None
        cached = self.archive.model_score(self.s.model, champ_id, "audit")
        if cached is not None:
            return cached
        self.set_stage("measure", f"audit score of current model {self.s.model}")
        a = self.evolver.evaluate_tasks(champ_id, audit, "audit", record=False)
        if self.stop.is_set():
            return None
        self.archive.set_model_score(self.s.model, champ_id, "audit", a.score)
        cur = self.archive.get_model(self.s.model)
        if cur and cur.audit is None:
            self.archive.update_model(self.s.model, audit=a.score)
        return a.score

    def run_round(self, r: int) -> str:
        """Returns the round's outcome status."""
        # 1. scaffold self-modification on the current weights
        if self.s.rsi_scaffold_generations > 0:
            self.set_stage("scaffold", f"{self.s.rsi_scaffold_generations} generation(s) on {self.s.model}")
            saved = self.s.require_approval
            self.s.require_approval = saved and not self.s.rsi_auto_promote
            try:
                self.evolver.run(self.s.rsi_scaffold_generations, clear_stop=False)
            finally:
                self.s.require_approval = saved
        if self.stop.is_set():
            return "stopped"

        champ = self.archive.champion()
        cur = self.archive.get_model(self.s.model)
        if cur and cur.dev is None and champ:
            self.archive.update_model(self.s.model, dev=champ.dev, holdout=champ.holdout)
        base_audit = self._baseline_audit(champ.id)
        if self.stop.is_set():
            return "stopped"

        # 2. recipe, proposed by the current model
        self.set_stage("recipe", f"round {r}: {self.s.model} designs its training")
        recipe, why = self.propose_recipe()
        tag = self.model_tag(r)
        self.archive.add_model(tag, self.s.model, r, "training", recipe=recipe)
        changes = R.diff(R.clamp(cur.recipe) if cur else R.DEFAULT_RECIPE, recipe)
        self.emit("recipe", f"{tag}: {'; '.join(changes) or 'unchanged'}\n{why[:600]}")
        rdir = self.root / f"round_{r:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "recipe.json").write_text(json.dumps({"recipe": recipe, "analysis": why}, indent=2), encoding="utf-8")

        def fail(status: str, error: str) -> str:
            self.archive.update_model(tag, status=status, error=error[:2000])
            self.emit("round", f"{tag} {status}: {error[:300]}")
            return status

        # 3. self-generated, verified tasks
        self.set_stage("tasks", f"generating up to {recipe['task_gen']['tasks_per_round']} verified tasks")
        pool = [GenTask.from_dict(d) for d in self._jsonl_read(self.pool_path)]
        try:
            new_tasks = generate_tasks(self.client, self.s.model, recipe, round_no=r, protected=self.evolver.tasks,
                                       pool=pool, num_ctx=self.s.num_ctx, on_event=self.emit, stop=self.stop)
        except OllamaError as e:
            raise InfraError(str(e))
        self._jsonl_append(self.pool_path, [t.to_dict() for t in new_tasks])
        if self.stop.is_set():
            return fail("stopped", "stopped during task generation")
        if not new_tasks:
            return fail("failed", "no task passed verification")

        # 4. verified attempts -> training data
        self.set_stage("data", f"sampling {recipe['sampling']['attempts_per_task']} attempts x {len(new_tasks)} tasks")
        system = self.archive.read_files(champ.id).get("system_prompt.md", "")
        replay = self._jsonl_read(self.examples_path)
        try:
            examples, stats = collect_examples(
                new_tasks, recipe, client=self.client, model=self.s.model, system_prompt=system,
                num_ctx=self.s.num_ctx, round_no=r, on_event=self.emit, stop=self.stop,
                champion_solve=self._champion_solver(champ.id), replay_pool=replay, rng=self.rng)
        except OllamaError as e:
            raise InfraError(str(e))
        fresh = [e for e in examples if e.get("source") != "replay"]
        self._jsonl_append(self.examples_path, fresh)
        data_path = rdir / "data.jsonl"
        data_path.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
        st = dict(stats.to_dict(), new_tasks=len(new_tasks))
        self.archive.update_model(tag, stats=st)
        self.emit("data", f"{tag}: {stats.examples} examples ({stats.reference_examples} reference, "
                          f"{stats.replay_examples} replay); mean pass rate {stats.mean_pass_rate:.2f}")
        if self.stop.is_set():
            return fail("stopped", "stopped during data collection")
        if stats.examples < self.s.rsi_min_examples:
            return fail("failed", f"only {stats.examples} examples (< {self.s.rsi_min_examples}); "
                                  f"data kept for replay")

        # 5. train
        if self.trainer_ok is False:
            return fail("skipped", "trainer environment not available; data saved for when it is")
        self.set_stage("train", f"QLoRA on {stats.examples} examples")

        def on_train(ev: dict) -> None:
            if ev.get("event") == "progress":
                self.stage = f"train {ev.get('step')}/{ev.get('max_steps')} loss {ev.get('loss')}"
                if ev.get("step", 0) % 10 == 0:
                    self.emit("train", self.stage)
            elif ev.get("event") in ("stage", "data", "done"):
                self.emit("train", json.dumps(ev)[:300])
        job = {"base_model": self.s.hf_base_model, "data_path": str(data_path), "train": recipe["train"], "seed": r}
        try:
            res = self.trainer.run("train", job, rdir, on_event=on_train, stop=self.stop,
                                   timeout_s=self.s.rsi_train_timeout_h * 3600)
        except (TrainerError, OSError) as e:
            return fail("stopped" if self.stop.is_set() else "failed", f"training: {e}")
        st.update(first_loss=res.get("first_loss"), final_loss=res.get("final_loss"),
                  train_seconds=res.get("seconds"))
        self.archive.update_model(tag, stats=st)

        # 6. import into Ollama (+ probe)
        self.set_stage("import", f"creating {tag} in Ollama")
        ok, detail = False, ""
        try:
            self.importer.create_from_adapter(tag, self.s.base_model, Path(res["adapter_dir"]), rdir)
            ok, detail = self.importer.probe(tag)
        except (ImportError_, KeyError) as e:
            detail = str(e)
        if not ok:
            self.emit("import", f"adapter import unusable ({detail[:200]}); merging weights instead")
            try:
                res = self.trainer.run("merge", job, rdir, on_event=on_train, stop=self.stop,
                                       timeout_s=self.s.rsi_train_timeout_h * 3600)
                self.importer.create_from_merged(tag, self.s.base_model, Path(res["merged_dir"]), rdir)
                ok, detail = self.importer.probe(tag)
            except (TrainerError, ImportError_, KeyError, OSError) as e:
                ok, detail = False, str(e)
        if not ok:
            if not self.s.rsi_keep_failed_models:
                self.importer.remove(tag, rdir)
            return fail("failed", f"import: {detail}")

        # 7. gate: same champion genome, new weights
        return self.gate(tag, champ.id, base_audit)

    def gate(self, tag: str, champ_id: str, base_audit: float | None) -> str:
        champ = self.archive.get(champ_id)
        self.set_stage("gate", f"evaluating champion genome {champ_id} under {tag}")
        cand = self.archive.add(self.archive.read_files(champ_id), champ_id, f"champion genome under {tag}",
                                model=tag)
        self.evolver.model_override = tag
        try:
            d = self.evolver.evaluate_tasks(cand, self.evolver.dev, "dev")
            h = self.evolver.evaluate_tasks(cand, self.evolver.holdout, "holdout") if not self.stop.is_set() else None
            audit = self._audit_tasks()
            a = (self.evolver.evaluate_tasks(cand, audit, "audit", record=False)
                 if audit and not self.stop.is_set() else None)
        finally:
            self.evolver.model_override = None
        if self.stop.is_set() or h is None:
            self.archive.update(cand, status="rejected", error="stopped during gate")
            self.archive.update_model(tag, status="stopped", error="stopped during gate")
            return "stopped"
        smoke_ids = {t.id for t in self.evolver.smoke_tasks()}
        smoke = [g.score for g in d.results if g.task_id in smoke_ids]
        self.archive.update(cand, dev=d.score, holdout=h.score, smoke=sum(smoke) / max(1, len(smoke)))
        for split, val in (("dev", d.score), ("holdout", h.score), ("audit", a.score if a else None)):
            if val is not None:
                self.archive.set_model_score(tag, champ_id, split, val)
        self.archive.update_model(tag, dev=d.score, holdout=h.score, audit=a.score if a else None)

        cur_dev, cur_hold = champ.dev or 0.0, champ.holdout or 0.0
        audit_txt = "" if a is None or base_audit is None else f", audit {base_audit:.3f}->{a.score:.3f} (not gated)"
        summary = (f"dev {cur_dev:.3f}->{d.score:.3f}, hold-out {cur_hold:.3f}->{h.score:.3f}{audit_txt}")
        passed = h.score >= cur_hold + self.s.rsi_model_margin and d.score >= cur_dev - self.s.rsi_dev_tolerance
        if not passed:
            self.archive.update(cand, status="rejected", error=f"weights not better: {summary}")
            self.archive.update_model(tag, status="rejected", error=summary)
            if not self.s.rsi_keep_failed_models:
                self.importer.remove(tag, self.root)
            self.emit("round", f"{tag} rejected: {summary}")
            return "rejected"
        self.archive.update(cand, status="evaluated")
        self.archive.set_champion(cand)
        self.archive.update_model(tag, status="promoted")
        self.archive.set_current_model(tag)
        self.s.model = tag
        self.s.save()
        self.emit("champion", f"NEW WEIGHTS {tag} promoted: {summary}. All further work runs on {tag}.")
        return "promoted"

    # ------------------------------------------------------------------- run
    def run(self, rounds: int | None = None) -> None:
        self.stop.clear()
        try:
            self.ensure_models()
            self.set_stage("setup", f"seed/champion under {self.s.model}")
            self.evolver.ensure_seed()
            if self.stop.is_set():
                return
            self.set_stage("setup", "checking trainer environment")
            self.trainer_ok, info = self.trainer.check()
            if self.trainer_ok:
                self.emit("trainer", f"trainer ready: {info.get('gpu')} {info.get('vram_gb')} GB, "
                                     f"unsloth {info.get('unsloth')}" + (f" ({info['warning']})" if info.get('warning') else ""))
            else:
                self.emit("trainer", "trainer NOT available, rounds will improve the agent and collect verified "
                                     f"training data but cannot train weights: {json.dumps(info)[:400]}")
            n = 0
            while not self.stop.is_set() and (rounds is None or n < rounds):
                r = self.next_round()
                outcome = self.run_round(r)
                self.emit("round", f"round {r} finished: {outcome}")
                n += 1
            self.emit("done", f"RSI stopped after {n} round(s); current model {self.s.model}")
        except InfraError as e:
            self.emit("error", f"infrastructure problem, RSI paused: {e}")
            raise
        finally:
            self.stage = "idle"
