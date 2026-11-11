# Qwen RSI: design

A Windows 11 desktop app that connects to **Qwen3.8 27B** running locally in **Ollama**
and runs a **recursive self-improvement** loop on the agent built around the model.

## 1. What "recursive self-improvement" means here

The model weights stay fixed. What evolves is the **genome**: the agent code
(`agent.py`), its prompts, its sampling and thinking configuration, and its notes.
The genome has two entry points:

| Entry point | Job |
|---|---|
| `solve(payload, llm, tools)` | Solve a task (a coding task from the benchmark, or a chat turn). |
| `improve(payload, llm, tools)` | Read its own source and its evaluation report, then return a modified copy of itself. |

Because `improve()` is part of the genome, a child can improve **how it improves**.
A better improver produces better children, and parent selection favours variants
whose children tend to beat them (the *Improver* column in the app). That feedback loop
is what makes the process recursive, not just iterative.

This follows the **Darwin Gödel Machine** (DGM, Zhang et al., arXiv 2505.22954): replace
the original Gödel machine's requirement of formal proofs with empirical validation
on a benchmark, and keep an open-ended archive of every variant instead of only the
best one, so stepping-stone variants can still produce winners later.

## 2. Research findings that shaped the design

**The model.** `qwen3.8:27b` is in the official Ollama library. It supports tools,
thinking and vision. The Q4_K_M build is about 17 GB, plus a ~0.9 GB vision projector, and
the community card lists a 262K context window. Ollama ships a separate `-mlx` tag for
Apple Silicon only, so it does not apply here. On Windows, a 24 GB GPU holds the weights
plus a moderate KV cache. The app defaults to `num_ctx = 32768` and makes it configurable,
because VRAM, not the model, is the real limit.

**The Ollama API** (`docs/api.md` in the ollama repository):
- `POST /api/chat` with `model`, `messages`, `tools`, `think`, `format`, `options`,
  `stream`, `keep_alive`.
- `think` accepts a boolean or a level: `"low"`, `"medium"`, `"high"`, `"max"`. The app uses
  levels per role: medium for solving, high for self-modification.
- `format` takes `"json"` or a JSON Schema. The seed `improve()` uses a schema for its edit
  proposals and also has a tolerant JSON extractor, because older Ollama builds did not
  allow thinking and a grammar together.
- Responses stream as NDJSON chunks with `message.thinking`, `message.content` and
  `message.tool_calls`. The final chunk has `done_reason`, `prompt_eval_count` and `eval_count`.
- `/api/show` returns `capabilities` and `<arch>.context_length`. `/api/ps` returns VRAM use.
  `/api/pull` streams download progress. The app uses all three.

**Self-improving agents** (the DGM paper and its coverage):
- Validation must be empirical and **staged**, so bad variants are rejected cheaply
  (DGM: 10 tasks → 50 → 200).
- An open-ended archive with novelty-weighted parent selection beats greedy hill-climbing.
- **Objective hacking is real.** DGM observed a variant that removed hallucination-detection
  markers to game its evaluation. The evaluator therefore has to be out of the genome's reach.
  This drives the trust boundaries in §4.

## 3. Architecture

```
┌──────────────────────── Windows 11 desktop app (PySide6 / Qt 6, "windows11" style) ───────────────┐
│  Chat tab                 Self-improvement tab                     Settings tab                   │
│  (stream + thinking)      (start/stop, score chart, archive table,  (Ollama, budgets, pull model) │
│                            diff viewer, approve/rollback, log)                                    │
└───────────────┬──────────────────────────┬────────────────────────────────────────────────────────┘
                │ QThread workers           │
┌───────────────▼──────────────────────────▼───────── SUPERVISOR (immutable, trusted) ────────────┐
│  Evolver ── parent selection ── staged evaluation ── promotion / approval gate                   │
│     │                                   │                                                        │
│  Archive (SQLite + genome dirs)     Benchmark + hidden checks + grader                          │
│     │                                   │                                                        │
│  Broker: clamps options, enforces LLM / exec budgets, fixes num_ctx, logs tokens                 │
└─────┬───────────────────────────────────┬────────────────────────────────────────────────────────┘
      │ JSON lines over stdin/stdout       │ one-shot runs
┌─────▼─────────────── SANDBOX PROCESS ─┐ ┌▼──────── SANDBOX PROCESS ───────┐      ┌──────────────┐
│ runner_shim.py (trusted)              │ │ solution + grader harness       │      │ Ollama       │
│   imports genome/agent.py (MUTABLE)   │ │ (hidden checks, random marker)  │      │ qwen3.8:27b  │
│   llm.chat ─────────── request ───────┼─┼─────────────────────────────────┼─────►│ 127.0.0.1:   │
│   tools.run_python ─── request        │ └─────────────────────────────────┘      │ 11434        │
└───────────────────────────────────────┘                                          └──────────────┘
```

| Module | Role |
|---|---|
| `qwen_rsi/ollama_client.py` | Stdlib-only Ollama client: streaming chat, thinking, tools, show/ps/pull. |
| `qwen_rsi/evolve.py` | The loop: selection, `improve()` runs, validation, staged evaluation, promotion. |
| `qwen_rsi/archive.py` | SQLite archive of variants, per-task evaluations, events, and write-once genome directories. |
| `qwen_rsi/benchmark/` | 30 original coding tasks (18 dev, 12 hold-out) with hidden checks; the grader. |
| `qwen_rsi/sandbox.py` | Process sandbox (Job Objects on Windows) and the JSON-lines broker. |
| `qwen_rsi/runner_shim.py` | Trusted code inside the sandbox that hosts the genome and relays its requests. |
| `qwen_rsi/guard.py` | AST allowlist for genome and solution code. |
| `qwen_rsi/seed_genome/` | Generation 0: `agent.py`, `system_prompt.md`, `improve_prompt.md`, `config.json`, `NOTES.md`. |
| `qwen_rsi/gui/` | The desktop app. |

## 4. Trust boundaries (anti-gaming)

| Can the genome… | Answer | Why |
|---|---|---|
| change the supervisor, grader or benchmark? | No | It runs in a separate process. Its files are copied into a temp directory, and only `GENOME_FILES` are accepted from `improve()`. |
| see hidden checks? | No | Checks run only in the grader process, never in the genome's process. |
| see hold-out tasks? | No | Reports and history include dev failures only. Hold-out contributes one number. A test asserts this. |
| spoof a grade? | No | The grader prints its result behind a random per-run marker, and the solution runs in its own namespace. |
| raise its own budgets? | No | Budgets are counted in the supervisor's broker. Forging protocol messages only produces more budgeted requests. |
| set a huge context or output length? | No | The broker forces `num_ctx` and clamps `num_predict` and the other options to an allowlist. |
| reach files, network or processes? | Blocked twice | The AST guard allowlists imports and blocks `open`, `exec`, `eval`, dunders and `getattr`. The process gets a scrubbed env, a temp cwd, `python -I`, a timeout and a Job Object (memory cap, process cap, kill-on-close). |
| become champion unnoticed? | Only if allowed | Promotion needs a hold-out gain of at least the margin, and by default a human approval after reviewing the diff. |

**Limits, stated plainly.** A Python AST guard can be bypassed by a determined adversary.
The real boundary is the OS process plus the Job Object, which is not a VM. The sandbox does
not block outbound network at the OS level. For untrusted long runs, run the whole app inside
**Windows Sandbox** or a VM with only the Ollama port forwarded.

## 5. The loop

```
ensure_seed():  import seed → full evaluation (dev + hold-out) → champion
repeat:
    parent  ← sample(archive, weight = sigmoid(λ·(dev − α₀)) · 1/(1+children) · (1 + productivity))
    child   ← sandbox(parent.improve, {own files, dev report, lineage history, contract})
    validate(child): allowed files only, size limit, guard OK, defines solve() and improve()
    smoke   ← first k dev tasks;  reject if smoke < ρ · parent.smoke
    dev     ← all dev tasks (partial credit = fraction of hidden checks passed)
    if dev ≥ champion.dev:  holdout ← hold-out tasks
        if holdout ≥ champion.holdout + margin:  queue for approval (or promote)
    archive(child)   # always, including failures: stepping stones and negative examples
```

- **Productivity**, the fraction of a variant's children that beat it, measures the quality
  of that variant's `improve()`. Selection weights it, so lineages that are good at
  self-modification get more chances. This is the selective pressure on the recursion itself.
- **History** passed to `improve()` holds the lineage's rationales and scores, plus earlier
  failed attempts on the same parent, so the model does not keep retrying the same idea.
- **Contract text** (generated from live settings) tells the genome its exact API, its
  budgets and the allowed imports.
- **Partial credit** smooths the signal on a 30-task benchmark. The hold-out margin (default
  0.05) absorbs sampling noise.

### Seed genome

- `solve()` makes one thinking call (medium), extracts the code block, runs a load check in
  the sandbox, and asks for one fix if loading fails.
- `improve()` makes a thinking call (high) with a JSON-schema edit proposal (search/replace or
  whole file). It applies the edits, checks that `agent.py` still parses and `config.json`
  still loads, and feeds errors back for up to 3 attempts.
- The seed deliberately leaves headroom: it never writes or runs its own tests, never samples
  several candidates, and never uses the `tools.run_python` budget beyond the load check.
  `improve_prompt.md` suggests these directions.

## 6. Windows 11 integration

- Qt 6.7+ `windows11` style (Fluent metrics, rounded controls, follows system light/dark). It
  falls back to `windowsvista` or Fusion.
- An explicit AppUserModelID gives the app its own taskbar identity and icon.
- Tray notifications (`QSystemTrayIcon.showMessage`, shown as Windows notifications) fire when
  a candidate needs approval or the loop pauses because Ollama is unreachable.
- Sandboxed code runs in **Job Objects** (`JOB_OBJECT_LIMIT_JOB_MEMORY`, `ACTIVE_PROCESS`,
  `KILL_ON_JOB_CLOSE`) with `CREATE_NO_WINDOW`, so no console windows flash. The sandbox uses the
  base `python.exe`, not the venv launcher or `pythonw.exe`, so the process cap and stdio pipes
  work.
- All state lives in `%LOCALAPPDATA%\QwenRSI` (`settings.json`, `archive.db`, `genomes\`, `tasks\`).
- `scripts/install.ps1` installs Python and Ollama with `winget`, pulls the model, checks VRAM
  with `nvidia-smi`, creates a venv, adds a Start menu shortcut and runs a self-check.

## 7. Cost and throughput

A full evaluation is 30 tasks × (1–3 LLM calls) × (1–4k tokens with thinking). At roughly
30–40 tokens/s on a 24 GB GPU, that is on the order of **30–60 minutes per surviving
child**. Smoke-rejected children cost only a few minutes. Expect a handful of generations
per night. `python -m qwen_rsi evolve` runs the same loop headless. Ways to speed it up:
- lower thinking levels in `config.json`,
- reduce `smoke_tasks`,
- set `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` to fit more context in VRAM.

Measure your own token rate with `ollama run qwen3.8:27b --verbose`. The 30–40 tokens/s
figure is an estimate.

## 8. Known limitations

- **Domain.** The built-in benchmark is Python coding. Chat improvements come only indirectly,
  through better prompts and procedures. To optimise for your own work, add tasks to
  `%LOCALAPPDATA%\QwenRSI\tasks\*.json`:
  `{"id", "entry_point", "prompt", "checks": [python boolean expressions], "split": "dev"|"holdout"}`.
- **Noise.** Scores come from sampled generations. A variant can be promoted by luck within the
  margin. Re-run or roll back from the archive (any variant can be made champion).
- **Hold-out exhaustion.** After many promotions the hold-out set itself gets overfit through
  selection. Rotate in fresh hold-out tasks periodically.
- **No weight updates.** A natural extension is to collect the traces of successful runs and
  LoRA-fine-tune the model offline. That is out of scope for an Ollama-only app.

## 9. Verification

`pytest` runs 22 tests against a scripted fake Ollama server (`tests/fake_ollama.py`). They cover:
- **Benchmark:** every hidden check passes against the reference solutions (`tests/reference_solutions.py`) and fails against stubs.
- **Guard and sandbox:** escapes, timeouts, memory caps and environment scrubbing.
- **Grader:** tamper resistance.
- **Full loop:** seed → improve → staged eval → approval, plus auto-promotion, invalid proposals, guard violations, Ollama outages, and a check that hold-out tasks never leak into reports.
- **GUI:** the window drives a generation and both chat modes, offscreen.

The Windows-only code paths (Job Object, AppUserModelID, the PowerShell installer) were written
against the documented Win32 and PowerShell APIs, but this repository's tests run on Linux. Run
the suite on Windows before relying on them.
