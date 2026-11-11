# Qwen RSI

A Windows 11 desktop app that runs **Qwen3.8 27B** locally through **Ollama** and makes the
agent around it **recursively self-improve**. The agent rewrites its own code, prompts and
self-improvement procedure. Each variant is tested against hidden tests in a sandbox, and a
variant is kept only if it does better on held-out tasks it never saw.

Design, research notes and trust boundaries: [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

- Windows 11
- An NVIDIA, AMD or Intel GPU with about 20 GB or more of VRAM recommended. `qwen3.8:27b` Q4_K_M is about 17 GB. With less VRAM it still runs, partly on the CPU and slowly.
- About 25 GB of free disk space

## Install

```powershell
git clone <this repo>; cd RSI
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The script uses `winget` to install Python 3.12 and Ollama if they are missing. It then:
- runs `ollama pull qwen3.8:27b`,
- creates a venv in `%LOCALAPPDATA%\QwenRSI`,
- adds a **Qwen RSI** Start menu shortcut,
- runs a connectivity self-check.

Manual install:

```powershell
winget install Ollama.Ollama
ollama pull qwen3.8:27b
py -3.12 -m pip install .
qwen-rsi
```

## The RSI button

The **RSI** tab has one button. Each round it:

1. lets the agent rewrite its own code and prompts for a few generations (as in the *Agent evolution* tab),
2. asks the **current model** to design its own next training recipe from the results of past rounds,
3. has the model write new coding problems; a problem is kept only if its reference solution passes its own
   checks in the sandbox, a do-nothing stub fails them, and it doesn't resemble any benchmark task,
4. has the model attempt each problem several times and keeps only answers that pass every check
   (plus replay of earlier rounds' data to limit forgetting),
5. fine-tunes the model on those answers with QLoRA (Unsloth) in a separate training environment,
6. imports the result into Ollama as `qwen3.8-rsi:rN` and probes it for sanity,
7. promotes the new weights only if the current champion agent scores higher with them on the hold-out tasks
   (and doesn't drop on dev). Audit-task scores are recorded but never used to decide.

Once promoted, the new weights do everything that follows: agent evolution, problem writing, recipe design and
data collection. That feedback loop is the recursive part. Rejected models are deleted from Ollama, and any
model can be reverted by setting it as the model in Settings.

**What to expect.** Each round is slow: at roughly 30–40 tokens/s, sampling hundreds of attempts with thinking
takes hours, and the fine-tune adds more. Plan on about one round per day on a single 24 GB GPU. Gains, if
any, will come in steps and shrink over rounds; there is no guarantee any round is promoted. If the agent
reaches 100% on the 12 hold-out tasks, new weights can no longer win, so add harder hold-out tasks in
`%LOCALAPPDATA%\QwenRSI\tasks\`.

**Training setup (once).** Training needs an NVIDIA GPU with about 24 GB of VRAM and about 40 GB of free disk:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_trainer.ps1
```

Then press **Check trainer** on the RSI tab. Without a working trainer, the button still evolves the agent and
saves verified training data (`%LOCALAPPDATA%\QwenRSI\selftrain\`) for later rounds.

## Use

- **Chat.** Talk to the model with live thinking output, or switch the mode to *Champion agent* to send your message through the current best self-improved genome.
- **Agent evolution.** The agent-only loop (no weight training). Press **Start**. The app evaluates the seed genome, then loops: pick a parent, let it rewrite itself, then score the child (smoke stage → dev stage → hold-out stage).
  - The chart and archive table show every variant.
  - The diff tab shows exactly what changed.
  - When a child beats the champion on the hold-out set, you get a Windows notification. Review the diff, then click **Approve as champion**. The same button rolls back to any earlier variant.
- **Settings.** Configure the Ollama URL and model, context size, thinking level, budgets and timeouts. You can also pull the model with a progress bar here.

### Headless (overnight runs)

```powershell
qwen-rsi-cli evolve -n 10     # 10 generations; omit -n to run until Ctrl+C
qwen-rsi-cli status           # archive table
qwen-rsi-cli approve g0007    # promote or roll back
qwen-rsi-cli check            # Ollama + model diagnostics
qwen-rsi-cli chat             # terminal chat with streaming thinking
```

### Your own tasks

Drop JSON files into `%LOCALAPPDATA%\QwenRSI\tasks\`:

```json
[{"id": "slugify", "entry_point": "slugify", "split": "dev",
  "prompt": "Write `slugify(s: str) -> str` that lowercases, ...",
  "checks": ["slugify('Hello World') == 'hello-world'", "slugify('') == ''"]}]
```

Put some tasks in `"split": "holdout"`. Only those decide promotions, so they keep the loop honest.

## Develop

```powershell
py -3.12 -m pip install -e .[dev]
pytest                  # 31 tests, using a scripted fake Ollama server and fake trainer; no GPU needed
python -m qwen_rsi      # run from source
```

## Safety notes

Model-written code runs in a separate process. On Windows that process runs inside a Job
Object, which caps memory and process count and kills everything when the job closes. The
process also gets a scrubbed environment and an AST import allowlist. LLM and code-execution
budgets are enforced outside the genome. Promotions need your approval by default.

This is process isolation, not a VM. For long unattended runs, run the app inside Windows
Sandbox or a VM. See DESIGN.md §4.
