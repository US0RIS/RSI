"""Entry points.

    qwen-rsi                       launch the desktop app
    python -m qwen_rsi check       verify Ollama + model
    python -m qwen_rsi evolve -n 5 run generations headless (overnight runs)
    python -m qwen_rsi status      print the archive
    python -m qwen_rsi approve ID  make a variant the champion (also rollback)
    python -m qwen_rsi chat        terminal chat with streaming
"""
from __future__ import annotations

import argparse
import sys


def _cmd_check(args, s) -> int:
    from .ollama_client import OllamaClient, OllamaError
    c = OllamaClient(s.ollama_url, 30)
    try:
        print(f"Ollama {c.version()} at {s.ollama_url}")
        if not c.has_model(s.model):
            print(f"Model {s.model} is not pulled. Run:  ollama pull {s.model}")
            return 1
        caps = c.capabilities(s.model)
        print(f"Model {s.model}: capabilities={caps}, max context={c.context_length(s.model)}")
        missing = {"tools", "thinking"} - set(caps)
        if missing:
            print(f"warning: model lacks {sorted(missing)}; the loop still works but genomes cannot use them")
        r = c.chat(s.model, [{"role": "user", "content": "Reply with the single word: ready"}],
                   think=False, options={"num_ctx": 4096, "num_predict": 16})
        print(f"Model replied: {r.content.strip()!r}")
        return 0
    except OllamaError as e:
        print(f"error: {e}")
        return 1


def _cmd_evolve(args, s) -> int:
    from .evolve import Evolver, InfraError
    ev = Evolver(s, on_event=lambda kind, msg: print(f"[{kind}] {msg}", flush=True))
    try:
        ev.run(generations=args.generations or None)
    except InfraError:
        return 2
    except KeyboardInterrupt:
        ev.stop.set()
        print("stopped")
    return 0


def _cmd_status(args, s) -> int:
    from .archive import Archive
    from .config import data_dir
    a = Archive(data_dir())
    fmt = "{:<7} {:<7} {:>3} {:<10} {:>6} {:>6} {:>7} {:>5}  {}"
    print(fmt.format("id", "parent", "gen", "status", "dev", "hold", "tokens", "flag", "rationale"))
    for v in a.all():
        f = lambda x: "-" if x is None else f"{x:.3f}"
        flag = "CHAMP" if v.is_champion else ("ASK" if v.awaiting_approval else "")
        print(fmt.format(v.id, v.parent_id or "-", v.generation, v.status, f(v.dev), f(v.holdout),
                         v.prompt_tokens + v.output_tokens, flag, v.rationale.splitlines()[0][:60] if v.rationale else ""))
    return 0


def _cmd_approve(args, s) -> int:
    from .archive import Archive
    from .config import data_dir
    a = Archive(data_dir())
    a.get(args.id)
    a.set_champion(args.id)
    a.log("champion", f"{args.id} set as champion from the command line")
    print(f"{args.id} is now the champion")
    return 0


def _cmd_chat(args, s) -> int:
    from .ollama_client import OllamaClient
    c = OllamaClient(s.ollama_url, s.request_timeout_s)
    history: list[dict] = []
    print(f"Chatting with {s.model} (think={s.chat_think}). Empty line to quit.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        history.append({"role": "user", "content": line})
        state = {"thinking": False}

        def on_token(kind: str, text: str) -> None:
            if kind == "thinking" and not state["thinking"]:
                state["thinking"] = True
                print("\x1b[2m", end="")
            if kind == "content" and state["thinking"]:
                state["thinking"] = False
                print("\x1b[0m\n", end="")
            print(text, end="", flush=True)
        r = c.chat(s.model, history, think=s.think_value(), options={"num_ctx": s.num_ctx},
                   keep_alive=s.keep_alive, on_token=on_token)
        print("\x1b[0m")
        history.append({"role": "assistant", "content": r.content})
    return 0


def main(argv: list[str] | None = None) -> int:
    from .config import Settings
    p = argparse.ArgumentParser(prog="qwen-rsi", description="Recursive self-improvement on Qwen3.8 via Ollama")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("gui")
    sub.add_parser("check")
    e = sub.add_parser("evolve")
    e.add_argument("-n", "--generations", type=int, default=0, help="0 = run until Ctrl+C")
    sub.add_parser("status")
    a = sub.add_parser("approve")
    a.add_argument("id")
    sub.add_parser("chat")
    args = p.parse_args(argv)
    s = Settings.load()
    if args.cmd in (None, "gui"):
        return gui_main()
    return {"check": _cmd_check, "evolve": _cmd_evolve, "status": _cmd_status,
            "approve": _cmd_approve, "chat": _cmd_chat}[args.cmd](args, s)


def gui_main() -> int:
    from .gui.app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
