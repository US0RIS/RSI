"""QLoRA fine-tuning job, run by the *trainer* Python environment (PyTorch + Unsloth),
not by the app itself.

    python train_lora.py check                 # report GPU / library status
    python train_lora.py train <job.json>      # train a LoRA adapter
    python train_lora.py merge <job.json>      # merge adapter into 16-bit safetensors (fallback import path)

Progress is printed as lines starting with "QRSI " followed by JSON. The final
result is written to <out_dir>/result.json.

Only the language layers are trained (vision layers frozen), the loss covers
only the assistant's answer (and its reasoning, when present), and the adapter
is saved in PEFT safetensors format so Ollama can apply it with ADAPTER.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


def emit(**kw) -> None:
    print("QRSI " + json.dumps(kw), flush=True)


# --------------------------------------------------------------------------- data (pure, unit-tested)
def build_example(messages: list[dict], thinking: str, render, encode, max_len: int, eos: str = ""):
    """Return (input_ids, labels) with the prompt masked out, or None if it doesn't fit.

    render(messages, add_generation_prompt) -> str applies the chat template;
    encode(text) -> list[int] tokenizes without adding special tokens.
    """
    prompt_msgs, answer = messages[:-1], dict(messages[-1])
    prompt_text = render(prompt_msgs, True)
    if thinking:
        answer["reasoning_content"] = thinking
    full_text = render(prompt_msgs + [answer], False)
    if not full_text.startswith(prompt_text) or (thinking and thinking.strip()[:200] not in full_text):
        # Template doesn't nest cleanly (or dropped the reasoning): build the completion by hand.
        body = (f"<think>\n{thinking.strip()}\n</think>\n\n" if thinking else "") + answer["content"]
        completion = body + (eos or "")
    else:
        completion = full_text[len(prompt_text):]
    p_ids, c_ids = encode(prompt_text), encode(completion)
    if not c_ids or len(p_ids) + len(c_ids) > max_len:
        return None
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


# --------------------------------------------------------------------------- modes
def check() -> int:
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1024 ** 3, 1)
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:  # noqa: BLE001 - report anything
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import unsloth  # noqa: F401
        info["unsloth"] = getattr(unsloth, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        info["unsloth_error"] = f"{type(e).__name__}: {e}"[:500]
    ok = info.get("cuda") and "unsloth" in info
    if ok and info.get("vram_gb", 0) < 22:
        info["warning"] = "QLoRA on the 27B model needs about 24 GB of VRAM"
    emit(event="check", ok=bool(ok), **info)
    return 0 if ok else 1


def _load(job: dict, adapter: str | None = None):
    from unsloth import FastModel
    model, proc = FastModel.from_pretrained(
        model_name=adapter or job["base_model"],
        max_seq_length=int(job["train"]["max_seq_length"]),
        load_in_4bit=True,
        full_finetuning=False,
    )
    tok = getattr(proc, "tokenizer", proc)
    return model, proc, tok


def _renderer(proc, tok):
    templ = tok if getattr(tok, "chat_template", None) else proc

    def render(msgs, add_generation_prompt):
        kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            return templ.apply_chat_template(msgs, enable_thinking=True, **kw)
        except TypeError:
            return templ.apply_chat_template(msgs, **kw)
    return render


def train(job_path: str) -> int:
    job = json.load(open(job_path, encoding="utf-8"))
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cfg = job["train"]
    emit(event="stage", stage="loading", model=job["base_model"])
    import torch
    from unsloth import FastModel  # must be imported before transformers
    from transformers import Trainer, TrainerCallback, TrainingArguments

    model, proc, tok = _load(job)
    model = FastModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        bias="none", random_state=int(job.get("seed", 3407)), use_gradient_checkpointing="unsloth",
    )
    render = _renderer(proc, tok)
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731

    emit(event="stage", stage="tokenizing")
    rows, skipped = [], 0
    with open(job["data_path"], encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            built = build_example(ex["messages"], ex.get("thinking", ""), render, encode,
                                  int(cfg["max_seq_length"]), tok.eos_token or "")
            if built is None:
                skipped += 1
                continue
            rows.append({"input_ids": built[0], "labels": built[1]})
    if not rows:
        raise RuntimeError("no training examples fit in max_seq_length")
    emit(event="data", examples=len(rows), skipped=skipped)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

    losses: list[float] = []

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append(float(logs["loss"]))
                emit(event="progress", step=state.global_step, max_steps=state.max_steps, loss=logs["loss"])

    bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=os.path.join(out, "checkpoints"),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        num_train_epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=cfg["lr_scheduler"],
        logging_steps=1,
        save_strategy="no",
        bf16=bf16, fp16=not bf16,
        optim="adamw_8bit",
        seed=int(job.get("seed", 3407)),
        report_to=[],
        remove_unused_columns=False,
    )
    emit(event="stage", stage="training", examples=len(rows))
    trainer = Trainer(model=model, args=args, train_dataset=Data(), data_collator=collate,
                      callbacks=[Progress()])
    trainer.train()

    adapter_dir = os.path.join(out, "adapter")
    emit(event="stage", stage="saving")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    result = {"ok": True, "adapter_dir": adapter_dir, "examples": len(rows), "skipped": skipped,
              "final_loss": losses[-1] if losses else None,
              "first_loss": losses[0] if losses else None, "seconds": round(time.time() - t0)}
    json.dump(result, open(os.path.join(out, "result.json"), "w", encoding="utf-8"), indent=2)
    emit(event="done", **result)
    return 0


def merge(job_path: str) -> int:
    job = json.load(open(job_path, encoding="utf-8"))
    out = job["out_dir"]
    adapter = os.path.join(out, "adapter")
    emit(event="stage", stage="merging")
    model, proc, tok = _load(job, adapter=adapter)
    merged = os.path.join(out, "merged")
    model.save_pretrained_merged(merged, proc, save_method="merged_16bit")
    res_path = os.path.join(out, "result.json")
    result = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {"ok": True}
    result["merged_dir"] = merged
    json.dump(result, open(res_path, "w", encoding="utf-8"), indent=2)
    emit(event="done", merged_dir=merged)
    return 0


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    try:
        if mode == "check":
            return check()
        if mode == "train":
            return train(argv[2])
        if mode == "merge":
            return merge(argv[2])
        emit(event="error", message=f"unknown mode {mode}")
        return 2
    except Exception as e:  # noqa: BLE001 - report everything to the app
        emit(event="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
