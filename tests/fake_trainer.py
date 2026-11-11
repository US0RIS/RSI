"""Stand-in for train_lora.py with the same CLI and output protocol (no GPU needed)."""
import json
import os
import sys


def emit(**kw):
    print("QRSI " + json.dumps(kw), flush=True)


def main():
    mode = sys.argv[1]
    if mode == "check":
        emit(event="check", ok=True, gpu="FakeGPU", vram_gb=24.0, unsloth="fake")
        return 0
    job = json.load(open(sys.argv[2], encoding="utf-8"))
    out = job["out_dir"]
    if os.environ.get("FAKE_TRAINER_FAIL"):
        emit(event="error", message="simulated CUDA out of memory")
        return 1
    if mode == "train":
        n = sum(1 for _ in open(job["data_path"], encoding="utf-8"))
        for step in range(1, 4):
            emit(event="progress", step=step, max_steps=3, loss=1.0 / step)
        adapter = os.path.join(out, "adapter")
        os.makedirs(adapter, exist_ok=True)
        open(os.path.join(adapter, "adapter_config.json"), "w").write(json.dumps({"r": job["train"]["lora_r"]}))
        res = {"ok": True, "adapter_dir": adapter, "examples": n, "first_loss": 1.0, "final_loss": 0.33, "seconds": 1}
    else:
        merged = os.path.join(out, "merged")
        os.makedirs(merged, exist_ok=True)
        res = json.load(open(os.path.join(out, "result.json")))
        res["merged_dir"] = merged
    json.dump(res, open(os.path.join(out, "result.json"), "w"))
    emit(event="done", **res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
