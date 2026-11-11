"""The training recipe: the part of the weight-level loop the model evolves.

Every field has hard bounds enforced here (supervisor side), so a proposed
recipe can be bold but never unsafe or absurdly expensive.
"""
from __future__ import annotations

import copy
import json

DEFAULT_RECIPE: dict = {
    "task_gen": {
        "tasks_per_round": 120,
        "tasks_per_call": 5,
        "think": "high",
        "temperature": 0.9,
        "topics": [
            "string parsing and tokenizing with tricky edge cases",
            "dynamic programming with reconstruction or counting",
            "graph algorithms (shortest paths, components, ordering)",
            "interval, scheduling and sweep-line problems",
            "simulation of a precisely specified process or state machine",
            "number theory and exact arithmetic without floats",
            "data transformation of nested lists and dicts",
            "input validation that must raise specific exceptions",
        ],
        "difficulty": "hard enough that a strong engineer needs to think about edge cases; "
                      "solvable in under 60 lines of standard-library Python",
    },
    "sampling": {
        "attempts_per_task": 4,
        "think": "medium",
        "temperature": 0.8,
        "mode": "direct",          # direct = single model call; champion = full champion agent (distillation)
    },
    "filter": {
        "max_pass_rate": 0.75,     # drop tasks the model already solves nearly always
        "keep_per_task": 2,        # verified solutions kept per task
        "prefer_shortest": True,   # among correct answers, train on the most concise ones
        "use_reference_when_unsolved": True,
    },
    "data": {
        "include_thinking": True,
        "replay_fraction": 0.3,    # share of examples drawn from earlier rounds (limits forgetting)
        "max_examples": 1200,
    },
    "train": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "learning_rate": 1e-4,
        "epochs": 1,
        "max_seq_length": 8192,
        "batch_size": 1,
        "grad_accum": 8,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "lr_scheduler": "cosine",
    },
}

THINK = ["low", "medium", "high", "max"]
# (type, min, max) for numbers; ("enum", [...]); ("bool",); ("topics",); ("text", max_len)
BOUNDS: dict = {
    "task_gen": {
        "tasks_per_round": ("int", 10, 400), "tasks_per_call": ("int", 1, 8), "think": ("enum", THINK),
        "temperature": ("float", 0.2, 1.3), "topics": ("topics",), "difficulty": ("text", 400),
    },
    "sampling": {
        "attempts_per_task": ("int", 1, 12), "think": ("enum", ["off"] + THINK),
        "temperature": ("float", 0.1, 1.3), "mode": ("enum", ["direct", "champion"]),
    },
    "filter": {
        "max_pass_rate": ("float", 0.0, 1.0), "keep_per_task": ("int", 1, 6),
        "prefer_shortest": ("bool",), "use_reference_when_unsolved": ("bool",),
    },
    "data": {
        "include_thinking": ("bool",), "replay_fraction": ("float", 0.0, 0.8), "max_examples": ("int", 20, 5000),
    },
    "train": {
        "lora_r": ("int", 4, 128), "lora_alpha": ("int", 4, 256), "lora_dropout": ("float", 0.0, 0.3),
        "learning_rate": ("float", 1e-6, 5e-4), "epochs": ("int", 1, 4), "max_seq_length": ("int", 2048, 32768),
        "batch_size": ("int", 1, 8), "grad_accum": ("int", 1, 64), "warmup_ratio": ("float", 0.0, 0.3),
        "weight_decay": ("float", 0.0, 0.2), "lr_scheduler": ("enum", ["cosine", "linear", "constant"]),
    },
}


def _coerce(spec: tuple, value, default):
    kind = spec[0]
    try:
        if kind == "int":
            return max(spec[1], min(spec[2], int(value)))
        if kind == "float":
            return max(spec[1], min(spec[2], float(value)))
        if kind == "bool":
            return value if isinstance(value, bool) else default
        if kind == "enum":
            return value if value in spec[1] else default
        if kind == "text":
            return str(value)[: spec[1]] if value else default
        if kind == "topics":
            items = [str(t)[:200] for t in value if str(t).strip()][:16] if isinstance(value, list) else []
            return items or default
    except (TypeError, ValueError):
        return default
    return default


def clamp(recipe: dict | None, base: dict | None = None) -> dict:
    """Return a complete, in-bounds recipe; missing keys come from ``base``
    (default: DEFAULT_RECIPE), unknown keys are dropped."""
    out = copy.deepcopy(DEFAULT_RECIPE)
    if base is not None and base is not recipe:
        out = clamp(base)
    recipe = recipe if isinstance(recipe, dict) else {}
    for section, fields in BOUNDS.items():
        given = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
        for key, spec in fields.items():
            if key in given:
                out[section][key] = _coerce(spec, given[key], out[section][key])
    return out


def describe_bounds() -> str:
    lines = []
    for section, fields in BOUNDS.items():
        for key, spec in fields.items():
            if spec[0] in ("int", "float"):
                rng = f"{spec[1]}..{spec[2]}"
            elif spec[0] == "enum":
                rng = " | ".join(spec[1])
            elif spec[0] == "topics":
                rng = "list of up to 16 short strings"
            elif spec[0] == "text":
                rng = f"text up to {spec[1]} chars"
            else:
                rng = "true | false"
            lines.append(f"{section}.{key}: {rng}")
    return "\n".join(lines)


def diff(old: dict, new: dict) -> list[str]:
    changes = []
    for section in BOUNDS:
        for key in BOUNDS[section]:
            a, b = old.get(section, {}).get(key), new.get(section, {}).get(key)
            if a != b:
                changes.append(f"{section}.{key}: {json.dumps(a)} -> {json.dumps(b)}")
    return changes


MUTABLE = [
    ("train", "learning_rate", lambda v, r: v * r.choice([0.5, 2.0])),
    ("train", "lora_r", lambda v, r: v * r.choice([0.5, 2])),
    ("train", "epochs", lambda v, r: v + r.choice([-1, 1])),
    ("sampling", "attempts_per_task", lambda v, r: v + r.choice([-2, 2])),
    ("filter", "max_pass_rate", lambda v, r: v + r.choice([-0.15, 0.15])),
    ("data", "replay_fraction", lambda v, r: v + r.choice([-0.1, 0.1])),
    ("task_gen", "tasks_per_round", lambda v, r: v + r.choice([-40, 40])),
    ("filter", "keep_per_task", lambda v, r: v + r.choice([-1, 1])),
]


def mutate(recipe: dict, rng) -> tuple[dict, list[str]]:
    """Random exploration step used when the model's proposal repeats a failed recipe."""
    new = copy.deepcopy(recipe)
    for section, key, fn in rng.sample(MUTABLE, 2):
        new[section][key] = fn(new[section][key], rng)
    new = clamp(new, base=recipe)
    if new["train"]["lora_alpha"] < new["train"]["lora_r"]:
        new["train"]["lora_alpha"] = min(256, 2 * new["train"]["lora_r"])
    return new, diff(recipe, new)
