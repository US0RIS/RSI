"""Weight-level self-improvement: the model generates verified training data,
proposes its own training recipe, is fine-tuned (LoRA) and re-imported into
Ollama, and the new weights are kept only if they beat the current ones on
hold-out tasks."""
