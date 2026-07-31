---
name: slaivina-finetune
description: Use when setting up or running LoRA/QLoRA fine-tuning (continued pretraining or SFT) of the slaivina base model with Unsloth or mlx-lm.
---

# Fine-tuning

Reference: [PLAN.md — Fine-tuning](../../PLAN.md#fine-tuning).

## Steps
1. Confirm the current base model and hardware path per
   [PLAN.md — Target model selection](../../PLAN.md#target-model-selection-sota-small-open-weight)
   (NVIDIA GPU → Unsloth; CPU-only → plain `transformers`+`peft`+`bitsandbytes`;
   Apple Silicon → `mlx-lm`) — don't hardcode the model name here, it has
   already changed twice; PLAN.md is the single source of truth for it.
2. **Stage A – continued pretraining** (`training/stage_a_pretrain.py`):
   causal LM over `data/processed/pretrain.txt`. See
   [PLAN.md — Method](../../PLAN.md#method) for current epoch/LR guidance.
3. **Stage B – SFT** (`training/stage_b_sft.py`): chat-template training on
   `sft_train.jsonl`/`sft_val.jsonl`. See
   [PLAN.md — Method](../../PLAN.md#method) for current epoch/LR guidance.
4. QLoRA rank/alpha/target-module choices, sequence length, and batch size
   are also tracked in
   [PLAN.md — Method](../../PLAN.md#method) — update there, not here, when
   they change.
5. Merge adapters (`merge_and_unload`) before handing off to quantization.

## Guardrails
- Do not attempt full fine-tuning — corpus is small, will overfit/forget.
- Log loss/eval simply (CSV or W&B); this is a learning project, keep it light.
- Never commit checkpoints or adapter weights to git.
- If a vision-language checkpoint is ever chosen instead of the current
  text-only default, don't assume the vision tower is cleanly separable —
  some VL model families (e.g. Qwen3.5) use early fusion baked in at
  pretraining, unlike bolt-on adapter-style VL models. Confirm with
  Unsloth/the model card whether a text-only training path exists before
  committing to that base; see
  [PLAN.md — Target model selection](../../PLAN.md#target-model-selection-sota-small-open-weight)
  for the current reasoning behind the text-only choice.
