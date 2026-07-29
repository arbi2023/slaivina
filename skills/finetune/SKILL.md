---
name: slaivina-finetune
description: Use when setting up or running LoRA/QLoRA fine-tuning (continued pretraining or SFT) of the slaivina base model with Unsloth or mlx-lm.
---

# Fine-tuning

Reference: [PLAN.md — Fine-tuning](../../PLAN.md#fine-tuning).

## Steps
1. Confirm base model (default `Qwen3-4B-Instruct-2507`, see
   [PLAN.md — Target model selection](../../PLAN.md#target-model-selection-sota-small-open-weight)) and
   hardware path: NVIDIA GPU → Unsloth; CPU-only → plain
   `transformers`+`peft`+`bitsandbytes`; Apple Silicon → `mlx-lm`.
2. **Stage A – continued pretraining** (`training/stage_a_pretrain.py`):
   causal LM over `data/processed/pretrain.txt`, 1–3 epochs, LR ~1e-4.
3. **Stage B – SFT** (`training/stage_b_sft.py`): chat-template training on
   `sft_train.jsonl`/`sft_val.jsonl`, 3–5 epochs, LR ~2e-4.
4. QLoRA config: 4-bit nf4 base, LoRA rank 16–32 / alpha 32–64, targets
   `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
5. Sequence length 512–1024; batch size 1–4 with gradient accumulation to
   fit laptop VRAM/RAM.
6. Merge adapters (`merge_and_unload`) before handing off to quantization.

## Guardrails
- Do not attempt full fine-tuning — corpus is small, will overfit/forget.
- Log loss/eval simply (CSV or W&B); this is a learning project, keep it light.
- Never commit checkpoints or adapter weights to git.
- If a vision-language checkpoint is ever chosen instead (e.g. Qwen3.5-2B),
  don't assume the vision tower is cleanly separable — Qwen3.5 uses early
  fusion, baked in at pretraining, unlike bolt-on adapter-style VL models.
  Confirm with Unsloth/the model card whether a text-only training path
  exists before committing to that base.
