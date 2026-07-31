---
name: slaivina-quantize
description: Use when converting a merged slaivina checkpoint to GGUF (or AWQ/GPTQ) and producing quantized model variants for commodity hardware.
---

# Quantization

Reference: [PLAN.md — Quantization](../../PLAN.md#quantization).

## Steps
1. Convert the merged HF model directory to GGUF with `llama.cpp`'s
   `convert_hf_to_gguf.py`.
2. Quantize with `llama-quantize` to at least:
   - `Q4_K_M` (default balance),
   - `Q5_K_M` (higher quality),
   - `Q8_0` (near-lossless baseline for comparison).
3. Optionally export AWQ/GPTQ 4-bit variants for comparison.
4. Wire this up in `quantize/convert_and_quantize.sh <merged-model-dir>`.
5. Record tokens/sec and perplexity per quant level (feeds into the
   `slaivina-eval` skill) so size/quality tradeoffs are concrete.

## Guardrails
- If training happened via `mlx-lm` on Apple Silicon, merge back into a
  standard HF checkpoint first — `convert_hf_to_gguf.py` expects that format
  regardless of training framework.
- Name outputs `slaivina-<size>-<quant>.gguf` using the base model's size
  (e.g. `slaivina-4b-q4_k_m.gguf` for the current default — see
  [PLAN.md — Target model selection](../../PLAN.md#target-model-selection-sota-small-open-weight)),
  not a hardcoded size that can go stale if the base model changes.
- Don't commit GGUF/quantized weights to git; ship via release assets or HF Hub.
