---
name: slaivina-eval
description: Use when measuring perplexity, style similarity, or running human/blind evaluation of slaivina model outputs across training stages or quantization levels.
---

# Evaluation

Reference: [PLAN.md — Evaluation](../../PLAN.md#evaluation).

## Steps
1. **Perplexity** (`eval/perplexity.py`): compute on held-out val text for
   base vs. fine-tuned vs. each quant level.
2. **Style similarity** (`eval/style_similarity.py`): embed generated
   samples and real posts (see
   [PLAN.md — Evaluation](../../PLAN.md#evaluation) for the current
   embedding model choice), compare cosine similarity distributions
   (generated-vs-corpus vs. base-model-vs-corpus).
3. **Human/blind eval**: generate N samples from base vs. fine-tuned given
   identical prompts; blind-rate which sound authentic.
4. Log everything qualitatively in `eval/EVAL.md` — side-by-side samples
   across training stages and quant levels.

## Guardrails
- Val set will be tiny (small corpus) — treat perplexity as a sanity check,
  not the primary signal; weight qualitative/style review heavily.
- Keep eval prompts fixed across comparisons so results are apples-to-apples.
