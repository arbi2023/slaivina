---
name: slaivina-eval
description: Use when measuring perplexity, style similarity, or running human/blind evaluation of slaivina model outputs across training stages or quantization levels.
---

# Evaluation

Reference: [PLAN.md — Evaluation](../../PLAN.md#evaluation).

## Steps
1. **Quick qualitative milestone check** (`eval/quick_style_check.py`): the
   cheapest sanity check — run this first, right after training, before
   quantization. Loads a model (merged fine-tune by default, or any HF
   repo/path via `--model` for comparison against base), builds the same
   persona + few-shot prompt used at inference time
   ([PLAN.md — Deployment and serving](../../PLAN.md#deployment-and-serving)),
   and generates a few short completions per seed word. Prints to stdout
   and appends to `eval/EVAL.md` by default (`--no-log` to skip).
   ```
   uv run eval/quick_style_check.py                       # fine-tuned merged model
   uv run eval/quick_style_check.py --model <base repo id>  # base, for comparison
   ```
   Not a substitute for perplexity/style-similarity below — just the
   fastest way to eyeball whether training moved anything before investing
   in quantization.
2. **Perplexity** (`eval/perplexity.py`): compute on held-out val text for
   base vs. fine-tuned vs. each quant level.
3. **Style similarity** (`eval/style_similarity.py`): embed generated
   samples and real posts (see
   [PLAN.md — Evaluation](../../PLAN.md#evaluation) for the current
   embedding model choice), compare cosine similarity distributions
   (generated-vs-corpus vs. base-model-vs-corpus).
4. **Human/blind eval**: generate N samples from base vs. fine-tuned given
   identical prompts; blind-rate which sound authentic.
5. Log everything qualitatively in `eval/EVAL.md` — side-by-side samples
   across training stages and quant levels. `quick_style_check.py` writes
   here automatically; add perplexity/style-similarity/blind-eval results
   here too as those scripts land.

## Guardrails
- Val set will be tiny (small corpus) — treat perplexity as a sanity check,
  not the primary signal; weight qualitative/style review heavily.
- Keep eval prompts fixed across comparisons so results are apples-to-apples.
