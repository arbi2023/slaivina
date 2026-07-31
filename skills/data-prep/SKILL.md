---
name: slaivina-data-prep
description: Use when acquiring, cleaning, or building datasets from the "Come me su una slavina" blog for slaivina — scraping/export, HTML cleaning, synthetic SFT prompt generation, or train/val splitting.
---

# Data preparation

Reference: [PLAN.md — Data preparation](../../PLAN.md#data-preparation).

## Steps
1. **Acquire** raw content into `data/raw/` (untouched):
   - Preferred: Ghost content-export JSON (Admin → Labs → Export).
   - Fallback (current reality): `scripts/scrape.py` against a config in
     `configs/sites/` — see the script's module docstring and
     `configs/sites/README.md` for the query language, config schema, and
     how to adapt to a new site.
2. **Clean** with `scripts/clean.py`: strip HTML/`kg-card` markup, normalize
   unicode/whitespace, de-dupe near-identical posts. Output
   `data/processed/posts.jsonl` with `{slug, title, date, tags, text}`.
3. **Build training shapes**:
   - `data/processed/pretrain.txt` — concatenated raw bodies (continued-pretraining).
   - `scripts/make_sft_pairs.py` → `sft_train.jsonl`/`sft_val.jsonl` as
     `{prompt, response}`, generating the `prompt` synthetically per post
     (see [PLAN.md — Synthetic prompt generation](../../PLAN.md#synthetic-prompt-generation)
     for the persona/system-prompt framing).
4. **Split** with `scripts/split_dataset.py`: 90/10 or 95/5, stratified by
   year/tag if possible.

## Guardrails
- Never commit anything under `data/raw/` or `data/processed/` except a
  handful of illustrative sample rows.
- Keep prompts/persona text in Italian; keep script code/comments in English.
- Re-run cleaning idempotently — scripts should be safe to re-execute as new
  posts are published.
