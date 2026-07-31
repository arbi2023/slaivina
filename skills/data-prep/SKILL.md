---
name: slaivina-data-prep
description: Use when acquiring, cleaning, or building the continued-pretraining dataset from the "Come me su una slavina" blog for slaivina — scraping/export, HTML cleaning, or train/val splitting.
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
   unicode/whitespace, de-dupe near-identical posts, drop image-only posts
   (no text signal). Output `data/processed/posts.jsonl` with
   `{slug, title, date, tags, text}` and `data/processed/pretrain.txt`
   (concatenated bodies for continued pretraining — the only dataset shape
   built; see
   [PLAN.md — Why no SFT stage](../../PLAN.md#why-no-sft-stage) for why
   there are no synthetic instruction/response pairs).
3. **Split** with `scripts/split_dataset.py`: plain random split with a
   fixed seed (tags are too sparse to stratify by — 14/403 posts in the
   real corpus) into `pretrain_train.txt`/`pretrain_val.txt`.

## Guardrails
- Never commit anything under `data/raw/` or `data/processed/` except a
  handful of illustrative sample rows.
- Keep prompts/persona text in Italian; keep script code/comments in English.
- Re-run cleaning idempotently — scripts should be safe to re-execute as new
  posts are published.
- Don't add synthetic instruction/prompt generation without re-reading
  [PLAN.md — Why no SFT stage](../../PLAN.md#why-no-sft-stage) first — it
  was deliberately rejected for this corpus (spontaneous, non-dialogic
  aphorisms), not merely deferred.
