# Agent instructions for slaivina

This file is auto-loaded as repo instructions by Copilot CLI and compatible
coding agents (Claude, Gemini, etc.). Keep it short and operational.
The full project spec, rationale, and roadmap live in **[PLAN.md](PLAN.md)** —
read that for background before making non-trivial changes.

## Project in one line
Fine-tune, quantize, and RAG-augment a small open-weight LLM to write in the
style of the Italian blog "Come me su una slavina". See PLAN.md for the full
phased plan (data prep → fine-tuning → quantization → eval → RAG → deploy).

## Repo layout (target — most dirs don't exist yet)
```
data/{raw,processed}/   # gitignored except small samples; raw is untouched source
scripts/                # scrape.py, clean.py, make_sft_pairs.py, split_dataset.py
training/               # stage_a_pretrain.py, stage_b_sft.py, configs/*.yaml
quantize/               # convert_and_quantize.sh
eval/                   # perplexity.py, style_similarity.py, EVAL.md
rag/                    # build_index.py, query.py, server.py
deployment/             # Modelfile, docker-compose.yml
skills/                 # Agent Skill scaffolds, one per project phase (see skills/README.md)
```

## Conventions
- Italian is the content language; keep prompts/personas/system text in Italian,
  keep code/comments/docs in English.
- Never commit raw scraped data, model weights, checkpoints, or GGUF files —
  `data/raw`, `data/processed`, and model output dirs must stay gitignored.
  Only small illustrative samples may be committed.
- Prefer LoRA/QLoRA over full fine-tuning (corpus is small; full FT overfits).
- Default base model: Qwen3-4B-Instruct-2507 (see
  [Target model selection](PLAN.md#target-model-selection-sota-small-open-weight)
  before switching).
- Python: `pyproject.toml` at repo root, managed with `uv` (`uv sync` to
  install, `uv run <cmd>` to execute); no separate `requirements.txt`.
- When a phase's scripts don't exist yet, scaffold them under the paths above
  rather than inventing new top-level directories.

## Git usage
- Only run read-only git commands (e.g. `status`, `diff`, `log`, `show`).
- Never run mutating git actions — no `commit`, `push`, `add`, `mv`, `rm`,
  `checkout`, `branch`, `merge`, `rebase`, `reset`, etc. — even if asked to
  stage or commit changes. Leave those actions to the user.

## Working agreement
- PLAN.md is a living spec — update it in the same change when you alter
  approach, tooling, or milestones, so it stays authoritative.
- Use `skills/` for reusable, phase-specific agent instructions (data prep,
  fine-tuning, quantization, eval, RAG, deployment). Add a new skill only when
  a phase has enough concrete, repeatable steps to justify it.
