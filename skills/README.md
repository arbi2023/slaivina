# Skills

Each subdirectory is an [Agent Skill](https://docs.github.com/copilot) —
a self-contained `SKILL.md` (YAML frontmatter + instructions) that an AI
coding agent can load on demand for a specific project phase, instead of
carrying all of PLAN.md in context at once.

Convention: `skills/<name>/SKILL.md` with frontmatter

```yaml
---
name: slaivina-<name>
description: One sentence, third person, stating when to invoke this skill.
---
```

followed by concrete, imperative steps and the exact commands/paths for that
phase. Skills should reference PLAN.md sections for rationale rather than
duplicating it.

| Skill | Project phase (PLAN.md) |
|---|---|
| `bootstrap` | Meta-skill: bootstrapping agent docs/skills for this or a new repo (no PLAN.md section — see `SKILL.md`) |
| `data-prep` | [Data preparation](../PLAN.md#data-preparation) (scrape, clean, SFT pairs, splits) |
| `finetune` | [Fine-tuning](../PLAN.md#fine-tuning) (Unsloth QLoRA, stage A/B) |
| `quantize` | [Quantization](../PLAN.md#quantization) (GGUF conversion, quant levels) |
| `eval` | [Evaluation](../PLAN.md#evaluation) (perplexity, style similarity, human eval) |
| `rag` | [RAG system](../PLAN.md#rag-system-extending-content-knowledge-beyond-style) (index build, retrieval, generation) |
| `deploy` | [Deployment and serving](../PLAN.md#deployment-and-serving) (Ollama, llama.cpp server, UI) |

Status: scaffolds only — fill in real commands as each phase's scripts land
under `scripts/`, `training/`, `quantize/`, `eval/`, `rag/`, `deployment/`.
