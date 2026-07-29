---
name: slaivina-deploy
description: Use when packaging or serving the slaivina model locally via Ollama or llama.cpp server, or wiring up a chat UI.
---

# Deployment and serving

Reference: [PLAN.md — Deployment and serving](../../PLAN.md#deployment-and-serving).

## Steps
1. **Ollama** (simplest): write `deployment/Modelfile` wrapping the GGUF +
   persona system prompt, then `ollama create slaivina -f Modelfile` and
   `ollama run slaivina`.
2. **llama.cpp server**: run `llama-server` for an OpenAI-compatible local
   API — this is what `rag/query.py`/`rag/server.py` should call over HTTP.
3. **UI** (optional, pick one): `open-webui` pointed at Ollama, or a minimal
   Gradio/Streamlit app with a "RAG-grounded" vs "pure generation" toggle.

## Guardrails
- Keep the persona system prompt in Italian and consistent with the one used
  during SFT (see `slaivina-finetune` skill and
  [PLAN.md — Synthetic prompt generation](../../PLAN.md#synthetic-prompt-generation)).
- `deployment/docker-compose.yml` (if added) should be optional, not required
  for local dev.
