---
name: slaivina-deploy
description: Use when packaging or serving the slaivina model locally via Ollama or llama.cpp server for unconditional/seeded fragment generation (not a chatbot).
---

# Deployment and serving

Reference: [PLAN.md — Deployment and serving](../../PLAN.md#deployment-and-serving).

The target experience is **generation of style-matched fragments**, not
conversation — triggered like a "shuffle" action (empty/persona-only prompt,
or a short seed word/phrase), not multi-turn chat. See
[PLAN.md — Deployment and serving](../../PLAN.md#deployment-and-serving) for
the unconditional-vs-seeded-sampling framing and why there's no SFT/chat
stage behind it ([Why no SFT stage](../../PLAN.md#why-no-sft-stage)).

## Steps
1. **Ollama** (simplest): write `deployment/Modelfile` wrapping the GGUF +
   persona system prompt, then `ollama create slaivina -f Modelfile` and
   `ollama run slaivina`.
2. **llama.cpp server**: run `llama-server` for an OpenAI-compatible local
   API — the underlying chat-completions endpoint is just plumbing; the
   CLI/UI built on top should expose a single generate action, not a chat
   transcript.
3. **UI** (optional, pick one): `open-webui` pointed at Ollama, or a minimal
   Gradio/Streamlit app exposing a "generate" button (optionally with a seed
   text field) rather than a chat box.

## Guardrails
- Keep the persona system prompt in Italian, consistent across
  training-time few-shot context and inference-time serving.
- `deployment/docker-compose.yml` (if added) should be optional, not required
  for local dev.
- Docker (if used) is for the *serving stack* only (`llama-server`/Ollama +
  RAG API) — model weights are distributed via Hugging Face Hub, not baked
  into an image. See
  [PLAN.md — Docker versus Hugging Face Hub for distribution](../../PLAN.md#docker-versus-hugging-face-hub-for-distribution).
- CI/CD automation (scheduled scraping, build/quantize/publish pipelines) is
  a planned-but-not-started idea — see
  [PLAN.md — Future CI and CD plans](../../PLAN.md#future-ci-and-cd-plans-not-started);
  don't start implementing it without being asked.
- The RAG pipeline's "question → grounded answer" framing doesn't fit this
  non-chatbot experience as-is — don't wire `rag/query.py` into serving
  without first re-reading the reconsideration note in
  [PLAN.md — Deployment and serving](../../PLAN.md#deployment-and-serving).
