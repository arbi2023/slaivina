---
name: slaivina-rag
description: Use when building or querying the slaivina retrieval-augmented generation index over blog posts (chunking, embeddings, vector store, retrieval-grounded generation).
---

# RAG system

Reference: [PLAN.md — RAG system](../../PLAN.md#rag-system-extending-content-knowledge-beyond-style).

## Steps
1. **Chunk**: each post is a natural chunk; paragraph-split longer posts.
2. **Embed**: see
   [PLAN.md — RAG pipeline](../../PLAN.md#rag-pipeline) for the current
   embedding model choice.
3. **Index** (`rag/build_index.py`): embeds `data/processed/posts.jsonl`
   into the vector store named in
   [PLAN.md — RAG pipeline](../../PLAN.md#rag-pipeline).
4. **Query** (`rag/query.py`): top-k similarity search, optional re-ranking
   (see [PLAN.md — RAG pipeline](../../PLAN.md#rag-pipeline) for current
   k/re-ranker choice), then feed retrieved chunks + persona system prompt +
   user query to the fine-tuned model.
5. Optional `rag/server.py`: thin FastAPI wrapper for the same flow.

## Guardrails
- Keep orchestration hand-rolled/plain Python — this is a learning project,
  avoid pulling in a heavy framework unless it teaches something new.
- Rebuild the index whenever `posts.jsonl` changes (new posts published).
- Don't hardcode specific embedding/vector-store/re-ranker model names in
  this file — they've already been revised once in PLAN.md; link to
  [PLAN.md — RAG pipeline](../../PLAN.md#rag-pipeline) instead so this
  skill can't drift out of sync with it.
