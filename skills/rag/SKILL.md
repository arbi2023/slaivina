---
name: slaivina-rag
description: Use when building or querying the slaivina retrieval-augmented generation index over blog posts (chunking, embeddings, vector store, retrieval-grounded generation).
---

# RAG system

Reference: [PLAN.md — RAG system](../../PLAN.md#rag-system-extending-content-knowledge-beyond-style).

## Steps
1. **Chunk**: each post is a natural chunk; paragraph-split longer posts.
2. **Embed**: `intfloat/multilingual-e5-small` or
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. **Index** (`rag/build_index.py`): ChromaDB (default, embedded/file-based)
   or FAISS; embeds `data/processed/posts.jsonl`.
4. **Query** (`rag/query.py`): top-k (3–5) similarity search, optional
   `bge-reranker-base` re-ranking, then feed retrieved chunks + persona
   system prompt + user query to the fine-tuned model.
5. Optional `rag/server.py`: thin FastAPI wrapper for the same flow.

## Guardrails
- Keep orchestration hand-rolled/plain Python — this is a learning project,
  avoid pulling in a heavy framework unless it teaches something new.
- Rebuild the index whenever `posts.jsonl` changes (new posts published).
