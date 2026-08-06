#!/usr/bin/env python3
"""Retrieval-augmented generation query CLI for the slaivina blog corpus.

See PLAN.md#rag-system and skills/rag/SKILL.md.

Retrieves the top-k most similar post chunks from the ChromaDB index built
by rag/build_index.py, then feeds them (plus the author persona and the
user's question) to the fine-tuned model -- served via a running
`llama-server` (llama.cpp) -- to produce a grounded, in-style answer. This
lets the model reference actual post content/dates/themes without needing
it memorized in weights (the fine-tune teaches *voice*, RAG supplies
*facts*/recall -- see PLAN.md's RAG intro).

Requires a llama.cpp server already running with the quantized model, e.g.:
    ~/tools/llama.cpp/build/bin/llama-server \\
        -m quantize/output/slaivina-4b-q4_k_m.gguf -c 4096 --port 8080

Usage
--------------------------------------------------------------------
    uv run rag/query.py "Cosa hai scritto sul mare?"

    # show retrieved chunks without generating (debug retrieval quality)
    uv run rag/query.py --retrieve-only "insonnia"

    # change how many chunks to retrieve, or point at a different server
    uv run rag/query.py -k 5 --server-url http://localhost:8080 "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "slaivina_posts"
DEFAULT_INDEX_DIR = Path("rag/index")
DEFAULT_SERVER_URL = "http://localhost:8080"

PERSONA = (
    "Sei l'autore del blog 'Come me su una slavina': scrivi in italiano, "
    "con tono intimo, immagini oniriche/malinconiche, frasi brevi. "
    "Usa i frammenti di post qui sotto come riferimento per rispondere in "
    "modo coerente col tuo blog, ma non citarli letteralmente a meno che "
    "non sia utile."
)


def embed_query_prefix(text: str) -> str:
    """See rag/build_index.py's embed_query_prefix -- e5 models want a
    "query: " prefix at search time (vs. "passage: " when indexing)."""
    return f"query: {text}"


def retrieve(question: str, index_dir: Path, k: int) -> list[dict]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(COLLECTION_NAME)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = embedder.encode(
        [embed_query_prefix(question)], normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n".join(f"- {c['text']}" for c in chunks)
    return f"Frammenti dal blog (per riferimento):\n{context}\n\nDomanda: {question}"


def generate(server_url: str, system_prompt: str, user_prompt: str) -> str:
    """Calls a running llama-server's OpenAI-compatible chat endpoint."""
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
            "top_p": 0.95,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(
            f"error: could not reach llama-server at {server_url}: {e}\n"
            "Is it running? See this script's module docstring for the "
            "launch command.",
            file=sys.stderr,
        )
        sys.exit(1)
    return body["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", help="the question/prompt to answer")
    parser.add_argument("-k", type=int, default=3, help="chunks to retrieve")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="print retrieved chunks and exit, skip generation "
        "(useful for debugging retrieval quality without a server running)",
    )
    args = parser.parse_args()

    if not args.index_dir.exists():
        print(
            f"error: index dir not found: {args.index_dir}\n"
            "Run rag/build_index.py first.",
            file=sys.stderr,
        )
        return 1

    chunks = retrieve(args.question, args.index_dir, args.k)

    print("Retrieved chunks:", file=sys.stderr)
    for c in chunks:
        title = c["metadata"].get("title", "")
        print(f"  [dist={c['distance']:.3f}] {title!r}: {c['text']}", file=sys.stderr)
    print(file=sys.stderr)

    if args.retrieve_only:
        return 0

    user_prompt = build_prompt(args.question, chunks)
    answer = generate(args.server_url, PERSONA, user_prompt)
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
