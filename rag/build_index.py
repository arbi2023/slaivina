#!/usr/bin/env python3
"""Build a ChromaDB index over the blog posts for retrieval-augmented
generation.

See PLAN.md#rag-system and skills/rag/SKILL.md.

Each post in data/processed/posts.jsonl is already a natural short chunk
(the corpus is mostly one-liners/short fragments -- see PLAN.md#data-
preparation), so we index one chunk per post rather than paragraph-
splitting. Embeddings are computed with a small multilingual model
(Italian-capable, CPU-friendly) and stored, along with post metadata, in a
persistent on-disk Chroma collection.

Usage
--------------------------------------------------------------------
    uv run rag/build_index.py

    # rebuild from scratch (e.g. after posts.jsonl changed)
    uv run rag/build_index.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "slaivina_posts"
DEFAULT_INDEX_DIR = Path("rag/index")
DEFAULT_POSTS_PATH = Path("data/processed/posts.jsonl")


def load_posts(posts_path: Path) -> list[dict]:
    posts = []
    with posts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    return posts


def embed_query_prefix(text: str) -> str:
    """e5 models are trained with an instruction prefix distinguishing
    "documents" being indexed from "queries" searching them -- using the
    right one measurably improves retrieval quality. See the model card:
    https://huggingface.co/intfloat/multilingual-e5-small
    """
    return f"passage: {text}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts-path", type=Path, default=DEFAULT_POSTS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate the collection instead of upserting into it",
    )
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    posts = load_posts(args.posts_path)
    if not posts:
        print(f"error: no posts found in {args.posts_path}", file=sys.stderr)
        return 1
    print(f"Loaded {len(posts)} posts from {args.posts_path}", file=sys.stderr)

    print(f"Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.index_dir))

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    ids = [p["slug"] for p in posts]
    documents = [p["text"] for p in posts]
    metadatas = [
        {
            "title": p.get("title", ""),
            "date": p.get("date", ""),
            "tags": ",".join(p.get("tags", [])),
        }
        for p in posts
    ]

    print("Embedding posts...", file=sys.stderr)
    embeddings = embedder.encode(
        [embed_query_prefix(d) for d in documents],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(
        f"Indexed {collection.count()} chunks into {args.index_dir} "
        f"(collection={COLLECTION_NAME!r})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
