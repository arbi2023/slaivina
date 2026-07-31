#!/usr/bin/env python3
"""Split the cleaned corpus into train/val pretraining text files.

Reads `data/processed/posts.jsonl` (one JSON record per post, written by
`scripts/clean.py`) and writes a plain random train/val split of the
`text` field:

  - `data/processed/pretrain_train.txt`
  - `data/processed/pretrain_val.txt`

Why read `posts.jsonl` rather than re-parsing `pretrain.txt`: posts can
contain internal blank-line paragraph breaks (e.g. the corpus's dialogue-
style "A: ... \n\n B: ..." posts), which are visually indistinguishable
from the blank line *between* posts in the concatenated text file.
Splitting `pretrain.txt` back apart on blank lines would silently
fragment those multi-paragraph posts into separate "documents". The
JSONL file has one record per post, so it's the unambiguous source of
post boundaries; `pretrain.txt` itself is a derived, human-inspectable
concatenation only (not meant to be re-parsed).

Why plain random, not stratified: see PLAN.md -- Splits & size
expectations. Tags are far too sparse in this corpus (14/403 raw posts) to
stratify by, and there's no other reliable grouping axis (year-stratifying
a corpus this small just adds complexity for no real benefit). A fixed
`--seed` keeps the split reproducible across runs.

Usage
--------------------------------------------------------------------
    python scripts/split_dataset.py
    python scripts/split_dataset.py --val-fraction 0.05 --seed 42
    python scripts/split_dataset.py --in data/processed/posts.jsonl \\
        --out-train data/processed/pretrain_train.txt \\
        --out-val data/processed/pretrain_val.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("split_dataset")


def load_posts(path: Path) -> list[str]:
    """Read post `text` fields out of a cleaned posts.jsonl."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def split_posts(
    posts: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    shuffled = posts.copy()
    random.Random(seed).shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction))
    val_posts = shuffled[:n_val]
    train_posts = shuffled[n_val:]
    return train_posts, val_posts


def write_posts(path: Path, posts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(posts))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in", dest="in_path", type=Path, default=Path("data/processed/posts.jsonl")
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=Path("data/processed/pretrain_train.txt"),
    )
    parser.add_argument(
        "--out-val", type=Path, default=Path("data/processed/pretrain_val.txt")
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of posts held out for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="shuffle seed, for reproducibility"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.in_path.exists():
        logger.error(
            "input file not found: %s (run scripts/clean.py first)", args.in_path
        )
        return 1

    posts = load_posts(args.in_path)
    if len(posts) < 2:
        logger.error("need at least 2 posts to split, found %d", len(posts))
        return 1

    train_posts, val_posts = split_posts(posts, args.val_fraction, args.seed)

    write_posts(args.out_train, train_posts)
    write_posts(args.out_val, val_posts)

    logger.info(
        "total=%d train=%d val=%d (seed=%d)",
        len(posts),
        len(train_posts),
        len(val_posts),
        args.seed,
    )
    logger.info("wrote %s and %s", args.out_train, args.out_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
