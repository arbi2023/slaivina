#!/usr/bin/env python3
"""Clean raw scraped/exported posts into the processed dataset shapes.

Turns `data/raw/posts.jsonl` (verbatim HTML bodies, one JSON record per
line -- see `scripts/scrape.py`) into:

  - `data/processed/posts.jsonl`  -- {slug, title, date, tags, text}, with
    `text` fully de-HTML'd, entity-unescaped, unicode-normalized, and
    paragraph breaks preserved as blank lines.
  - `data/processed/pretrain.txt` -- the same `text` fields concatenated,
    one post per blank-line-separated block, for continued pretraining.
    (Train/val splitting happens later, in `scripts/split_dataset.py` --
    this script only cleans and filters, it does not split.)

What gets filtered out, and why (see PLAN.md -- Cleaning for the corpus
reality-check this was based on):
  - **Image-only posts** (no visible text after stripping tags/whitespace)
    -- old Tumblr imports with no textual content to learn from.
  - **Legacy link-only stub posts** -- a handful of old Tumblr-import pages
    whose only visible text lives inside an `<a>` tag (e.g. a bare
    "autocit." cross-reference back to the original Tumblr post, sometimes
    alongside an "original photo" link) -- there is no actual aphorism text
    on these pages, just a cross-reference.
  - **Exact-duplicate text** (after normalization) -- the same aphorism
    republished under a different slug/date; keeps the earliest instance
    (by `published_at`) and drops the rest, so the training corpus doesn't
    over-weight reposts.

What is deliberately NOT done here:
  - No length-based filtering. The corpus is genuinely aphoristic (median
    post length ~11 words) -- brevity is the style, not noise to remove.
  - No synthetic instruction/prompt generation -- see
    PLAN.md -- Why no SFT stage. This script only ever produces the
    continued-pretraining shape.

Usage
--------------------------------------------------------------------
    python scripts/clean.py
    python scripts/clean.py --in data/raw/posts.jsonl \\
        --out-jsonl data/processed/posts.jsonl \\
        --out-pretrain data/processed/pretrain.txt

    # see what would be dropped and why, without writing any files:
    python scripts/clean.py --dry-run -v
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger("clean")

_BLOCK_TAGS = {"p", "blockquote", "div", "br", "li", "h1", "h2", "h3", "h4"}


@dataclasses.dataclass
class CleanStats:
    total: int = 0
    kept: int = 0
    empty: int = 0
    link_only_stub: int = 0
    duplicate: int = 0


def _is_link_only(soup: BeautifulSoup) -> bool:
    """True if every scrap of visible text lives inside an <a> tag.

    Legacy Tumblr-import posts sometimes carry no original aphorism at
    all -- just a cross-reference link (e.g. "autocit." or "autocit. and
    original photo") back to the source Tumblr post/photo. There's no
    prose to learn from on these pages, so they should be dropped like the
    image-only posts, not kept as junk one-liners.
    """
    text_outside_links = "".join(
        node for node in soup.find_all(string=True) if node.find_parent("a") is None
    )
    return not text_outside_links.strip()


def html_to_text(content_html: str) -> tuple[str, bool]:
    """Strip kg-card wrappers/images/tags, preserving paragraph breaks.

    Block-level tags (p, blockquote, br, ...) become paragraph breaks;
    everything else (images, embedded links, inline markup) is dropped
    down to its visible text, since only the words matter for a
    continued-pretraining corpus.

    Returns `(text, is_link_only_stub)` -- see `_is_link_only`.
    """
    soup = BeautifulSoup(content_html, "html.parser")

    # Ghost wraps card content in HTML comments (<!--kg-card-begin: html-->
    # ... <!--kg-card-end: html-->); these aren't rendered but BeautifulSoup
    # keeps them as Comment nodes -- drop them explicitly.
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    link_only_stub = _is_link_only(soup)

    # Images carry no text signal for this corpus (captions aren't used on
    # this blog) -- drop the elements entirely rather than emitting alt text.
    for img in soup.find_all(["img", "figure"]):
        img.decompose()

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")

    raw_text = soup.get_text()
    # NFC-normalize (fixes mixed-composition Unicode from copy/paste
    # sources) before whitespace collapsing.
    raw_text = unicodedata.normalize("NFC", raw_text)

    paragraphs = [
        re.sub(r"\s+", " ", para).strip()
        for para in re.split(r"\n\s*\n", raw_text)
    ]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs), link_only_stub


def clean_posts(raw_posts: list[dict]) -> tuple[list[dict], CleanStats]:
    stats = CleanStats(total=len(raw_posts))
    seen_text: dict[str, dict] = {}  # normalized text -> kept record
    cleaned: list[dict] = []

    for post in raw_posts:
        text, link_only_stub = html_to_text(post["content_html"])

        if not text:
            stats.empty += 1
            logger.debug("dropping empty (image-only) post: %s", post["slug"])
            continue

        if link_only_stub:
            stats.link_only_stub += 1
            logger.debug("dropping link-only stub post: %s", post["slug"])
            continue

        dedup_key = text.lower()
        existing = seen_text.get(dedup_key)
        if existing is not None:
            stats.duplicate += 1
            # Keep whichever instance was published first.
            existing_date = existing.get("date") or ""
            new_date = post.get("published_at") or ""
            if new_date and (not existing_date or new_date < existing_date):
                logger.debug(
                    "duplicate text: replacing %s with earlier %s",
                    existing["slug"],
                    post["slug"],
                )
                seen_text[dedup_key]["_replaced_by_earlier"] = True
            else:
                logger.debug(
                    "duplicate text: keeping %s over %s",
                    existing["slug"],
                    post["slug"],
                )
                continue

        record = {
            "slug": post["slug"],
            "title": unicodedata.normalize(
                "NFC", BeautifulSoup(post.get("title") or "", "html.parser").get_text()
            ).strip(),
            "date": post.get("published_at"),
            "tags": post.get("tags") or [],
            "text": text,
        }
        seen_text[dedup_key] = record

    cleaned = list(seen_text.values())
    for record in cleaned:
        record.pop("_replaced_by_earlier", None)

    # Sort by date (falling back to slug for posts with no parseable date)
    # so output order is stable/reproducible across runs.
    cleaned.sort(key=lambda r: (r["date"] or "", r["slug"]))
    stats.kept = len(cleaned)
    return cleaned, stats


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pretrain_txt(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(record["text"] for record in records))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in", dest="in_path", type=Path, default=Path("data/raw/posts.jsonl")
    )
    parser.add_argument(
        "--out-jsonl", type=Path, default=Path("data/processed/posts.jsonl")
    )
    parser.add_argument(
        "--out-pretrain", type=Path, default=Path("data/processed/pretrain.txt")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report stats, write nothing"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.in_path.exists():
        logger.error("input file not found: %s", args.in_path)
        return 1

    raw_posts = load_jsonl(args.in_path)
    cleaned, stats = clean_posts(raw_posts)

    logger.info(
        "total=%d kept=%d empty=%d link_only_stub=%d duplicate=%d",
        stats.total,
        stats.kept,
        stats.empty,
        stats.link_only_stub,
        stats.duplicate,
    )

    if args.dry_run:
        logger.info("dry run -- not writing any files")
        return 0

    write_jsonl(args.out_jsonl, cleaned)
    write_pretrain_txt(args.out_pretrain, cleaned)
    logger.info("wrote %s and %s", args.out_jsonl, args.out_pretrain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
