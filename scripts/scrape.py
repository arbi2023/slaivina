#!/usr/bin/env python3
"""Generic, config-driven scraper for Ghost/WordPress/Hugo-style blogs.

Rather than hard-coding one site's HTML structure, the crawl logic here is
generic: it walks a paginated listing, follows post permalinks, and pulls
fields out of each post page using a small CSS-selector-based query
language borrowed from Scrapy/parsel conventions. Adapting to a new site
should only require writing a new YAML config (see configs/sites/), not
touching this file.

Query mini-language (used for every selector string in the config)
--------------------------------------------------------------------
    "<css selector>::text"          -> element's visible text (stripped)
    "<css selector>::html"          -> element's inner HTML (unmodified)
    "<css selector>::attr(<name>)"  -> the given attribute of each match
    "<css selector>"                -> defaults to ::text

Each match-producing selector can return multiple values (one per matched
element); config fields are either "first non-empty selector wins" (title,
date, content) or "gather + dedupe every match across every selector"
(tags).

Config schema (YAML) -- see configs/sites/comemesuunaslavina.yaml
--------------------------------------------------------------------
    name: str                          # used only for logging
    base_url: str                      # for resolving relative links
    start_url: str                     # first listing page to fetch
    user_agent: str
    request_delay_seconds: float        # per-worker politeness delay
    timeout_seconds: float
    max_retries: int
    respect_robots_txt: bool
    date_dayfirst: bool                 # DD-MM-YYYY vs MM-DD-YYYY ambiguity
    max_workers: int                    # concurrent post-page fetches (also --workers)
    listing:
      post_link_selectors: [str, ...]   # permalinks found on a listing page
      next_page_selectors: [str, ...]   # tried in order per page
      max_pages: int                    # hard safety cap
    post:
      title: [str, ...]
      published_at: [str, ...]
      tags: [str, ...]
      content_html: [str, ...]

Output
--------------------------------------------------------------------
For each new post, writes:
  - <out-dir>/html/<slug>.html   the verbatim fetched post page (provenance)
  - one line appended to <out>   a JSON record: {url, slug, title,
    published_at (ISO 8601 or null), tags, content_html, scraped_at}

Re-running is resumable/idempotent: URLs already present in <out> are
skipped unless --refresh is passed.

Usage
--------------------------------------------------------------------
    python scripts/scrape.py --config configs/sites/comemesuunaslavina.yaml \\
        --out data/raw/posts.jsonl

    # quick smoke test against just a couple of new posts:
    python scripts/scrape.py --config configs/sites/comemesuunaslavina.yaml \\
        --out data/raw/posts.jsonl --limit 3 --dry-run

    # fetch post pages concurrently (listing pagination stays sequential;
    # only the many per-post fetches are parallelized). --workers overrides
    # the config's max_workers. Fine to push higher against small static
    # sites (e.g. S3/CloudFront) that can comfortably serve many
    # connections; lower it for sites that ask to be crawled gently.
    python scripts/scrape.py --config configs/sites/comemesuunaslavina.yaml \\
        --out data/raw/posts.jsonl --workers 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import ssl
import sys
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

logger = logging.getLogger("scrape")

_SELECTOR_RE = re.compile(r"^(?P<css>.+?)(::(?P<mode>text|html|attr)(\((?P<attr>[^)]+)\))?)?$")


@dataclasses.dataclass
class ListingConfig:
    post_link_selectors: list[str]
    next_page_selectors: list[str] = dataclasses.field(default_factory=list)
    max_pages: int = 200


@dataclasses.dataclass
class PostConfig:
    title: list[str]
    published_at: list[str]
    content_html: list[str]
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SiteConfig:
    name: str
    base_url: str
    start_url: str
    listing: ListingConfig
    post: PostConfig
    user_agent: str = "slaivina-scraper/0.1"
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 20.0
    max_retries: int = 3
    respect_robots_txt: bool = True
    date_dayfirst: bool = True
    max_workers: int = 1  # concurrent post-page fetches; see --workers

    @classmethod
    def from_yaml(cls, path: Path) -> "SiteConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["listing"] = ListingConfig(**raw["listing"])
        raw["post"] = PostConfig(**raw["post"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def parse_selector(expr: str) -> tuple[str, str, str | None]:
    """Split "<css>::mode(arg)" into (css, mode, arg). Defaults mode to text."""
    m = _SELECTOR_RE.match(expr.strip())
    if not m:
        raise ValueError(f"Unparseable selector: {expr!r}")
    css = m.group("css").strip()
    mode = m.group("mode") or "text"
    attr = m.group("attr")
    if mode == "attr" and not attr:
        raise ValueError(f"::attr(...) selector missing attribute name: {expr!r}")
    return css, mode, attr


def select_values(soup: BeautifulSoup, expr: str) -> list[str]:
    """Return every value matched by one selector expression."""
    css, mode, attr = parse_selector(expr)
    values: list[str] = []
    for el in soup.select(css):
        if mode == "attr":
            if el.has_attr(attr):
                v = el.get(attr)
                if isinstance(v, list):  # e.g. class-like multi-valued attrs
                    v = " ".join(v)
                values.append(v)
        elif mode == "html":
            values.append(el.decode_contents().strip())
        else:  # text
            text = el.get_text(strip=True)
            if text:
                values.append(text)
    return values


def first_nonempty(soup: BeautifulSoup, exprs: Iterable[str]) -> str | None:
    for expr in exprs:
        for value in select_values(soup, expr):
            if value and value.strip():
                return value.strip()
    return None


def all_values(soup: BeautifulSoup, exprs: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for expr in exprs:
        for value in select_values(soup, expr):
            v = value.strip()
            if v:
                seen.setdefault(v, None)
    return list(seen.keys())


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.endswith("index.html"):
        path = path[: -len("index.html")].strip("/")
    return path.rsplit("/", 1)[-1] or "index"


def resolve_ca_bundle() -> str | None:
    """Pick a CA bundle path from common env var conventions.

    Plain ``requests``/``httpx`` only auto-honor REQUESTS_CA_BUNDLE/
    CURL_CA_BUNDLE and otherwise pin to their bundled certifi store,
    ignoring the OS trust store. That breaks transparently behind
    corporate/SSL-inspection proxies that only set SSL_CERT_FILE (OpenSSL
    convention) or NODE_EXTRA_CA_CERTS/AWS_CA_BUNDLE. Check the common
    conventions ourselves so the scraper works out of the box in those
    environments.
    """
    for var in (
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "NODE_EXTRA_CA_CERTS",
        "AWS_CA_BUNDLE",
    ):
        path = os.environ.get(var)
        if path and Path(path).is_file():
            logger.debug("Using CA bundle from %s=%s", var, path)
            return path
    return None  # fall back to httpx/certifi's default verification


_AKID_ERROR_SUBSTRING = "Missing Authority Key Identifier"


def build_ssl_context(*, relaxed: bool = False) -> ssl.SSLContext:
    """Build a verifying SSL context, optionally relaxing RFC 5280 strictness.

    Some SSL-inspection proxies mint intermediate certificates that omit the
    Authority Key Identifier extension. OpenSSL 3.2+ enforces that field by
    default (``ssl.VERIFY_X509_STRICT``) and rejects the chain outright,
    even though older OpenSSL/LibreSSL builds (e.g. macOS/Homebrew clients)
    accept it and the chain is otherwise trusted. We still verify against
    the resolved CA bundle either way; ``relaxed`` only lifts that one
    strict-mode check, and is only used as a fallback after a normal
    handshake fails with that specific error (see ``RateLimitedSession``).
    """
    ctx = ssl.create_default_context(cafile=resolve_ca_bundle())
    if relaxed:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def _make_client(cfg: SiteConfig, *, relaxed: bool) -> httpx.Client:
    # Size the pool for cfg.max_workers concurrent post-page fetches, plus
    # slack for keep-alive reuse across the listing-page walk.
    limits = httpx.Limits(
        max_connections=max(cfg.max_workers * 2, 5),
        max_keepalive_connections=max(cfg.max_workers, 5),
    )
    return httpx.Client(
        headers={"User-Agent": cfg.user_agent},
        verify=build_ssl_context(relaxed=relaxed),
        follow_redirects=True,
        limits=limits,
    )


class RateLimitedSession:
    """Thin wrapper adding delay, retries, UA, and robots.txt checks.

    Safe to share across threads: ``get()`` only reads ``self.client``
    through a lock-guarded accessor, and TLS relaxation (see
    ``_relax_tls``) swaps it atomically under the same lock.
    """

    def __init__(self, cfg: SiteConfig):
        self.cfg = cfg
        self._relaxed_tls = False
        self._lock = threading.Lock()
        self.client = _make_client(cfg, relaxed=False)
        self._robots: urllib.robotparser.RobotFileParser | None = None
        if cfg.respect_robots_txt:
            robots_url = urljoin(cfg.base_url, "/robots.txt")
            try:
                # Fetch via our own client (correct CA bundle, UA, timeout)
                # rather than RobotFileParser.read()'s internal urllib call.
                # Note: self._robots must stay None for this fetch, since an
                # empty/unpopulated RobotFileParser denies can_fetch() by
                # default and would otherwise block its own fetch.
                resp = self.get(robots_url)
                if resp is not None and resp.status_code == 200:
                    self._robots = urllib.robotparser.RobotFileParser()
                    self._robots.parse(resp.text.splitlines())
                else:
                    logger.info(
                        "robots.txt at %s unavailable; allowing all URLs", robots_url,
                    )
                    self._robots = None
            except Exception as exc:  # noqa: BLE001 - robots.txt is best-effort
                logger.warning("Could not fetch %s (%s); allowing all URLs", robots_url, exc)
                self._robots = None

    def allowed(self, url: str) -> bool:
        if self._robots is None:
            return True
        return self._robots.can_fetch(self.cfg.user_agent, url)

    def _relax_tls(self) -> None:
        """Rebuild the client with strict X.509 checks lifted, once.

        Guarded by ``self._lock`` so concurrent worker threads hitting the
        same AKID error don't race to rebuild the client repeatedly.
        """
        with self._lock:
            if self._relaxed_tls:
                return
            logger.warning(
                "TLS handshake failed on Authority Key Identifier strictness; "
                "retrying with ssl.VERIFY_X509_STRICT relaxed (proxy-injected "
                "intermediate certs are often missing this extension)."
            )
            old_client = self.client
            self.client = _make_client(self.cfg, relaxed=True)
            self._relaxed_tls = True
        old_client.close()

    def get(self, url: str) -> httpx.Response | None:
        if not self.allowed(url):
            logger.warning("Skipping %s (disallowed by robots.txt)", url)
            return None
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                client = self.client  # stable local ref even if _relax_tls swaps it mid-loop
                resp = client.get(url, timeout=self.cfg.timeout_seconds)
                resp.raise_for_status()
                time.sleep(self.cfg.request_delay_seconds)
                return resp
            except httpx.HTTPStatusError as exc:
                # 4xx (other than 429 Too Many Requests) is a definitive
                # answer, not a transient failure -- retrying just wastes
                # time waiting on exponential backoff for a result that
                # will never change.
                if exc.response.status_code != 429:
                    logger.warning("Fetch failed for %s: %s", url, exc)
                    return None
                last_exc = exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if not self._relaxed_tls and _AKID_ERROR_SUBSTRING in str(exc):
                    self._relax_tls()
                    continue  # retry immediately with the relaxed client
            wait = self.cfg.request_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Fetch failed (%s/%s) for %s: %s -- retrying in %.1fs",
                attempt, self.cfg.max_retries, url, last_exc, wait,
            )
            time.sleep(wait)
        logger.error("Giving up on %s: %s", url, last_exc)
        return None


def parse_date(value: str | None, dayfirst: bool) -> str | None:
    if not value:
        return None
    try:
        dt = dateparser.parse(value, dayfirst=dayfirst)
    except (ValueError, OverflowError):
        logger.warning("Could not parse date %r", value)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def load_seen_urls(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def extract_post_links(soup: BeautifulSoup, page_url: str, cfg: ListingConfig) -> list[str]:
    links = []
    for expr in cfg.post_link_selectors:
        for href in select_values(soup, expr):
            links.append(urljoin(page_url, href))
    # de-dup, preserve order
    return list(dict.fromkeys(links))


def find_next_page(soup: BeautifulSoup, page_url: str, cfg: ListingConfig) -> str | None:
    for expr in cfg.next_page_selectors:
        values = select_values(soup, expr)
        if values:
            return urljoin(page_url, values[0])
    return None


def scrape_post(client: RateLimitedSession, cfg: SiteConfig, url: str) -> dict | None:
    resp = client.get(url)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    content_html = first_nonempty(soup, cfg.post.content_html)
    if not content_html:
        logger.warning("No content found at %s (selectors may need updating); skipping", url)
        return None
    return {
        "url": url,
        "slug": slug_from_url(url),
        "title": first_nonempty(soup, cfg.post.title),
        "published_at": parse_date(
            first_nonempty(soup, cfg.post.published_at), cfg.date_dayfirst
        ),
        "tags": all_values(soup, cfg.post.tags),
        "content_html": content_html,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "_raw_html": resp.text,
    }


def run(cfg: SiteConfig, out_path: Path, limit: int | None, refresh: bool, dry_run: bool) -> None:
    client = RateLimitedSession(cfg)
    seen = set() if refresh else load_seen_urls(out_path)
    submitted: set[str] = set()  # in-flight or already-completed this run
    html_dir = out_path.parent / "html"
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)

    page_url: str | None = cfg.start_url
    visited_pages: set[str] = set()
    pages_fetched = 0
    new_posts = 0
    skipped_existing = 0
    errors = 0

    out_fh = None
    if not dry_run:
        out_fh = out_path.open("a", encoding="utf-8")

    futures: dict = {}

    def handle_result(post_url: str, record: dict | None) -> None:
        nonlocal new_posts, errors
        if record is None:
            errors += 1
            return
        new_posts += 1
        if dry_run:
            logger.info("[dry-run] %s -> %r", post_url, record["title"])
            return
        raw_html = record.pop("_raw_html")
        (html_dir / f"{record['slug']}.html").write_text(raw_html, encoding="utf-8")
        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_fh.flush()

    def drain(*, block: bool) -> None:
        """Collect completed fetches and write their results out.

        Writes happen only on this (main) thread, so no lock is needed for
        the output file or the html cache directory.
        """
        if not futures:
            return
        done = list(futures.keys()) if block else [f for f in futures if f.done()]
        for fut in done:
            post_url = futures.pop(fut)
            try:
                handle_result(post_url, fut.result())
            except Exception as exc:  # noqa: BLE001 - keep crawling on one bad post
                logger.error("Unexpected error fetching %s: %s", post_url, exc)
                errors += 1

    try:
        with ThreadPoolExecutor(max_workers=max(cfg.max_workers, 1)) as pool:
            while page_url and pages_fetched < cfg.listing.max_pages:
                if page_url in visited_pages:
                    logger.info("Pagination loop detected at %s; stopping", page_url)
                    break
                visited_pages.add(page_url)

                resp = client.get(page_url)
                pages_fetched += 1
                if resp is None:
                    errors += 1
                    logger.error("Aborting: could not fetch listing page %s", page_url)
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                post_urls = extract_post_links(soup, page_url, cfg.listing)
                logger.info("Page %d (%s): %d post link(s)", pages_fetched, page_url, len(post_urls))

                limit_reached = False
                for post_url in post_urls:
                    in_flight_total = new_posts + len(futures)
                    if limit is not None and in_flight_total >= limit:
                        limit_reached = True
                        break
                    if post_url in seen or post_url in submitted:
                        skipped_existing += 1
                        continue
                    submitted.add(post_url)
                    futures[pool.submit(scrape_post, client, cfg, post_url)] = post_url

                drain(block=False)  # opportunistically write finished posts between pages
                next_url = find_next_page(soup, page_url, cfg.listing)
                if limit_reached:
                    logger.info("Reached --limit %d new posts (in flight); stopping", limit)
                    break
                page_url = next_url

            logger.info("Waiting for %d in-flight fetch(es) to finish...", len(futures))
            drain(block=True)
    finally:
        if out_fh:
            out_fh.close()

    logger.info(
        "Done. pages=%d new_posts=%d skipped_existing=%d errors=%d",
        pages_fetched, new_posts, skipped_existing, errors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a site YAML config")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path (e.g. data/raw/posts.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N new posts (for smoke-testing)")
    parser.add_argument("--refresh", action="store_true", help="Re-scrape URLs already present in --out")
    parser.add_argument("--dry-run", action="store_true", help="Crawl and parse but don't write anything")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Concurrent post-page fetches; overrides the config's max_workers",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = SiteConfig.from_yaml(args.config)
    if args.workers is not None:
        cfg.max_workers = args.workers
    logger.info(
        "Scraping %s starting at %s (workers=%d)", cfg.name, cfg.start_url, cfg.max_workers,
    )
    run(cfg, args.out, args.limit, args.refresh, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
