# Site configs for `scripts/scrape.py`

Each YAML file here describes how to crawl one blog-like site: where to start,
how to find the "next page" of an index/listing, which link on a listing page
is a post permalink, and which CSS-selector-ish "queries" pull the title,
date, tags, and body out of a post page.

See `comemesuunaslavina.yaml` for a fully worked example (the source blog for
this project) and `scripts/scrape.py`'s module docstring for the full query
mini-language and config schema reference.

To adapt to a new site: copy an existing config, open one listing page and
one post page in "view source", and update the selectors to match. No code
changes should be needed for typical Ghost/WordPress/Hugo-style blogs.
