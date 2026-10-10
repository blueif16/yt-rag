"""
Ingest a curated manifest into one collection, each source with its own tags.

    .venv/bin/python scripts/ingest_manifest.py COLLECTION MANIFEST.json [MANIFEST.json ...]

A manifest is a list of entries:
    {"url", "kind": "video"|"page", "type": "tutorial"|"doc"|"article" (pages),
     "title", "author", "published", "text_file" (optional: saved Markdown for a
     page fetch_page cannot read), "tags": {topics, scenarios, level, impact, tools}}

Videos go one by one (each with its own tags). Pages are fetched per site in
one batch so same-site boilerplate is recognised; pages with a text_file go in
through ingest_text. Idempotent: re-running skips chunks already stored.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from yt_rag.ingest import ingest_videos
from yt_rag.taxonomy import validate_tags
from yt_rag.text_ingest import ingest_pages, ingest_text


def main() -> None:
    collection, paths = sys.argv[1], sys.argv[2:]
    entries = [e for p in paths for e in json.loads(Path(p).read_text())]
    for e in entries:  # fail fast on bad tags before any network traffic
        e["tags"] = validate_tags(e.get("tags"))

    report = {"videos": 0, "video_chunks": 0, "no_subs": 0, "pages": 0, "page_chunks": 0,
              "texts": 0, "text_chunks": 0, "errors": []}
    videos = [e for e in entries if e["kind"] == "video"]
    texts = [e for e in entries if e["kind"] == "page" and e.get("text_file")]
    pages = [e for e in entries if e["kind"] == "page" and not e.get("text_file")]

    for i, e in enumerate(videos, 1):
        print(f"\n=== video {i}/{len(videos)} {e['url']}", file=sys.stderr)
        try:
            r = ingest_videos([e["url"]], collection, tags=e["tags"])
            report["videos"] += 1
            report["video_chunks"] += r["created"]
            report["no_subs"] += r["no_subs"]
        except Exception as ex:
            report["errors"].append({"url": e["url"], "error": str(ex)})

    by_site = defaultdict(list)
    for e in pages:
        by_site[urlparse(e["url"]).netloc].append(e)
    for site, group in by_site.items():
        print(f"\n=== {len(group)} page(s) from {site}", file=sys.stderr)
        try:
            r = ingest_pages([e["url"] for e in group], collection,
                             type={e["url"]: e.get("type", "article") for e in group},
                             tags_by_url={e["url"]: e["tags"] for e in group})
            report["pages"] += r["pages"] - r["failed"]
            report["page_chunks"] += r["created"]
            report["errors"] += r["errors"]
        except Exception as ex:
            report["errors"].append({"url": site, "error": str(ex)})

    for e in texts:
        print(f"\n=== text {e['text_file']}", file=sys.stderr)
        try:
            r = ingest_text(collection, Path(e["text_file"]).read_text(), title=e.get("title", ""),
                            source_url=e["url"], author=e.get("author", ""),
                            published_at=e.get("published", ""), type=e.get("type", "doc"),
                            tags=e["tags"])
            report["texts"] += 1
            report["text_chunks"] += r["created"]
        except Exception as ex:
            report["errors"].append({"url": e["url"], "error": str(ex)})

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
