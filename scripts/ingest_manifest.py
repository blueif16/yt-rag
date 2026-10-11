"""
Ingest a curated manifest into one collection, each source with its own tags.

    .venv/bin/python scripts/ingest_manifest.py COLLECTION MANIFEST.json [MANIFEST.json ...]
    .venv/bin/python scripts/ingest_manifest.py --retag-only COLLECTION MANIFEST.json [...]
    .venv/bin/python scripts/ingest_manifest.py --refresh=written|manual-captions|ID,ID COLLECTION MANIFEST.json [...]

--refresh clears the chosen sources' chunks first, so they are rebuilt with the
current parsers (e.g. after a chunking or caption fix).

A manifest is a list of entries:
    {"url", "kind": "video"|"page", "type": "tutorial"|"doc"|"article" (pages),
     "title", "author", "published", "text_file" (optional: saved Markdown for a
     page fetch_page cannot read), "tags": {topics, scenarios, level, impact, tools}}

Several manifests may list the same source: entries are merged by video id /
URL, unioning topics, scenarios and tools and keeping the highest impact.
Videos go one by one (each with its own tags). Pages are fetched per site in
one batch so same-site boilerplate is recognised; pages with a text_file go in
through ingest_text. Finally every source is (re)tagged via set_doc_tags, so
tags are right even for chunks an earlier run already stored.
Idempotent: re-running skips chunks already stored, and sources that finished
completely (recorded in <first manifest dir>/ingest_state_<collection>.json)
are skipped without being downloaded again. A daily embedding quota stops the
run early; run it again after the reset.
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from yt_rag import RAGStore
from yt_rag.ingest import _extract_video_id, ingest_videos
from yt_rag.taxonomy import IMPACTS, validate_tags
from yt_rag.text_ingest import ingest_pages, ingest_text, page_doc_id


def doc_id(e: dict) -> str:
    return _extract_video_id(e["url"]) if e["kind"] == "video" else page_doc_id(e["url"])


def source_key(e: dict) -> str:
    """Same source despite http/https, www. or a trailing slash."""
    if e["kind"] == "video":
        return _extract_video_id(e["url"])
    u = re.sub(r"^https?://(www\.)?", "", e["url"].strip()).rstrip("/")
    return u.split("#", 1)[0]


def merge(entries: list[dict]) -> list[dict]:
    """One entry per source; tags unioned, highest impact kept. `aliases` keeps
    every URL the source was listed under (its chunks are stored under one)."""
    out: dict[str, dict] = {}
    for e in entries:
        e["tags"] = validate_tags(e.get("tags"))  # fail fast, before any network traffic
        key = source_key(e)
        if key not in out:
            out[key] = e
            e["aliases"] = [e["url"]]
            continue
        if e["url"] not in out[key]["aliases"]:
            out[key]["aliases"].append(e["url"])
        a, b = out[key]["tags"], e["tags"]
        for f in ("topics", "scenarios", "tools"):
            if f in a or f in b:
                a[f] = sorted(set(a.get(f, [])) | set(b.get(f, [])))
        if b.get("impact") and (not a.get("impact") or
                                IMPACTS.index(b["impact"]) < IMPACTS.index(a["impact"])):
            a["impact"] = b["impact"]
        a.setdefault("level", b.get("level"))
        if e.get("text_file") and not out[key].get("text_file"):
            out[key]["text_file"] = e["text_file"]
    return list(out.values())


def retag(collection: str, entries: list[dict]) -> int:
    """Set each source's merged tags on all its chunks; returns sources found."""
    rag = RAGStore(namespace=collection)
    found = 0
    for e in entries:
        n = 0
        for url in e.get("aliases") or [e["url"]]:
            n += rag.client.rpc("set_doc_tags", {
                "p_namespace": collection, "p_doc_id": doc_id({**e, "url": url}),
                "p_tags": e["tags"]}).execute().data or 0
        found += 1 if n else 0
    return found


def refresh_targets(collection: str, entries: list[dict], which: str) -> list[dict]:
    if which == "written":
        return [e for e in entries if e["kind"] == "page"]
    if which == "manual-captions":
        rag = RAGStore(namespace=collection)
        manual, off = set(), 0
        while True:
            rows = rag.client.table("documents").select("metadata->>doc_id").eq(
                "namespace", collection).eq("metadata->>captions", "manual").range(
                off, off + 999).execute().data
            manual |= {r["doc_id"] for r in rows}
            off += 1000
            if len(rows) < 1000:
                break
        return [e for e in entries if e["kind"] == "video" and doc_id(e) in manual]
    ids = set(which.split(","))
    return [e for e in entries if doc_id(e) in ids]


def clear(collection: str, targets: list[dict]) -> set[str]:
    """Delete every chunk of the target sources (under any alias URL)."""
    rag = RAGStore(namespace=collection)
    cleared = set()
    for e in targets:
        for url in e.get("aliases") or [e["url"]]:
            d = doc_id({**e, "url": url})
            rag.client.table("documents").delete().eq("namespace", collection).eq(
                "metadata->>doc_id", d).execute()
            cleared.add(d)
    return cleared


def main() -> None:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    retag_only = "--retag-only" in flags
    refresh = next((f.split("=", 1)[1] for f in flags if f.startswith("--refresh=")), None)
    collection, paths = args[0], args[1:]
    raw = [e for p in paths for e in json.loads(Path(p).read_text())]
    entries = merge(raw)
    print(f"{len(raw)} entries → {len(entries)} unique sources", file=sys.stderr)
    if retag_only:
        print(json.dumps({"sources": len(entries), "retagged_sources": retag(collection, entries)}))
        return

    state_path = Path(paths[0]).parent / f"ingest_state_{collection}.json"
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    if refresh:
        targets = refresh_targets(collection, entries, refresh)
        done -= clear(collection, targets)
        state_path.write_text(json.dumps(sorted(done)))
        print(f"refresh={refresh}: cleared {len(targets)} sources", file=sys.stderr)

    def mark(e):
        done.add(doc_id(e))
        state_path.write_text(json.dumps(sorted(done)))

    def quota_hit(ex) -> bool:
        return "daily embedding quota" in str(ex)

    todo = [e for e in entries
            if not any(doc_id({**e, "url": u}) in done for u in e["aliases"])]
    print(f"{len(entries) - len(todo)} already complete, {len(todo)} to go", file=sys.stderr)
    report = {"videos": 0, "video_chunks": 0, "no_subs": 0, "pages": 0, "page_chunks": 0,
              "texts": 0, "text_chunks": 0, "errors": [], "stopped_on_daily_quota": False}
    entries_all, entries = entries, todo
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
            if not r["no_subs"]:
                mark(e)
        except Exception as ex:
            if quota_hit(ex):
                report["stopped_on_daily_quota"] = True
                break
            report["errors"].append({"url": e["url"], "error": str(ex)})

    by_site = defaultdict(list)
    for e in pages:
        by_site[urlparse(e["url"]).netloc].append(e)
    for site, group in by_site.items():
        if report["stopped_on_daily_quota"]:
            break
        print(f"\n=== {len(group)} page(s) from {site}", file=sys.stderr)
        try:
            r = ingest_pages([e["url"] for e in group], collection,
                             type={e["url"]: e.get("type", "article") for e in group},
                             tags_by_url={e["url"]: e["tags"] for e in group})
            report["pages"] += r["pages"] - r["failed"]
            report["page_chunks"] += r["created"]
            report["errors"] += r["errors"]
            failed = {x["url"] for x in r["errors"]}
            for e in group:
                if e["url"] not in failed:
                    mark(e)
        except Exception as ex:
            if quota_hit(ex):
                report["stopped_on_daily_quota"] = True
                break
            report["errors"].append({"url": site, "error": str(ex)})

    for e in texts:
        if report["stopped_on_daily_quota"]:
            break
        print(f"\n=== text {e['text_file']}", file=sys.stderr)
        try:
            r = ingest_text(collection, Path(e["text_file"]).read_text(), title=e.get("title", ""),
                            source_url=e["url"], author=e.get("author", ""),
                            published_at=e.get("published", ""), type=e.get("type", "doc"),
                            tags=e["tags"])
            report["texts"] += 1
            report["text_chunks"] += r["created"]
            mark(e)
        except Exception as ex:
            if quota_hit(ex):
                report["stopped_on_daily_quota"] = True
                break
            report["errors"].append({"url": e["url"], "error": str(ex)})

    # (Re)tag every source, including chunks a previous run already stored.
    report["retagged_sources"] = retag(collection, entries_all)
    report["sources"] = len(entries_all)
    report["complete"] = len(done & {doc_id(e) for e in entries_all})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
