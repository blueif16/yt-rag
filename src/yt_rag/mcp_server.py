"""
yt-rag MCP server.

The corpus is organized into topic COLLECTIONS — one per utility (what the
material teaches / what it's for), never one per channel or site. A collection
mixes video captions, web pages and pasted text: everything is a text chunk
with the same metadata envelope (see chunks.py). See collections.py for the rules.

Tools:
  - list_repository()                 collections + purposes + stats (call first)
  - search(query, collection?, top_k) similarity + rare-word boost, passages, ?t=<sec> deep-links
  - create_collection(name, purpose)  register a new topic (only if none fits)
  - ingest_channels(collection, channels, n_videos?)  latest uploads → collection
  - ingest_videos(collection, urls)   cherry-picked videos → collection
  - ingest_urls(collection, urls, type?)  any links: YouTube → captions, else page text
  - ingest_text(collection, text, title, source_url, ...)  text you already have

Run via:
    python -m yt_rag.mcp_server
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from the project root, regardless of cwd (this matters because the
# MCP client spawns this process with an arbitrary working directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from yt_rag import RAGStore  # noqa: E402
from yt_rag.collections import CollectionError, CollectionRegistry, SupabaseBackend  # noqa: E402
from yt_rag.ingest import ingest_channel as _ingest_channel  # noqa: E402
from yt_rag.ingest import ingest_videos as _ingest_videos  # noqa: E402
from yt_rag.text_ingest import ingest_text as _ingest_text  # noqa: E402
from yt_rag.text_ingest import ingest_urls as _ingest_urls  # noqa: E402
from yt_rag.taxonomy import IMPACTS, LEVELS, SCENARIOS, TOPICS, TagError  # noqa: E402


mcp = FastMCP("yt-rag")


@lru_cache(maxsize=32)
def _store(namespace: str) -> RAGStore:
    """One RAGStore (Supabase + Gemini clients) per collection, reused across
    calls — building the clients costs more than a search."""
    return RAGStore(namespace=namespace)


@mcp.tool()
def search(query: str, collection: str | None = None, top_k: int = 10,
           scenario: str | None = None, topic: str | None = None,
           type: str | None = None, tools_available: list[str] | None = None,
           per_source: int = 3, context: int = 0) -> list[dict]:
    """
    Search the corpus (video captions + web pages + notes): vector similarity
    plus a small boost for chunks that contain the query's rare words (effect,
    plugin and menu names).

    Each result is one passage: a ~90-second stretch of a video or a
    ~1500-char section of a page, or two such chunks back to back when both
    match (`chunks`: 2). It starts with a self-describing header
    ([author — "title" — date — locator]). `source_url` deep-links to the
    exact second (videos) or section (pages).

    `relevance` is the cosine similarity to your query (gemini-embedding-2,
    calibrated on the after-effects collection). Read it as absolute: best hit
    0.77+ = a real answer; 0.70-0.77 = often partial, read and cross-check;
    below ~0.70 = the library most likely has nothing on this — say so
    instead of forcing an answer.

    Search twice. First ask the question in plain words. Then search again
    with only the terms the answer itself would contain: effect, property,
    menu, expression and plugin names and the trade words for the technique,
    3-8 terms, no sentence ("Puppet Position Pin Advanced Starch overlap",
    "Roto Brush Refine Edge decontaminate edge colors"). Read both result lists:
    the question finds the explanations, the terms find the recipes that use
    other words for the same thing.

    Args:
        query: A full natural-language question ("why does my glow disappear
            on a white background"), or the exact terms of the answer
            ("CC Force Motion Blur shutter angle", "loopOut cycle").
        collection: Scope to one topic collection (e.g. "after-effects"). Pick it
            by matching your question to each collection's `purpose` from
            list_repository(). Omit to search the entire corpus.
        top_k: Number of passages to return (default 10, cap 25). The best one
            often ranks 6-10, so read all of them before concluding.
        scenario: Only sources tagged for this kind of shot/video, e.g.
            "ui-promo", "logo-ident", "hero-fx" (vocabulary: list_repository).
        topic: Only sources tagged with this technique, e.g. "easing",
            "light-glow", "expressions".
        type: Only this kind of text: transcript | article | tutorial | doc | note.
            "doc" for exact settings/code, "transcript" for technique and feel.
        tools_available: Plugins you can use, e.g. ["deep-glow-2", "rsmb",
            "trapcode-particular"]; sources that need anything else are left
            out. [] = stock After Effects only.
        per_source: At most this many passages from one video/page (default 3),
            so one long tutorial or manual cannot fill every slot. 0 = no cap.
        context: 1 = also return the chunk before and after each passage
            (context_before / context_after), for steps or code that run
            across a boundary. Default 0.

    Returns:
        List of {content, relevance, source_url, type, title, author,
        published_at, locator, chunks, collection, tags} (+ video_id,
        chunk_start_sec, views for videos; + context_before, context_after when
        context=1). `tags` = {topics, scenarios, level, impact, tools} when the
        source is tagged.
    """
    top_k = max(1, min(int(top_k), 25))
    try:
        rag, passages = search_passages(query, collection, top_k, scenario=scenario, topic=topic,
                                        type=type, tools_available=tools_available,
                                        per_source=per_source)
    except Exception as e:
        return [{"error": f"search failed: {e}"}]
    out = [_passage_hit(p, collection) for p in passages]
    if context:
        _attach_neighbours(rag, passages, out)
    return out


def search_passages(query: str, collection: str | None = None, top_k: int = 10, *,
                    scenario: str | None = None, topic: str | None = None,
                    type: str | None = None, tools_available: list[str] | None = None,
                    per_source: int = 3, keyword_boost: float | None = None):
    """The rows behind search(): (store, passages), each passage a list of
    chunk rows (with ids) in reading order. Used by the tool and by the eval."""
    filter_meta: dict = {}
    if scenario:
        filter_meta["scenarios"] = [scenario]
    if topic:
        filter_meta["topics"] = [topic]
    if type:
        filter_meta["type"] = type
    rag = _store(collection or "*")
    # A big source (a manual, the expression reference) can fill a small pool on
    # its own; capping that pool then left agents with 3-4 hits. Draw a wide
    # pool so the cap still leaves top_k (the vector half is an exact scan, so
    # this costs a few ms).
    pool = min(100, max(top_k * 10, 30)) if per_source else top_k * 3
    rows = rag.search(query, top_k=pool, filter_meta=filter_meta or None,
                      tools_available=tools_available, keyword_boost=keyword_boost)
    return rag, group_passages(rows, per_source, top_k)


TAG_FIELDS = ("topics", "scenarios", "level", "impact", "tools")
MAX_PASSAGE_CHUNKS = 2


def _doc_key(r: dict) -> str:
    meta = r.get("metadata") or {}
    return meta.get("doc_id") or meta.get("video_id") or meta.get("source") or str(r.get("id"))


def cap_per_source(rows: list[dict], per_source: int) -> list[dict]:
    """Keep rows best-first, at most `per_source` per video/page (doc_id)."""
    if not per_source:
        return rows
    seen: dict[str, int] = {}
    kept = []
    for r in rows:
        doc = _doc_key(r)
        if seen.get(doc, 0) < per_source:
            seen[doc] = seen.get(doc, 0) + 1
            kept.append(r)
    return kept


def group_passages(rows: list[dict], per_source: int, top_k: int) -> list[list[dict]]:
    """Best-first rows -> at most top_k passages.

    A row next to a passage already chosen from the same video or page
    (chunk_index ± 1) joins that passage instead of taking a slot of its own,
    up to MAX_PASSAGE_CHUNKS chunks; the per-source cap counts passages. So a
    source whose matching chunks sit side by side reads as one passage, and
    its cap is left for its other sections (measured on the after-effects set:
    every chunk the plain cap kept is still there, plus the neighbours that
    carry the rest of a recipe). Once top_k passages exist, the scan stops at
    the first row that would need a slot of its own."""
    passages: list[list[dict]] = []
    seen: dict[str, int] = {}
    for r in rows:
        doc, idx = _doc_key(r), (r.get("metadata") or {}).get("chunk_index")
        if idx is not None:
            home = next((p for p in passages if _doc_key(p[0]) == doc
                         and len(p) < MAX_PASSAGE_CHUNKS
                         and any(abs(int(idx) - int(x["metadata"]["chunk_index"])) == 1 for x in p)),
                        None)
            if home is not None:
                home.append(r)
                home.sort(key=lambda x: int(x["metadata"]["chunk_index"]))
                continue
        if len(passages) >= top_k:
            break
        if per_source and seen.get(doc, 0) >= per_source:
            continue
        seen[doc] = seen.get(doc, 0) + 1
        passages.append([r])
    return passages


def _passage_hit(p: list[dict], collection: str | None) -> dict:
    first, last = p[0], p[-1]
    meta, last_meta = first.get("metadata") or {}, last.get("metadata") or {}
    content = first.get("content", "")
    for r in p[1:]:
        content += "\n\n" + _body(r.get("content", ""))
    sims = [r["similarity"] for r in p if r.get("similarity") is not None]
    locator = meta.get("locator", "")
    if len(p) > 1:
        if meta.get("chunk_start_sec") is not None and last_meta.get("locator"):
            locator = f"{locator.split('–')[0]}–{last_meta['locator'].split('–')[-1]}"
        elif last_meta.get("locator") and last_meta["locator"] != locator:
            locator = f"{locator} / {last_meta['locator']}"
    hit = {
        "content": content,
        "relevance": round(max(sims), 3) if sims else None,
        "source_url": meta.get("source", ""),
        "type": meta.get("type", ""),
        "title": meta.get("title", ""),
        "author": meta.get("author") or meta.get("channel", ""),
        "published_at": meta.get("published_at", ""),
        "locator": locator,
        "chunks": len(p),
        "collection": first.get("namespace") or collection or "",
    }
    tags = {k: meta[k] for k in TAG_FIELDS if k in meta}
    if tags:
        hit["tags"] = tags
    if meta.get("video_id"):
        hit["video_id"] = meta["video_id"]
        hit["chunk_start_sec"] = meta.get("chunk_start_sec")
        if meta.get("views") is not None:
            hit["views"] = meta["views"]
    return hit


def _body(content: str) -> str:
    """Chunk text without its [author — "title" — ...] header line."""
    return content.split("\n", 1)[1] if content.startswith("[") and "\n" in content else content


def _attach_neighbours(rag, passages: list[list[dict]], hits: list[dict]) -> None:
    """Add context_before / context_after: the chunk just before and just after
    each passage (same doc_id, first chunk_index - 1, last + 1), in one query."""
    keys, clauses = [], []
    for p, hit in zip(passages, hits):
        meta, last = p[0].get("metadata") or {}, p[-1].get("metadata") or {}
        doc, lo, hi = meta.get("doc_id"), meta.get("chunk_index"), last.get("chunk_index")
        if doc is None or lo is None or hi is None:
            keys.append(None)
            continue
        keys.append((hit["collection"], doc, int(lo), int(hi)))
        clauses.append(f'and(namespace.eq."{hit["collection"]}",metadata->>doc_id.eq."{doc}",'
                       f'metadata->>chunk_index.in.({int(lo) - 1},{int(hi) + 1}))')
    if not clauses:
        return
    found = rag.client.table("documents").select("namespace, content, metadata").or_(
        ",".join(clauses)).execute().data
    by_key = {(f["namespace"], f["metadata"].get("doc_id"), int(f["metadata"].get("chunk_index"))): f["content"]
              for f in found}
    for key, hit in zip(keys, hits):
        if key:
            ns, doc, lo, hi = key
            if (ns, doc, lo - 1) in by_key:
                hit["context_before"] = _body(by_key[(ns, doc, lo - 1)])
            if (ns, doc, hi + 1) in by_key:
                hit["context_after"] = _body(by_key[(ns, doc, hi + 1)])


@mcp.tool()
def list_repository() -> dict:
    """
    Inventory of the corpus: every topic collection with its purpose and stats
    (items = distinct videos/pages/notes, by_type = items per type).

    ALWAYS call this before search() (to pick the collection whose purpose
    matches the question) and before any ingest (to reuse an existing
    collection instead of creating a near-duplicate).

    Returns:
        {
          total_chunks, total_videos, total_collections,
          collections: [
            {name, purpose, chunks, items, by_type: {transcript: n, article: n, ...},
             tags: {topics: {easing: n, ...}, scenarios: {...}, impact: {...}, ...},
             videos, channels, channel_handles, authors,
             date_range: [earliest, latest], sample_titles}
          ],
          tag_vocabulary: {topics, scenarios, level, impact, tools}  # filter values for search()
          unregistered: [ same shape, purpose=null ]   # legacy data with no
              # registered collection (e.g. old per-channel "yt_<handle>")
        }
    """
    rag = RAGStore(namespace="_meta")  # only using the raw client
    # Pull all rows in pages (PostgREST default page is 1000)
    rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        page = rag.client.table("documents").select(
            "namespace, metadata"
        ).range(offset, offset + page_size - 1).execute()
        rows.extend(page.data)
        if len(page.data) < page_size:
            break
        offset += page_size

    by_ns: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "chunks": 0,
        "videos": set(),
        "channels": set(),
        "channel_handles": set(),
        "authors": set(),
        "items": set(),
        "by_type": defaultdict(set),
        "tags": defaultdict(lambda: defaultdict(set)),
        "titles": set(),
        "earliest": None,
        "latest": None,
    })

    for r in rows:
        ns = r.get("namespace") or "default"
        meta = r.get("metadata") or {}
        b = by_ns[ns]
        b["chunks"] += 1
        if meta.get("video_id"):
            b["videos"].add(meta["video_id"])
        if meta.get("channel"):
            b["channels"].add(meta["channel"])
        if meta.get("channel_handle"):
            b["channel_handles"].add(meta["channel_handle"])
        if meta.get("title"):
            b["titles"].add(meta["title"])
        doc = meta.get("doc_id") or meta.get("video_id") or meta.get("source")
        if doc:
            b["items"].add(doc)
            b["by_type"][meta.get("type") or "unknown"].add(doc)
        if meta.get("author") or meta.get("channel"):
            b["authors"].add(meta.get("author") or meta["channel"])
        if doc:
            for field in ("topics", "scenarios", "tools"):
                for v in meta.get(field) or []:
                    b["tags"][field][v].add(doc)
            for field in ("level", "impact"):
                if meta.get(field):
                    b["tags"][field][meta[field]].add(doc)
        d = meta.get("published_at")
        if d:
            if b["earliest"] is None or d < b["earliest"]:
                b["earliest"] = d
            if b["latest"] is None or d > b["latest"]:
                b["latest"] = d

    purposes = {c["name"]: c["purpose"] for c in _registry(rag).list()}
    collections, unregistered = [], []
    for ns in sorted(set(by_ns) | set(purposes)):
        b = by_ns[ns]
        entry = {
            "name": ns,
            "purpose": purposes.get(ns),
            "chunks": b["chunks"],
            "items": len(b["items"]),
            "by_type": {t: len(d) for t, d in sorted(b["by_type"].items())},
            # items per tag value — what the collection covers, and how well
            "tags": {f: dict(sorted(((v, len(d)) for v, d in vals.items()), key=lambda x: -x[1]))
                     for f, vals in b["tags"].items()},
            "videos": len(b["videos"]),
            "channels": sorted(b["channels"]),
            "channel_handles": sorted(b["channel_handles"]),
            "authors": sorted(b["authors"]),
            "date_range": [b["earliest"], b["latest"]],
            "sample_titles": sorted(b["titles"])[:5],
        }
        (collections if ns in purposes else unregistered).append(entry)

    return {
        "total_chunks": sum(b["chunks"] for b in by_ns.values()),
        "total_items": sum(len(b["items"]) for b in by_ns.values()),
        "total_videos": sum(len(b["videos"]) for b in by_ns.values()),
        "total_collections": len(collections),
        "collections": collections,
        "unregistered": unregistered,
        "tag_vocabulary": {"topics": TOPICS, "scenarios": SCENARIOS,
                           "level": list(LEVELS), "impact": list(IMPACTS),
                           "tools": "required plugins, kebab-case; [] = stock AE"},
    }


@mcp.tool()
def create_collection(name: str, purpose: str) -> dict:
    """
    Register a new topic collection. Do this ONLY after list_repository()
    shows no existing collection whose purpose already covers these videos.

    A collection = one utility: videos, pages and notes that teach the same
    thing / serve the same job, from any number of channels or sites. NEVER one
    collection per channel or site.

    Args:
        name: Topic slug, lowercase kebab-case, 3-40 chars (e.g. "agentic-rag",
            "crypto-staking"). Rejected: channel-derived names ("yt_<handle>",
            "@handle") and catch-alls ("default", "misc", "general", ...).
        purpose: One sentence (>= 20 chars) saying what the videos are FOR, so a
            later caller can tell whether a new video belongs here. Good:
            "Tutorials on building agentic RAG pipelines with LangGraph".
            Bad: "videos", "Lewis Jackson's channel".

    Returns:
        {name, purpose} on success, or {error} explaining which rule failed.
    """
    try:
        return _registry().create(name, purpose)
    except CollectionError as e:
        return {"error": str(e)}


@mcp.tool()
def ingest_channels(collection: str, channels: list[str], n_videos: int = 30,
                    sub_langs: list[str] | None = None,
                    cookiefile: str | None = None) -> dict:
    """
    Pull the latest N uploads from one or more channels into ONE existing topic
    collection. Channels are only a source of videos; the collection is the group.

    Prefer ingest_videos when only some of a channel's uploads fit the
    collection's purpose — whole-channel pulls mix in off-topic videos.

    Pulls the STATIC source/manual subtitle track via yt-dlp (translated tracks are
    skipped to dodge YouTube's 429-prone translation endpoint), chunks each video
    into ~90-second windows, and stores them with vector + BM25 indexes.
    Idempotent (content_hash dedupe). Slow: ~30s-2min per 30 videos.
    Continues past per-channel errors.

    Args:
        collection: Name of an existing collection (see list_repository; create
            one with create_collection only if none fits).
        channels: Channel handles ("@LewisWJackson"), URLs, or bare names.
        n_videos: Most-recent uploads per channel (default 30, cap 100).
        sub_langs: yt-dlp subtitle-language patterns. Default ["en.*"]. For a
            non-English channel pass its static source track, e.g.
            ["zh-Hant-orig","zh-Hans-orig","zh-orig","zh-TW","zh-Hant","zh-Hans","zh"].
        cookiefile: Optional Netscape cookies.txt. Use a THROWAWAY Google account
            (never your primary) for higher rate limits.

    Returns:
        {"collection", "totals": {videos, chunks, created, skipped, no_subs},
         "channels": [...]} or {error} if the collection does not exist.
    """
    import time
    n = max(1, min(int(n_videos), 100))
    results = []
    for i, ch in enumerate(channels):
        if i:
            time.sleep(4)  # pace between channels; YouTube rate-limits subtitle pulls
        rec = {"channel": ch}
        try:
            rec.update(_ingest_channel(ch, collection, n=n, sub_langs=sub_langs,
                                       cookiefile=cookiefile))
        except CollectionError as e:
            return {"error": str(e)}  # same for every channel — stop now
        except Exception as e:  # one bad channel must not abort the batch
            rec["error"] = str(e)
        results.append(rec)
    totals = {k: sum(r.get(k, 0) for r in results)
              for k in ("videos", "chunks", "created", "skipped", "no_subs")}
    return {"collection": collection, "totals": totals, "channels": results}


@mcp.tool()
def ingest_videos(collection: str, urls: list[str],
                  sub_langs: list[str] | None = None,
                  cookiefile: str | None = None, tags: dict | None = None) -> dict:
    """
    Cherry-pick specific YouTube videos into ONE existing topic collection.
    The preferred ingest: pick only videos that serve the collection's purpose,
    across any number of creators.

    Args:
        collection: Name of an existing collection (see list_repository; create
            one with create_collection only if none fits).
        urls: YouTube video URLs or 11-char IDs ("watch?v=", "youtu.be/",
            "/embed/", "/shorts/", or raw ID).
        sub_langs: Same as ingest_channels (default ["en.*"]).
        cookiefile: Same as ingest_channels.
        tags: What these videos are good for, applied to every chunk:
            {topics, scenarios, level, impact, tools} from list_repository's
            tag_vocabulary. Tag one source per call when tags differ.

    Returns:
        {videos, no_subs, chunks, created, skipped, collection}
        or {error} if the collection does not exist.
    """
    try:
        return _ingest_videos(urls, collection, sub_langs=sub_langs, cookiefile=cookiefile,
                              tags=tags)
    except (CollectionError, TagError) as e:
        return {"error": str(e)}


@mcp.tool()
def ingest_urls(collection: str, urls: list[str], type: str = "article",
                sub_langs: list[str] | None = None,
                cookiefile: str | None = None, tags: dict | None = None) -> dict:
    """
    Add any mix of links to ONE existing collection. YouTube links go through
    captions (same as ingest_videos); every other link is fetched, reduced to
    its main text, split on headings into ~1500-char chunks, and stored next to
    the captions with the same metadata (title, author, date, section, URL).
    Web pages, PDFs (manuals, papers — chunks link to #page=N) and local files
    (file:///abs/path or /abs/path: .pdf, .md, .txt, .html) all work.

    Args:
        collection: Name of an existing collection (see list_repository).
        urls: YouTube and/or web page URLs.
        type: What the web pages are: "tutorial" (step-by-step how-to),
            "doc" (reference/manual), or "article" (default; essays, explainers).
            Videos are always "transcript".
        sub_langs, cookiefile: Same as ingest_videos (YouTube links only).
        tags: Same as ingest_videos, applied to every link in this call.

    Returns:
        {collection, videos?: {...}, pages?: {pages, failed, chunks, created,
         skipped, errors}} or {error}. Pages that fail (HTTP error, paywall,
        JS-only) are listed in pages.errors — pass their text via ingest_text.
    """
    try:
        return _ingest_urls(urls, collection, type=type, sub_langs=sub_langs,
                            cookiefile=cookiefile, tags=tags)
    except (CollectionError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def ingest_text(collection: str, text: str, title: str, source_url: str,
                author: str = "", published_at: str = "", type: str = "note",
                tags: dict | None = None) -> dict:
    """
    Store text you already have in ONE existing collection: an extract you
    fetched yourself, a page ingest_urls could not read, a transcript from
    another platform, or your own research note. Markdown headings are used to
    split it; each chunk keeps the source.

    Args:
        collection: Name of an existing collection.
        text: The text (Markdown welcome). Keep it faithful to the source;
            say so in the title if it is your own summary.
        title: Title of the source.
        source_url: Where it came from (required) — a URL, or file:///path.
        author: Author, channel or site.
        published_at: YYYY-MM-DD if known.
        type: transcript | article | tutorial | doc | note (default).
        tags: Same as ingest_videos.

    Returns:
        {collection, chunks, created, skipped} or {error}.
    """
    try:
        return _ingest_text(collection, text, title, source_url, author=author,
                            published_at=published_at, type=type, tags=tags)
    except (CollectionError, ValueError) as e:
        return {"error": str(e)}


def _registry(rag=None) -> CollectionRegistry:
    rag = rag or RAGStore(namespace="_meta")
    return CollectionRegistry(SupabaseBackend(rag.client))


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
