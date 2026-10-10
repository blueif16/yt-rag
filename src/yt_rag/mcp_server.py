"""
yt-rag MCP server.

The corpus is organized into topic COLLECTIONS — one per utility (what the
videos teach / what they're for), never one per channel. See collections.py
for the rules.

Tools:
  - list_repository()                 collections + purposes + stats (call first)
  - search(query, collection?, top_k) hybrid RRF + graph search, ?t=<sec> deep-links
  - create_collection(name, purpose)  register a new topic (only if none fits)
  - ingest_channels(collection, channels, n_videos?)  latest uploads → collection
  - ingest_videos(collection, urls)   cherry-picked videos → collection

Run via:
    python -m yt_rag.mcp_server
"""
from __future__ import annotations
from collections import defaultdict
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


mcp = FastMCP("yt-rag")


@mcp.tool()
def search(query: str, collection: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Search ingested YouTube transcripts via hybrid RRF (BM25 + vector) + graph traversal.

    Use this to find moments in videos that match a question, theme, or
    keyword. Each result is a ~90-second chunk with a self-describing
    prefix ([Channel — "Title" — date — MM:SS]) and a deep-link URL
    that jumps the viewer to the exact second.

    Args:
        query: What to search for. Semantic, keyword, or both — RRF handles
            mixed queries well. Be specific ("Bittensor TAO staking yield"
            beats "crypto").
        collection: Scope to one topic collection (e.g. "agentic-rag"). Pick it
            by matching your question to each collection's `purpose` from
            list_repository(). Omit to search the entire corpus.
        top_k: Number of chunks to return (default 5, cap 25).

    Returns:
        List of {content, source_url, channel, title, published_at,
        chunk_start_sec, video_id, collection}. `content` includes the prefix
        and the cleaned transcript; `source_url` is the deep-link.
    """
    namespace = collection
    top_k = max(1, min(int(top_k), 25))
    rag = RAGStore(namespace=namespace or "default")
    if namespace is None:
        # cross-namespace search via direct RPC (no namespace filter)
        embedding = rag._embeddings.embed_query(query)
        try:
            result = rag.client.rpc("search_context_mesh", {
                "query_text": query,
                "query_embedding": embedding,
                "match_count": top_k,
                "rrf_k": rag.config.rrf_k,
                "graph_depth": rag.config.graph_depth,
                "filter_namespace": None,
            }).execute()
            rows = result.data
        except Exception as e:
            return [{
                "error": f"Cross-namespace search failed: {e}. "
                         "Apply migrations/002_fix_search_namespace_null.sql in the Supabase "
                         "SQL editor, or pass a specific `collection` to scope the search."
            }]
    else:
        rows = rag.search(query, top_k=top_k)

    out = []
    for r in rows:
        meta = r.get("metadata") or {}
        out.append({
            "content": r.get("content", ""),
            "source_url": meta.get("source", ""),
            "channel": meta.get("channel", ""),
            "title": meta.get("title", ""),
            "published_at": meta.get("published_at", ""),
            "chunk_start_sec": meta.get("chunk_start_sec"),
            "video_id": meta.get("video_id", ""),
            "collection": namespace or _namespace_of(rag, r.get("id")),
        })
    return out


@mcp.tool()
def list_repository() -> dict:
    """
    Inventory of the corpus: every topic collection with its purpose and stats.

    ALWAYS call this before search() (to pick the collection whose purpose
    matches the question) and before any ingest (to reuse an existing
    collection instead of creating a near-duplicate).

    Returns:
        {
          total_chunks, total_videos, total_collections,
          collections: [
            {name, purpose, chunks, videos, channels, channel_handles,
             date_range: [earliest, latest], sample_titles}
          ],
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
            "videos": len(b["videos"]),
            "channels": sorted(b["channels"]),
            "channel_handles": sorted(b["channel_handles"]),
            "date_range": [b["earliest"], b["latest"]],
            "sample_titles": sorted(b["titles"])[:5],
        }
        (collections if ns in purposes else unregistered).append(entry)

    return {
        "total_chunks": sum(b["chunks"] for b in by_ns.values()),
        "total_videos": sum(len(b["videos"]) for b in by_ns.values()),
        "total_collections": len(collections),
        "collections": collections,
        "unregistered": unregistered,
    }


@mcp.tool()
def create_collection(name: str, purpose: str) -> dict:
    """
    Register a new topic collection. Do this ONLY after list_repository()
    shows no existing collection whose purpose already covers these videos.

    A collection = one utility: videos that teach the same thing / serve the
    same job, from any number of channels. NEVER one collection per channel.

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
                  cookiefile: str | None = None) -> dict:
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

    Returns:
        {videos, no_subs, chunks, created, skipped, collection}
        or {error} if the collection does not exist.
    """
    try:
        return _ingest_videos(urls, collection, sub_langs=sub_langs, cookiefile=cookiefile)
    except CollectionError as e:
        return {"error": str(e)}


def _registry(rag=None) -> CollectionRegistry:
    rag = rag or RAGStore(namespace="_meta")
    return CollectionRegistry(SupabaseBackend(rag.client))


def _namespace_of(rag, doc_id) -> str:
    """Look up the namespace of a single document (only used when search() is
    called without a namespace filter, so we can attach it per-result)."""
    if doc_id is None:
        return ""
    r = rag.client.table("documents").select("namespace").eq("id", doc_id).execute()
    return r.data[0]["namespace"] if r.data else ""


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
