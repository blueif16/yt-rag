"""
yt-rag MCP server.

Exposes three tools to any MCP-aware client (Claude Code, Claude Desktop, etc.):

  - search(query, namespace?, top_k=5)
      Hybrid RRF + graph search over ingested YouTube transcripts. Each hit
      includes the self-describing prefix and a ?t=<sec> deep-link.

  - list_repository()
      Inventory of namespaces: channels, video counts, chunk counts, date
      ranges, sample titles. Call this before search to know what's queryable.

  - ingest_channel(channel, n_videos=30, namespace?)
      Pull the latest n uploads from a YouTube channel, chunk transcripts,
      and store them. Idempotent — re-runs dedupe by content_hash.

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
load_dotenv(_PROJECT_ROOT / ".env")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from yt_rag import RAGStore  # noqa: E402
from yt_rag.ingest import ingest_channel as _ingest_channel  # noqa: E402


mcp = FastMCP("yt-rag")


@mcp.tool()
def search(query: str, namespace: str | None = None, top_k: int = 5) -> list[dict]:
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
        namespace: Scope to one creator (e.g., "yt_lewiswjackson"). Omit to
            search the entire corpus. Call list_repository() first if unsure
            what namespaces exist.
        top_k: Number of chunks to return (default 5, cap 25).

    Returns:
        List of {content, source_url, channel, title, published_at,
        chunk_start_sec, video_id, namespace}. `content` includes the prefix
        and the cleaned transcript; `source_url` is the deep-link.
    """
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
                         "SQL editor, or pass a specific `namespace` to scope the search."
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
            "namespace": namespace or _namespace_of(rag, r.get("id")),
        })
    return out


@mcp.tool()
def list_repository() -> dict:
    """
    Inventory of the ingested corpus across all namespaces.

    Call this before search() to discover what creators and topics are
    queryable, or to confirm an ingest landed. Aggregates by namespace
    and reports channels, video counts, chunk counts, date ranges, and
    sample titles.

    Returns:
        {
          total_chunks, total_videos, total_namespaces,
          namespaces: [
            {name, chunks, videos, channels, date_range: [earliest, latest],
             channel_handles, sample_titles}
          ]
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

    namespaces = []
    for ns in sorted(by_ns.keys()):
        b = by_ns[ns]
        namespaces.append({
            "name": ns,
            "chunks": b["chunks"],
            "videos": len(b["videos"]),
            "channels": sorted(b["channels"]),
            "channel_handles": sorted(b["channel_handles"]),
            "date_range": [b["earliest"], b["latest"]],
            "sample_titles": sorted(b["titles"])[:5],
        })

    return {
        "total_chunks": sum(b["chunks"] for b in by_ns.values()),
        "total_videos": sum(len(b["videos"]) for b in by_ns.values()),
        "total_namespaces": len(by_ns),
        "namespaces": namespaces,
    }


@mcp.tool()
def ingest_channel(channel: str, n_videos: int = 30, namespace: str | None = None) -> dict:
    """
    Ingest the latest N videos from a YouTube channel into the knowledge base.

    Downloads English transcripts via yt-dlp, parses YouTube's auto-subtitle
    format, chunks each video into ~90-second windows with a self-describing
    prefix, and stores in Supabase with vector + BM25 indexes. Idempotent.

    This is a synchronous, potentially slow operation: ~30s-2min for 30
    videos depending on transcript length. Run once per channel; subsequent
    runs will skip already-ingested chunks via content_hash dedupe.

    Args:
        channel: Channel handle ("@LewisWJackson"), URL
            ("https://www.youtube.com/@LewisWJackson"), or bare name
            ("LewisWJackson").
        n_videos: How many most-recent videos to pull (default 30, cap 100).
        namespace: Optional Supabase namespace; defaults to "yt_<handle>".
            Use a shared namespace across creators if you want a single
            cross-creator search bucket.

    Returns:
        {videos, no_subs, chunks, created, skipped, namespace}
    """
    n_videos = max(1, min(int(n_videos), 100))
    return _ingest_channel(channel, n=n_videos, namespace=namespace)


def _namespace_of(rag, doc_id) -> str:
    """Look up the namespace of a single document (only used when search() is
    called without a namespace filter, so we can attach it per-result)."""
    if doc_id is None:
        return ""
    r = rag.client.table("documents").select("namespace").eq("id", doc_id).execute()
    return r.data[0]["namespace"] if r.data else ""


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
