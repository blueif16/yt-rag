# yt-rag

An MCP server that turns YouTube videos into a searchable knowledge base.
It pulls each video's captions, splits them into ~90-second chunks, embeds them
with Gemini, and stores them in Supabase for hybrid search (BM25 + vector,
fused with RRF, plus graph traversal). Every hit links to the exact second
in the video.

## Collections: group by utility, not by channel

The corpus is organized into **collections**. A collection is one topic or job
the videos serve — e.g. `agentic-rag`, `crypto-staking` — and can draw from any
number of channels. Channels are only a *source* of videos.

Rules (enforced in `src/yt_rag/collections.py`):

1. Names are lowercase kebab-case topic slugs, 3–40 chars.
2. No channel-derived names (`yt_<handle>`, `@handle`) and no catch-alls
   (`default`, `misc`, `general`, …).
3. Every collection has a one-sentence **purpose** saying what its videos are for.
4. You can only ingest into a collection that already exists. Creating one is a
   separate step (`create_collection`), so you see existing collections first
   and reuse one when it fits.

## Tools

| Tool | What it does |
|---|---|
| `list_repository()` | Every collection with its purpose, video/chunk counts, channels, date range. Call first. |
| `search(query, collection?, top_k=5)` | Hybrid search; omit `collection` to search everything. |
| `create_collection(name, purpose)` | Register a new topic (only when none fits). |
| `ingest_videos(collection, urls)` | Cherry-pick videos into a collection (preferred). |
| `ingest_channels(collection, channels, n_videos=30)` | Pull the latest uploads of one or more channels into a collection. |

Ingest is idempotent (content-hash dedupe). Videos without captions are skipped —
there is no audio transcription; the server uses YouTube's existing manual or
auto-generated caption track via `yt-dlp`.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/blueif16/yt-rag && cd yt-rag
uv sync
cp .env.example .env   # fill in the keys below
```

**Keys** (see `.env.example`):

| Variable | Required | Where |
|---|---|---|
| `SUPABASE_URL` | yes | Supabase → Project Settings → API |
| `SUPABASE_SECRET_KEY` | yes | same page (secret / service-role key) |
| `GEMINI_API_KEY` | yes | https://aistudio.google.com/apikey |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | no | defaults `gemini-embedding-001`, `768` |

**Database:** run the files in `migrations/` in order (`001`, `002`, `003`) in the
Supabase SQL editor.

**Check it works:**

```bash
.venv/bin/python scripts/smoke_test.py   # Gemini + Supabase round-trip
.venv/bin/python -m pytest -q            # collection rules (offline)
```

## Register with Claude Code

```bash
claude mcp add yt-rag -s user -- /ABSOLUTE/PATH/yt-rag/.venv/bin/python -m yt_rag.mcp_server
```

The server loads `.env` from its own directory, so it works regardless of the
client's working directory.

## CLI scripts

```bash
.venv/bin/python scripts/ingest_videos.py  <collection> <url> [url ...]
.venv/bin/python scripts/ingest_channel.py <collection> @handle [n=30]
```

## Rate limits

YouTube rate-limits subtitle pulls (HTTP 429). The ingester paces requests,
skips translated tracks, and impersonates a browser. For heavy use you can pass
`cookiefile` (a Netscape `cookies.txt`) — use a throwaway Google account, never
your main one.

## License

MIT
