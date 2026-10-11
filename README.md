# yt-rag

An MCP server that turns YouTube videos, web pages and notes into one
searchable knowledge base. Everything becomes text chunks — ~90-second stretches
of a video's captions, ~1500-char sections of a page — embedded with Gemini and
stored in Supabase. Search ranks by cosine similarity plus a small boost for
chunks holding the query's rare words (effect, plugin and menu names), and
returns passages: neighbouring chunks of one source that both match are joined.
Every hit links back to the exact second of the video or the section of the page.

## Collections: group by utility, not by channel

The corpus is organized into **collections**. A collection is one topic or job
the material serves — e.g. `agentic-rag`, `after-effects` — and mixes videos,
pages and notes from any number of channels and sites. Channels and sites are
only *sources*.

Rules (enforced in `src/yt_rag/collections.py`):

1. Names are lowercase kebab-case topic slugs, 3–40 chars.
2. No channel-derived names (`yt_<handle>`, `@handle`) and no catch-alls
   (`default`, `misc`, `general`, …).
3. Every collection has a one-sentence **purpose** saying what its material is for.
4. You can only ingest into a collection that already exists. Creating one is a
   separate step (`create_collection`), so you see existing collections first
   and reuse one when it fits.

## One text layer

Every chunk carries the same metadata (`src/yt_rag/chunks.py`): `type`
(transcript / article / tutorial / doc / note), `source` (deep link),
`doc_id`, `title`, `author`, `published_at`, `locator` (timestamp or section)
and `chunk_index`. Search results read the same whether they came from a video
or a page.

**Tags** (`src/yt_rag/taxonomy.py`) say what a source is good *for*, set once
per source at ingest: `topics`, `scenarios`, `level`, `impact` (how big a
visible jump it gives) and `tools` (plugins it requires; `[]` = stock). Search
can filter on them; `list_repository` shows how many sources carry each tag.

Ingest details:
- **Captions:** one track per video (the uploader's, else YouTube's original
  speech recognition). Sponsor reads are cut using SponsorBlock.
- **Pages:** fetched as main-text Markdown and split on headings. Site menus,
  tables of contents, link lists and account/cookie/copyright lines are dropped,
  as are footers repeated across pages of the same site. Tables too big for one
  chunk split between rows, each piece starting with the header row.
- **PDFs:** converted to Markdown with headings (pymupdf4llm), page-tracked.
- **Errors:** Gemini and YouTube rate limits are retried.

## Tools

| Tool | What it does |
|---|---|
| `list_repository()` | Every collection: purpose, counts by type, tag coverage, authors; plus the tag vocabulary. Call first. |
| `search(query, collection?, top_k=10, scenario?, topic?, type?, tools_available?, per_source=3, context=0)` | Similarity search with a rare-word boost, an absolute `relevance` score, tag filters, at most `per_source` passages per video/page (adjacent matching chunks join into one passage), and optional neighbouring chunks. Agents should search twice: the question, then the answer's exact terms. |
| `create_collection(name, purpose)` | Register a new topic (only when none fits). |
| `ingest_urls(collection, urls, type?, tags?)` | Any links: YouTube → captions; web pages, PDFs (manuals, papers; chunks link to `#page=N`) and local files → text. |
| `ingest_text(collection, text, title, source_url, ..., tags?)` | Text you already have (a JS-only page, a transcript from elsewhere, a note). |
| `ingest_videos(collection, urls, tags?)` | Cherry-pick videos into a collection. |
| `ingest_channels(collection, channels, n_videos=30)` | Pull the latest uploads of one or more channels into a collection. |

Ingest is idempotent (content-hash dedupe). Videos without captions are skipped;
there is no audio transcription.

For a curated batch with per-source tags:
`.venv/bin/python scripts/ingest_manifest.py <collection> manifest.json`.

**Evaluating a collection:** `eval/run_queries.py` runs a question set and
records hits and timings; `eval/after-effects/` holds the pilot's question sets
and judgement.

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
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | no | defaults `gemini-embedding-2`, `768`. Models are not interchangeable: switching means `scripts/reembed.py` |

**Database:** run the files in `migrations/` in order (`001` … `010`) in the
Supabase SQL editor.

**Check it works:**

```bash
.venv/bin/python scripts/smoke_test.py   # Gemini + Supabase round-trip
.venv/bin/python -m pytest -q            # chunking, tags, collection rules (offline)
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
.venv/bin/python scripts/ingest_manifest.py <collection> manifest.json [...]
```

## Rate limits

YouTube rate-limits subtitle pulls (HTTP 429). The ingester paces requests,
downloads one caption track per video (never the 429-prone translated ones),
backs off on 429, and impersonates a browser. Gemini's free tier allows 100
embeddings a minute; the embedder waits out its 429s. For heavy use you can pass
`cookiefile` (a Netscape `cookies.txt`) — use a throwaway Google account, never
your main one.

## License

MIT
