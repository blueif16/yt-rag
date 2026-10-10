"""
Channel ingest: yt-dlp enumerate + VTT chunker → RAGStore.

Top-level entry point:
    ingest_channel(channel, n=30, namespace=None) -> dict

All progress logging goes to stderr (so this is safe to call from an MCP
server, which uses stdout for protocol traffic).

Each chunk's metadata captures everything list_repository needs to
describe the corpus without re-parsing the content prefix.
"""
from __future__ import annotations
import re
import sys
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.networking.impersonate import ImpersonateTarget


def _log(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


# ---------- yt-dlp ----------

def _yt_opts_base() -> dict:
    return {"quiet": True, "no_warnings": True, "noprogress": True}


def list_channel_videos(channel_url: str, n: int) -> list[dict]:
    opts = {**_yt_opts_base(), "extract_flat": True, "playlistend": n}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"{channel_url.rstrip('/')}/videos", download=False)
    return [{"id": e["id"], "title": e.get("title", "")} for e in info.get("entries", [])]


def download_video_subs(video_id: str, out_dir: Path, sub_langs: list[str] | None = None, cookiefile: str | None = None) -> dict | None:
    langs = sub_langs or ["en.*"]
    opts = {
        **_yt_opts_base(),
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        # Politeness/robustness: YouTube 429-rate-limits rapid subtitle pulls.
        "sleep_interval_requests": 1.5,
        "retries": 5,
        "extractor_retries": 3,
        # Avoid the 429-prone on-demand translation endpoint (the tlang= path):
        # skip translated/converted tracks so only static source/manual subs are
        # pulled. Impersonate a real browser (needs curl_cffi) to dodge bot checks.
        "extractor_args": {"youtube": {"skip": ["translated_subs"], "player_client": ["default", "-web"]}},
        "impersonate": ImpersonateTarget.from_str("chrome"),
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except Exception as e:
        _log(f"  ! yt-dlp failed: {e}")
        return None
    for path in sorted(out_dir.glob(f"{video_id}.*.vtt")):
        return {"info": info, "vtt_path": path}
    return None


# ---------- VTT parsing ----------

TIMESTAMP_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
INLINE_TS_RE = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
TAG_RE = re.compile(r"</?c[^>]*>")


def _ts_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _clean(line: str) -> str:
    return TAG_RE.sub("", INLINE_TS_RE.sub("", line)).strip()


def parse_vtt(text: str) -> list[tuple[float, float, str]]:
    """Returns deduped (start_sec, end_sec, caption_text)."""
    lines = text.splitlines()
    cues = []
    i = 0
    while i < len(lines):
        m = TIMESTAMP_RE.match(lines[i])
        if not m:
            i += 1
            continue
        g = m.groups()
        start, end = _ts_to_sec(*g[:4]), _ts_to_sec(*g[4:])
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            t = _clean(lines[i])
            if t:
                text_lines.append(t)
            i += 1
        if text_lines:
            cues.append((start, end, text_lines[-1]))
    out = []
    for c in cues:
        if not out or out[-1][2] != c[2]:
            out.append(c)
    return out


def chunk_cues(cues, window_sec: int = 90, max_chars: int = 1800) -> list[dict]:
    chunks, cur_text = [], []
    cur_start = cur_end = None
    for start, end, text in cues:
        if cur_start is None:
            cur_start = start
        cur_end = end
        cur_text.append(text)
        if (cur_end - cur_start) >= window_sec or sum(len(t) + 1 for t in cur_text) >= max_chars:
            chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text)})
            cur_text, cur_start, cur_end = [], None, None
    if cur_text:
        chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text)})
    return chunks


# ---------- prefix + build ----------

def _fmt_time(sec: float) -> str:
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"


def _fmt_date(yyyymmdd: str | None) -> str:
    if yyyymmdd and len(yyyymmdd) == 8:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or ""


def build_items_for_video(info: dict, vtt_text: str, channel_handle: str | None) -> list[dict]:
    cues = parse_vtt(vtt_text)
    raw_chunks = chunk_cues(cues)
    channel = info.get("channel") or info.get("uploader") or ""
    title = info.get("title") or ""
    date = _fmt_date(info.get("upload_date"))
    vid = info.get("id") or ""
    items = []
    for c in raw_chunks:
        prefix = f'[{channel} — "{title}" — {date} — {_fmt_time(c["start"])}]\n'
        items.append({
            "content": prefix + c["text"],
            "source": f"https://youtu.be/{vid}?t={int(c['start'])}",
            "type": "yt_transcript",
            # rich metadata enables list_repository without prefix-parsing
            "extra_meta": {
                "video_id": vid,
                "channel": channel,
                "channel_handle": channel_handle or "",
                "title": title,
                "published_at": date,
                "chunk_start_sec": int(c["start"]),
                "chunk_end_sec": int(c["end"]),
            },
        })
    return items


# ---------- channel/url helpers ----------

def _resolve_target(target: str) -> tuple[str, str, str]:
    """Returns (channel_url, channel_handle, default_namespace)."""
    t = target.strip()
    if t.startswith("@"):
        return f"https://www.youtube.com/{t}", t, f"yt_{t.lstrip('@').lower()}"
    if t.startswith("http"):
        tail = t.rstrip("/").split("/")[-1]
        handle = tail if tail.startswith("@") else ""
        ns = f"yt_{tail.lstrip('@').lower()}" if tail else "yt_channel"
        return t, handle, ns
    return f"https://www.youtube.com/@{t}", f"@{t}", f"yt_{t.lower()}"


# ---------- main entry point ----------

def ingest_channel(
    channel: str,
    n: int = 30,
    namespace: str | None = None,
    sub_langs: list[str] | None = None,
    cookiefile: str | None = None,
) -> dict:
    """
    Enumerate the latest n uploads from a channel, download English VTT subs,
    chunk into ~90s windows with self-describing prefixes, and ingest into
    RAGStore under `namespace` (default: `yt_<handle>`).

    Idempotent — RAGStore dedupes by content_hash.
    """
    from yt_rag import RAGStore  # lazy: only after yt-dlp validates

    url, handle, default_ns = _resolve_target(channel)
    ns = namespace or default_ns

    _log(f"channel  : {url}")
    _log(f"namespace: {ns}")
    _log(f"n_videos : {n}")

    _log("\nenumerating...")
    videos = list_channel_videos(url, n)
    _log(f"found {len(videos)} videos")

    rag = RAGStore(namespace=ns)
    tmp = Path(tempfile.mkdtemp(prefix="yt_rag_"))
    totals = {"videos": len(videos), "chunks": 0, "created": 0, "skipped": 0, "no_subs": 0}

    for i, v in enumerate(videos, 1):
        _log(f"\n[{i}/{len(videos)}] {v['id']}  {v['title'][:65]}")
        res = download_video_subs(v["id"], tmp, sub_langs, cookiefile)
        if not res:
            _log("  - no English subs, skipping")
            totals["no_subs"] += 1
            continue
        with open(res["vtt_path"]) as f:
            vtt_text = f.read()
        items = build_items_for_video(res["info"], vtt_text, handle)
        if not items:
            _log("  - 0 chunks after parsing, skipping")
            continue

        # Inline batch insert with rich metadata (RAGStore.ingest_batch only
        # accepts source/type — we need extra fields, so we go one level down).
        rag_items = _batch_ingest_with_meta(rag, items)
        totals["chunks"] += len(items)
        totals["created"] += rag_items["created"]
        totals["skipped"] += rag_items["skipped"]
        _log(f"  - chunks={len(items)} created={rag_items['created']} skipped={rag_items['skipped']}")

    _log(f"\nDONE  videos={len(videos)} no_subs={totals['no_subs']} "
         f"chunks={totals['chunks']} created={totals['created']} skipped={totals['skipped']}")
    totals["namespace"] = ns
    return totals


VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def _extract_video_id(url_or_id: str) -> str:
    """Pull the 11-char YouTube video ID from a URL or accept it raw."""
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = VIDEO_ID_RE.search(s)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract video_id from: {url_or_id}")


def ingest_videos(urls: list[str], namespace: str) -> dict:
    """
    Cherry-pick specific YouTube videos by URL/ID into the given namespace.

    Use when the videos you want are scattered across a creator's archive
    or buried beyond the latest-N window. Reuses the same chunking and
    metadata as ingest_channel.

    Idempotent — RAGStore dedupes by content_hash.
    """
    from yt_rag import RAGStore  # lazy

    video_ids: list[str] = []
    for u in urls:
        try:
            video_ids.append(_extract_video_id(u))
        except ValueError as e:
            _log(f"  ! {e}")

    _log(f"namespace: {namespace}")
    _log(f"n_videos : {len(video_ids)}")

    if not video_ids:
        return {"videos": 0, "chunks": 0, "created": 0, "skipped": 0,
                "no_subs": 0, "namespace": namespace}

    rag = RAGStore(namespace=namespace)
    tmp = Path(tempfile.mkdtemp(prefix="yt_rag_"))
    totals = {"videos": len(video_ids), "chunks": 0, "created": 0,
              "skipped": 0, "no_subs": 0}

    for i, vid in enumerate(video_ids, 1):
        _log(f"\n[{i}/{len(video_ids)}] {vid}")
        res = download_video_subs(vid, tmp)
        if not res:
            _log("  - no English subs, skipping")
            totals["no_subs"] += 1
            continue
        with open(res["vtt_path"]) as f:
            vtt_text = f.read()
        info = res["info"]
        handle = info.get("uploader_id") or ""
        if handle and not handle.startswith("@"):
            handle = f"@{handle}"
        items = build_items_for_video(info, vtt_text, handle)
        if not items:
            _log("  - 0 chunks after parsing, skipping")
            continue
        rag_items = _batch_ingest_with_meta(rag, items)
        totals["chunks"] += len(items)
        totals["created"] += rag_items["created"]
        totals["skipped"] += rag_items["skipped"]
        _log(f"  - chunks={len(items)} created={rag_items['created']} "
             f"skipped={rag_items['skipped']}")

    _log(f"\nDONE  videos={len(video_ids)} no_subs={totals['no_subs']} "
         f"chunks={totals['chunks']} created={totals['created']} "
         f"skipped={totals['skipped']}")
    totals["namespace"] = namespace
    return totals


def _batch_ingest_with_meta(rag, items: list[dict], batch_size: int = 50) -> dict:
    """
    Batch ingest preserving the rich extra_meta fields. Mirrors RAGStore.ingest_batch
    but merges extra_meta into the metadata JSONB column.
    """
    created, skipped = 0, 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        hashes = [rag._hash(it["content"]) for it in batch]
        existing = rag.client.table("documents").select("content_hash").in_(
            "content_hash", hashes
        ).eq("namespace", rag.namespace).execute()
        existing_hashes = {r["content_hash"] for r in existing.data}

        new, new_hashes = [], []
        for it, h in zip(batch, hashes):
            if h in existing_hashes:
                skipped += 1
            elif it["content"]:
                new.append(it)
                new_hashes.append(h)
        if not new:
            continue

        embeddings = rag._embeddings.embed_documents([it["content"] for it in new])
        records = []
        for it, emb, h in zip(new, embeddings, new_hashes):
            meta = {"source": it["source"], "type": it["type"], **it["extra_meta"]}
            records.append({
                "content": it["content"],
                "content_hash": h,
                "embedding": emb,
                "metadata": meta,
                "namespace": rag.namespace,
            })
        result = rag.client.table("documents").insert(records).execute()
        created += len(result.data)
    return {"created": created, "skipped": skipped}
