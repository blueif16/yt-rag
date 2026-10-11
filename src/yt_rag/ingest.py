"""
Ingest: yt-dlp captions + VTT chunker → RAGStore, into a topic collection.

Top-level entry points (both require an existing collection — see collections.py):
    ingest_channel(channel, collection, n=30) -> dict
    ingest_videos(urls, collection) -> dict

All progress logging goes to stderr (so this is safe to call from an MCP
server, which uses stdout for protocol traffic).

Each chunk's metadata captures everything list_repository needs to
describe the corpus without re-parsing the content prefix.
"""
from __future__ import annotations
import re
import sys
import tempfile
import time
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.networking.impersonate import ImpersonateTarget

from yt_rag.chunks import chunk_meta, header
from yt_rag.taxonomy import validate_tags


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


def pick_caption_track(info: dict, patterns: list[str]) -> tuple[str, str] | None:
    """Choose ONE caption track: the uploader's (manual) track if one matches,
    else YouTube's original-language speech recognition (`<lang>-orig`), else
    any matching auto track. Returns (lang, "manual" | "auto") or None.

    The plain auto track (e.g. "en" next to "en-orig") is served through the
    translation endpoint (tlang=), which YouTube rate-limits hard — asking for
    it is what made whole videos fail with HTTP 429.
    """
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    rxs = [re.compile(f"(?:{p})$") for p in patterns]
    for rx in rxs:
        for lang in sorted(manual):
            if rx.match(lang):
                return lang, "manual"
    for rx in rxs:
        for lang in sorted(auto):
            if lang.endswith("-orig") and rx.match(lang[:-5]):
                return lang, "auto"
    for rx in rxs:
        for lang in sorted(auto):
            if rx.match(lang):
                return lang, "auto"
    return None


def download_video_subs(video_id: str, out_dir: Path, sub_langs: list[str] | None = None, cookiefile: str | None = None) -> dict | None:
    """Returns {"info", "vtt_path", "captions": "manual" | "auto"} or None."""
    patterns = sub_langs or ["en.*"]
    base = {
        **_yt_opts_base(),
        "skip_download": True,
        # Politeness/robustness: YouTube 429-rate-limits rapid subtitle pulls.
        "sleep_interval_requests": 1.5,
        "retries": 5,
        "extractor_retries": 3,
        # Skip translated/converted tracks (the 429-prone tlang= endpoint).
        # Impersonate a real browser (needs curl_cffi) to dodge bot checks.
        "extractor_args": {"youtube": {"skip": ["translated_subs"], "player_client": ["default", "-web"]}},
        "impersonate": ImpersonateTarget.from_str("chrome"),
    }
    if cookiefile:
        base["cookiefile"] = cookiefile
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(1, 4):
        try:
            with YoutubeDL(base) as ydl:
                info = ydl.extract_info(url, download=False)
            track = pick_caption_track(info, patterns)
            if not track:
                return None
            lang, kind = track
            opts = {**base, "writesubtitles": kind == "manual",
                    "writeautomaticsub": kind == "auto", "subtitleslangs": [lang],
                    "subtitlesformat": "vtt", "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
            with YoutubeDL(opts) as ydl:
                ydl.process_ie_result(info, download=True)
            break
        except Exception as e:
            # YouTube 429s subtitle pulls in bursts; yt-dlp's own retries don't
            # cover the subtitle download, so back off and try again.
            if "429" in str(e) and attempt < 3:
                _log(f"  ! YouTube 429 on subtitles, waiting {60 * attempt}s")
                time.sleep(60 * attempt)
                continue
            _log(f"  ! yt-dlp failed: {e}")
            return None
    for path in sorted(out_dir.glob(f"{video_id}.*.vtt")):
        return {"info": info, "vtt_path": path, "captions": kind}
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


def chunk_cues(cues, window_sec: int = 90, max_chars: int = 1800,
               min_tail_chars: int = 300) -> list[dict]:
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
        tail = " ".join(cur_text)
        if chunks and len(tail) < min_tail_chars:  # e.g. a lone "time." at the end
            chunks[-1]["end"] = cur_end
            chunks[-1]["text"] += " " + tail
        else:
            chunks.append({"start": cur_start, "end": cur_end, "text": tail})
    return chunks


# ---------- SponsorBlock ----------

SKIP_CATEGORIES = ("sponsor", "selfpromo", "interaction")


def sponsor_segments(video_id: str) -> list[tuple[float, float]]:
    """Crowd-sourced sponsor / self-promo / "like and subscribe" ranges from
    SponsorBlock (sponsor.ajay.app). [] when the video has none or the API is
    unreachable — skipping is best-effort, never a reason to fail an ingest."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({"videoID": video_id, "categories": json.dumps(SKIP_CATEGORIES)})
    try:
        with urllib.request.urlopen(f"https://sponsor.ajay.app/api/skipSegments?{q}", timeout=10) as r:
            return [tuple(s["segment"]) for s in json.load(r)]
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError):
        return []


def drop_segments(cues, segments):
    """Remove cues whose midpoint falls inside any (start, end) segment."""
    if not segments:
        return cues
    return [c for c in cues
            if not any(a <= (c[0] + c[1]) / 2 <= b for a, b in segments)]


# ---------- prefix + build ----------

def _fmt_time(sec: float) -> str:
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"


def _fmt_date(yyyymmdd: str | None) -> str:
    if yyyymmdd and len(yyyymmdd) == 8:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or ""


def build_items_for_video(info: dict, vtt_text: str, channel_handle: str | None,
                          captions: str = "", skip: list[tuple[float, float]] | None = None,
                          tags: dict | None = None) -> list[dict]:
    cues = drop_segments(parse_vtt(vtt_text), skip)
    raw_chunks = chunk_cues(cues)
    channel = info.get("channel") or info.get("uploader") or ""
    title = info.get("title") or ""
    date = _fmt_date(info.get("upload_date"))
    vid = info.get("id") or ""
    items = []
    for i, c in enumerate(raw_chunks):
        start, end = int(c["start"]), int(c["end"])
        meta = chunk_meta(
            type="transcript", origin="youtube",
            source=f"https://youtu.be/{vid}?t={start}",
            doc_id=vid, title=title, author=channel, published_at=date,
            locator=f"{_fmt_time(start)}–{_fmt_time(end)}", chunk_index=i,
            # YouTube-specific extras (list_repository counts videos/channels)
            video_id=vid, channel=channel, channel_handle=channel_handle or "",
            chunk_start_sec=start, chunk_end_sec=end, captions=captions,
            views=info.get("view_count"), likes=info.get("like_count"),
            **(tags or {}),
        )
        items.append({
            "content": header(channel, title, date, _fmt_time(start)) + "\n" + c["text"],
            "meta": meta,
        })
    return items


# ---------- channel/url helpers ----------

def _resolve_target(target: str) -> tuple[str, str]:
    """Returns (channel_url, channel_handle)."""
    t = target.strip()
    if t.startswith("@"):
        return f"https://www.youtube.com/{t}", t
    if t.startswith("http"):
        tail = t.rstrip("/").split("/")[-1]
        return t, tail if tail.startswith("@") else ""
    return f"https://www.youtube.com/@{t}", f"@{t}"


def _open_collection(collection: str):
    """Fail fast (before any yt-dlp traffic) unless `collection` is registered."""
    from yt_rag import RAGStore  # lazy
    from yt_rag.collections import CollectionRegistry, SupabaseBackend

    rag = RAGStore(namespace=collection)
    CollectionRegistry(SupabaseBackend(rag.client)).require(collection)
    return rag


def _ingest_video_ids(rag, video_ids: list[str], sub_langs, cookiefile,
                      channel_handle: str | None = None, tags: dict | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="yt_rag_"))
    totals = {"videos": len(video_ids), "chunks": 0, "created": 0, "skipped": 0, "no_subs": 0}
    for i, vid in enumerate(video_ids, 1):
        _log(f"\n[{i}/{len(video_ids)}] {vid}")
        res = download_video_subs(vid, tmp, sub_langs, cookiefile)
        if not res:
            _log("  - no matching subs, skipping")
            totals["no_subs"] += 1
            continue
        with open(res["vtt_path"]) as f:
            vtt_text = f.read()
        info = res["info"]
        handle = channel_handle
        if handle is None:
            handle = info.get("uploader_id") or ""
            if handle and not handle.startswith("@"):
                handle = f"@{handle}"
        skip = sponsor_segments(vid)
        if skip:
            _log(f"  - SponsorBlock: skipping {len(skip)} segment(s), {sum(b - a for a, b in skip):.0f}s")
        items = build_items_for_video(info, vtt_text, handle, captions=res.get("captions", ""),
                                      skip=skip, tags=tags)
        if not items:
            _log("  - 0 chunks after parsing, skipping")
            continue
        # Inline batch insert with rich metadata (RAGStore.ingest_batch only
        # accepts source/type — we need extra fields, so we go one level down).
        r = _batch_ingest_with_meta(rag, items)
        totals["chunks"] += len(items)
        totals["created"] += r["created"]
        totals["skipped"] += r["skipped"]
        _log(f"  - chunks={len(items)} created={r['created']} skipped={r['skipped']}")
    _log(f"\nDONE  videos={totals['videos']} no_subs={totals['no_subs']} "
         f"chunks={totals['chunks']} created={totals['created']} skipped={totals['skipped']}")
    totals["collection"] = rag.namespace
    return totals


# ---------- main entry points ----------

def ingest_channel(
    channel: str,
    collection: str,
    n: int = 30,
    sub_langs: list[str] | None = None,
    cookiefile: str | None = None,
    tags: dict | None = None,
) -> dict:
    """
    Pull the latest n uploads from a channel into an existing topic collection.
    The channel is only a SOURCE; the collection decides the grouping.

    Idempotent — RAGStore dedupes by content_hash.
    """
    tags = validate_tags(tags)
    rag = _open_collection(collection)
    url, handle = _resolve_target(channel)
    _log(f"channel   : {url}\ncollection: {collection}\nn_videos  : {n}\n\nenumerating...")
    video_ids = [v["id"] for v in list_channel_videos(url, n)]
    _log(f"found {len(video_ids)} videos")
    return _ingest_video_ids(rag, video_ids, sub_langs, cookiefile, channel_handle=handle,
                             tags=tags)


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


def ingest_videos(
    urls: list[str],
    collection: str,
    sub_langs: list[str] | None = None,
    cookiefile: str | None = None,
    tags: dict | None = None,
) -> dict:
    """
    Cherry-pick specific YouTube videos by URL/ID into an existing topic
    collection. Reuses the same chunking and metadata as ingest_channel.

    Idempotent — RAGStore dedupes by content_hash.
    """
    tags = validate_tags(tags)
    rag = _open_collection(collection)
    video_ids: list[str] = []
    for u in urls:
        try:
            video_ids.append(_extract_video_id(u))
        except ValueError as e:
            _log(f"  ! {e}")
    _log(f"collection: {collection}\nn_videos  : {len(video_ids)}")
    return _ingest_video_ids(rag, video_ids, sub_langs, cookiefile, tags=tags)


def _batch_ingest_with_meta(rag, items: list[dict], batch_size: int = 50) -> dict:
    """
    Batch ingest of {"content", "meta"} items (meta = chunks.chunk_meta envelope).
    Mirrors RAGStore.ingest_batch, but stores the whole envelope in the
    metadata JSONB column. Dedupes by content hash within the collection.
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

        embeddings = rag._embeddings.embed_documents([it["content"] for it in new],
                                                     titles=[it["meta"].get("title") for it in new])
        records = [{
            "content": it["content"],
            "content_hash": h,
            "embedding": emb,
            "metadata": {**it["meta"], "embed_model": rag._embeddings.model},
            "namespace": rag.namespace,
        } for it, emb, h in zip(new, embeddings, new_hashes)]
        result = rag.client.table("documents").insert(records).execute()
        created += len(result.data)
    return {"created": created, "skipped": skipped}
