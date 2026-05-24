"""
Ingest the latest N videos from a YouTube channel into yt_rag.

Usage:
    .venv/bin/python scripts/ingest_channel.py @LewisWJackson 30
    .venv/bin/python scripts/ingest_channel.py <full_url> [n=30] [namespace]

Per chunk: ~90s of speech, ~1800 char ceiling, with a self-describing prefix
([Channel — "Title" — YYYY-MM-DD — MM:SS]) so BM25 and embeddings both have
context. Deep-link with ?t=<sec> goes into metadata.source.

Re-runs are safe — RAGStore dedupes by content_hash, and the prefix is unique
per (video, chunk_start).
"""
from __future__ import annotations
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from yt_dlp import YoutubeDL

load_dotenv(find_dotenv(usecwd=True))


# ---------- yt-dlp ----------

def list_channel_videos(channel_url: str, n: int) -> list[dict]:
    opts = {"extract_flat": True, "quiet": True, "no_warnings": True, "playlistend": n}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"{channel_url.rstrip('/')}/videos", download=False)
    return [{"id": e["id"], "title": e.get("title", "")} for e in info.get("entries", [])]


def download_video_subs(video_id: str, out_dir: Path) -> dict | None:
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US"],
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except Exception as e:
        print(f"  ! yt-dlp failed: {e}")
        return None
    for path in sorted(out_dir.glob(f"{video_id}.en*.vtt")):
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
            cues.append((start, end, text_lines[-1]))  # last line = newly-revealed caption
    # collapse consecutive duplicates (the 0.01s sentinel cue pattern)
    out = []
    for c in cues:
        if not out or out[-1][2] != c[2]:
            out.append(c)
    return out


def chunk_cues(cues, window_sec=90, max_chars=1800) -> list[dict]:
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

def fmt_time(sec: float) -> str:
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_date(yyyymmdd: str | None) -> str:
    if yyyymmdd and len(yyyymmdd) == 8:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or ""


def build_items_for_video(info: dict, vtt_text: str) -> list[dict]:
    cues = parse_vtt(vtt_text)
    raw_chunks = chunk_cues(cues)
    channel = info.get("channel") or info.get("uploader") or ""
    title = info.get("title") or ""
    date = fmt_date(info.get("upload_date"))
    vid = info.get("id") or ""
    items = []
    for c in raw_chunks:
        prefix = f'[{channel} — "{title}" — {date} — {fmt_time(c["start"])}]\n'
        items.append({
            "content": prefix + c["text"],
            "source": f"https://youtu.be/{vid}?t={int(c['start'])}",
            "type": "yt_transcript",
        })
    return items


# ---------- main ----------

def main():
    if len(sys.argv) < 2:
        print("usage: ingest_channel.py <channel_url_or_handle> [n=30] [namespace]")
        sys.exit(1)
    target = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if target.startswith("@"):
        url = f"https://www.youtube.com/{target}"
        default_ns = f"yt_{target.lstrip('@').lower()}"
    elif target.startswith("http"):
        url = target
        default_ns = "yt_" + (url.rstrip("/").split("/")[-1].lstrip("@").lower() or "channel")
    else:
        url = f"https://www.youtube.com/@{target}"
        default_ns = f"yt_{target.lower()}"
    namespace = sys.argv[3] if len(sys.argv) > 3 else default_ns

    print(f"channel  : {url}")
    print(f"n_videos : {n}")
    print(f"namespace: {namespace}")

    print("\nenumerating...")
    videos = list_channel_videos(url, n)
    print(f"found {len(videos)} videos")

    from yt_rag import RAGStore  # lazy: only after yt-dlp succeeded
    rag = RAGStore(namespace=namespace)

    tmp = Path(tempfile.mkdtemp(prefix="yt_rag_"))
    totals = {"chunks": 0, "created": 0, "skipped": 0, "no_subs": 0}
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {v['id']}  {v['title'][:65]}")
        res = download_video_subs(v["id"], tmp)
        if not res:
            print("  - no English subs, skipping")
            totals["no_subs"] += 1
            continue
        with open(res["vtt_path"]) as f:
            vtt_text = f.read()
        items = build_items_for_video(res["info"], vtt_text)
        if not items:
            print("  - 0 chunks after parsing, skipping")
            continue
        r = rag.ingest_batch(items)
        totals["chunks"] += len(items)
        totals["created"] += r["created"]
        totals["skipped"] += r["skipped"]
        print(f"  - chunks={len(items)} created={r['created']} skipped={r['skipped']}")

    print(f"\nDONE  videos={len(videos)} no_subs={totals['no_subs']} "
          f"chunks={totals['chunks']} created={totals['created']} skipped={totals['skipped']}")
    print(f"stats: {rag.stats()}")


if __name__ == "__main__":
    main()
