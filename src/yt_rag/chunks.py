"""
One metadata envelope for every chunk, whatever it came from.

Captions, web pages and pasted text all end up as text chunks in the same
collections. Each chunk's `metadata` carries the same core fields so search
results and list_repository read the same way for every source:

    type          transcript | article | tutorial | doc | note
    origin        youtube | web | text
    source        deep link back to the exact spot (?t=274, #:~:text=Heading)
    doc_id        the parent item (video id, or a hash of the page URL)
    title, author, published_at
    locator       where in the parent: "04:34–06:06" or a section heading
    chunk_index   position within the parent

Source-specific extras (video_id, channel, chunk_start_sec, site, ...) ride
alongside. The content itself starts with a self-describing header
`[author — "title" — date — locator]` so BM25 and the embedding both see it.
"""
from __future__ import annotations

TYPES = ("transcript", "article", "tutorial", "doc", "note")


def chunk_meta(*, type: str, origin: str, source: str, doc_id: str, title: str,
               author: str, published_at: str, locator: str, chunk_index: int,
               **extra) -> dict:
    if type not in TYPES:
        raise ValueError(f"type must be one of {TYPES}, got {type!r}")
    return {
        "type": type,
        "origin": origin,
        "source": source,
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "published_at": published_at,
        "locator": locator,
        "chunk_index": chunk_index,
        **extra,
    }


def header(author: str, title: str, published_at: str, locator: str) -> str:
    parts = [author, f'"{title}"' if title else "", published_at, locator]
    return "[" + " — ".join(p for p in parts if p) + "]"
