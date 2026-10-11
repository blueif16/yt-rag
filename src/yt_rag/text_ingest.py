"""
Text ingest: web pages, PDFs, local files and pasted text → the same chunks as captions.

Entry points (all require an existing collection — see collections.py):
    ingest_urls(urls, collection, type="article")   YouTube → captions, else page
    ingest_pages(urls, collection, type="article")  fetch page → main text
    ingest_text(collection, text, title, source_url, ...)  text you already have

Pages are fetched with a Chrome-impersonating client (curl_cffi), reduced to
their main text as Markdown (trafilatura), split on headings and packed into
~1500-char chunks. Every chunk gets the chunks.chunk_meta envelope, so a page
section and a video moment come back from search in the same shape.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import sys
from urllib.parse import quote, urlparse

from yt_rag.chunks import chunk_meta, header
from yt_rag.taxonomy import validate_tags
from yt_rag.ingest import _batch_ingest_with_meta, _open_collection, ingest_videos

MAX_CHARS = 1500
MIN_TAIL_CHARS = 300   # a last chunk shorter than this joins the one before
MIN_PAGE_CHARS = 400   # less than this after extraction = error page / stub
MAX_HEADING_CHARS = 90  # longer "headings" are figure captions, kept as text
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
HTML_HEADING_RE = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.S | re.I)
PAGE_MARK_RE = re.compile(r"^<!-- page (\d+) -->$")  # inserted between PDF pages
# trafilatura's Markdown backslash-escapes these; in code (AE expressions) the
# escapes break copy-paste, e.g. "nK.time \<= time".
MD_ESCAPE_RE = re.compile(r"\\([<>*_\[\]#`|])")
YOUTUBE_RE = re.compile(r"(?:youtube\.com/(?:watch|shorts/|embed/|live/)|youtu\.be/)")
HR_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")  # a horizontal rule: no content
# A markdown table's separator row: | --- | :---: | (outer pipes optional).
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
MIN_CHUNK_TEXT = 40  # a chunk with less text than this (headings aside) joins the next
ROW_LABEL_CHARS = 100  # a long table row repeats at most this much of its first cell

# ---- boilerplate: site menus, tables of contents, account and cookie lines ----
NAV_RUN_MIN = 8    # menu-like lines in a row before a run can be a menu
NAV_RUN_LONG = 20  # this many in a row is a site menu or table of contents on its own
LIST_MARK_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+")
CODEISH_RE = re.compile(r"[(){}=;<>\[\]`$|]|\w\.[A-Za-z_]|//")
SENT_END_RE = re.compile(r"[.!?:;…\"”'’]\s*$")
LINK_ONLY_RE = re.compile(r"^\s*(?:[-*+•]\s+)?\[[^\]]+\]\([^)]*\)\s*$")
BOILERPLATE_RE = re.compile(
    r"you are logged in|send you notifications|notification settings|"
    r"cookie (?:policy|settings|preferences|consent)|accept (?:all )?cookies|"
    r"subscribe to (?:our|the) newsletter|all rights reserved|"
    r"^\W*(?:isbn|ean)\b|©\s*(?:\d{4}|[a-z])", re.I)


def _log(*a) -> None:
    print(*a, file=sys.stderr)


# ---------- boilerplate ----------

def _nav_key(line: str) -> str | None:
    """A line as a menu entry: its normalised text if it is short and reads like
    a link label (no sentence end, no code), "" for a blank line or bare bullet,
    None for anything that reads as text."""
    s = LIST_MARK_RE.sub("", line.strip()).strip("*_ ").strip()
    if not s or s in "-*+•":
        return ""
    if line.lstrip().startswith(("#", "|")) or CODEISH_RE.search(s) or NUMBERED_RE.match(line):
        return None  # headings, tables, code and numbered steps are content
    if len(s) > 60 or len(s.split()) > 6 or SENT_END_RE.search(s):
        return None
    return s.lower()


def _bridge(line: str) -> bool:
    """A longer label that can sit inside a menu: one line, at most 12 words,
    no sentence end, not a heading, table row or numbered step."""
    s = LIST_MARK_RE.sub("", line.strip()).strip("*_ ").strip()
    return (bool(s) and len(s) <= 100 and len(s.split()) <= 12 and not SENT_END_RE.search(s)
            and not line.lstrip().startswith(("#", "|")) and not NUMBERED_RE.match(line))


def _code_lines(lines: list[str]) -> list[bool]:
    """True for the lines of fenced code blocks (fences included)."""
    fences_ok = sum(l.lstrip().startswith("```") for l in lines) % 2 == 0
    out, in_fence = [], False
    for line in lines:
        fence = fences_ok and line.lstrip().startswith("```")
        out.append(in_fence or fence)
        if fence:
            in_fence = not in_fence
    return out


def _line_keys(lines: list[str], code: list[bool]) -> list[str | None]:
    """_nav_key per line; code always counts as text."""
    return [None if c else _nav_key(line) for line, c in zip(lines, code)]


def nav_keys(md: str) -> set[str]:
    """The menu-like lines of a page, for spotting the menus that pages of one
    site share (pass them to strip_boilerplate as `elsewhere`)."""
    lines = md.splitlines()
    return {k for k in _line_keys(lines, _code_lines(lines)) if k}


def strip_boilerplate(md: str, elsewhere: set[str] | None = None) -> str:
    """Drop navigation and boilerplate lines from extracted page text.

    - Menus and tables of contents: a run of at least NAV_RUN_MIN short,
      link-label-like lines (blank lines and bare bullets may sit between) that
      is very long (NAV_RUN_LONG+), repeats its own entries (CineD's
      "Language / Info / Language"), or whose entries mostly appear on other
      pages of the same site (`elsewhere`, from nav_keys). A short list of
      blend modes or menu items in the body has none of these, so it stays.
    - Link lists: four or more lines that are nothing but a Markdown link.
    - Account, cookie, newsletter, copyright and ISBN lines.
    Code fences are never touched.
    """
    lines = md.splitlines()
    code = _code_lines(lines)
    keys = _line_keys(lines, code)
    drop: set[int] = set()
    i, n = 0, len(lines)
    while i < n:
        if not keys[i]:
            i += 1
            continue
        # A run goes on through blank lines and through the odd longer label
        # ("Work with Track Mattes and Traveling Mattes", "Frame.io") as long as
        # menu entries outnumber those 3 to 1.
        j, navs, bridges = i, [], 0
        while j < n:
            if keys[j] is None:
                if code[j] or not _bridge(lines[j]) or (bridges + 1) * 3 > len(navs):
                    break
                bridges += 1
            elif keys[j]:
                navs.append(j)
            j += 1
        j = max(j, navs[-1] + 1)
        if len(navs) >= NAV_RUN_MIN:
            entries = [keys[k] for k in navs]
            distinct = set(entries)
            repeats = len(entries) - len(distinct)
            shared = len(distinct & elsewhere) / len(distinct) if elsewhere else 0.0
            if len(navs) >= NAV_RUN_LONG or repeats >= 2 or shared >= 0.5:
                drop.update(range(i, navs[-1] + 1))
        i = j
    run: list[int] = []
    for k, line in enumerate(lines + [""]):
        if k < n and not code[k] and LINK_ONLY_RE.match(line):
            run.append(k)
        elif k < n and not line.strip() and run:
            continue
        else:
            if len(run) >= 4:
                drop.update(run)
            run = []
    for k, line in enumerate(lines):
        if not code[k] and len(line.split()) <= 25 and BOILERPLATE_RE.search(line):
            drop.add(k)
    if not drop:
        return md
    return "\n".join(l for k, l in enumerate(lines) if k not in drop)


def strip_site_boilerplate(pages: list[str]) -> list[str]:
    """strip_boilerplate for pages of ONE site, so menus they share are found."""
    keysets = [nav_keys(md) for md in pages]
    out = []
    for k, md in enumerate(pages):
        elsewhere = set().union(*(ks for j, ks in enumerate(keysets) if j != k)) if len(pages) > 1 else None
        out.append(strip_boilerplate(md, elsewhere))
    return out


# ---------- markdown → sections → chunks ----------

def _blocks(md: str) -> list[tuple[list[str], str, int | None]]:
    """Split Markdown into (heading_path, paragraph, pdf_page) blocks.

    A heading line stays attached to the paragraph after it; fenced code
    blocks are never split. A page's single H1 is its title, so it is left out
    of heading paths; over-long "headings" (figure captions) count as text.
    """
    lines = md.splitlines()
    h1s = [l for l in lines if l.startswith("# ")]
    skip_h1 = len(h1s) == 1
    # An odd number of ``` lines means the extractor dropped one; trusting the
    # fences would swallow every later heading (it did on the 174k-char
    # expression reference), so ignore them.
    fences_ok = sum(l.lstrip().startswith("```") for l in lines) % 2 == 0
    path: list[tuple[int, str]] = []
    blocks: list[tuple[list[str], str, int | None]] = []
    para: list[str] = []
    pending_heading = ""
    in_fence = False
    page: int | None = None

    def flush():
        nonlocal pending_heading
        text = "\n".join(para).strip()
        if text and HR_RE.match(text):  # a rule alone carries nothing; keep the heading
            text = ""
        if text:
            if pending_heading:
                text = pending_heading + "\n" + text
                pending_heading = ""
            blocks.append(([h for _, h in path], text, page))
        para.clear()

    for line in lines:
        if fences_ok and line.lstrip().startswith("```"):
            in_fence = not in_fence
            para.append(line)
            continue
        if in_fence:
            para.append(line)
            continue
        pm = PAGE_MARK_RE.match(line.strip())
        if pm:
            flush()
            page = int(pm.group(1))
            continue
        m = HEADING_RE.match(line)
        if m and len(m.group(2)) <= MAX_HEADING_CHARS:
            flush()
            level, title = len(m.group(1)), m.group(2).strip("#*¶ ").strip()
            pending_heading = line.strip()
            if level == 1 and skip_h1:
                path.clear()
                continue
            path[:] = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
        elif not line.strip():
            flush()
        else:
            para.append(line.lstrip("#").strip() if m else line)
    flush()
    return blocks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph at sentence ends (or hard, as a last resort)."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        while len(sent) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def _cells(row: str) -> list[str]:
    s = row.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def _row_record(header: str, row: str, max_chars: int) -> list[str]:
    """A table row too long for one chunk, as `column: value` lines packed into
    pieces; every later piece starts with the row's first cell (shortened if
    long), so it reads on its own."""
    names, cells = _cells(header), _cells(row)
    if len(cells) > len(names) and names:  # a stray "|" inside a cell
        cells = cells[:len(names) - 1] + [" | ".join(cells[len(names) - 1:])]
    fields = [f"{(names[k] if k < len(names) else '').strip('* ') or f'column {k + 1}'}: {c}"
              for k, c in enumerate(cells) if c]
    if not fields:
        return _hard_split(row, max_chars)
    label = fields[0] if len(fields[0]) <= ROW_LABEL_CHARS else \
        fields[0][:ROW_LABEL_CHARS].rsplit(" ", 1)[0] + " …"
    pieces, cur = [], ""
    for f in fields:
        for part in ([f] if len(label) + 1 + len(f) <= max_chars
                     else _hard_split(f, max_chars - len(label) - 1)):
            if cur and len(cur) + 1 + len(part) > max_chars:
                pieces.append(cur)
                cur = label
            cur = f"{cur}\n{part}" if cur else part
    pieces.append(cur)
    return pieces


def _table_pieces(header: str, sep: str, rows: list[str], max_chars: int) -> list[str]:
    """Rows packed into pieces of at most max_chars, never splitting a row; every
    piece starts with the header row, so a row's columns stay named."""
    head = header + "\n" + sep
    pieces, cur = [], []
    for row in rows:
        if len(head) + 1 + len(row) > max_chars:
            if cur:
                pieces.append(head + "\n" + "\n".join(cur))
                cur = []
            pieces += _row_record(header, row, max_chars)
            continue
        if cur and len(head) + sum(len(r) + 1 for r in cur) + 1 + len(row) > max_chars:
            pieces.append(head + "\n" + "\n".join(cur))
            cur = []
        cur.append(row)
    if cur:
        pieces.append(head + "\n" + "\n".join(cur))
    return pieces


def _split_block(text: str, max_chars: int) -> list[str]:
    """Split one oversized block: Markdown tables by row (header repeated),
    everything else at sentence ends."""
    lines = text.split("\n")
    segs: list[tuple] = []
    buf: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and "|" in lines[i] and TABLE_SEP_RE.match(lines[i + 1]):
            if buf:
                segs.append(("text", "\n".join(buf)))
                buf = []
            j, rows = i + 2, []
            while j < len(lines) and "|" in lines[j]:
                rows.append(lines[j])
                j += 1
            segs.append(("table", lines[i], lines[i + 1], rows))
            i = j
        else:
            buf.append(lines[i])
            i += 1
    if buf:
        segs.append(("text", "\n".join(buf)))
    if not any(s[0] == "table" for s in segs):
        return _hard_split(text, max_chars)
    pieces: list[str] = []
    lead = ""  # a short line just before a table (its heading) travels with it
    for k, seg in enumerate(segs):
        if seg[0] == "text":
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            if nxt and nxt[0] == "table" and len(seg[1]) < 200:
                lead = seg[1]
                continue
            pieces += [seg[1]] if len(seg[1]) <= max_chars else _hard_split(seg[1], max_chars)
        else:
            tp = _table_pieces(*seg[1:], max_chars - (len(lead) + 1 if lead else 0))
            if lead:
                tp[0] = lead + "\n" + tp[0]
                lead = ""
            pieces += tp
    return pieces


def _text_chars(chunk: str) -> int:
    """Characters of real text: headings, the "(section path)" line and rules aside."""
    lines = chunk.strip().splitlines()
    if lines and lines[0].startswith("(") and lines[0].endswith(")"):
        lines = lines[1:]
    def heading(line: str) -> bool:  # over-long "headings" are text, as in _blocks
        m = HEADING_RE.match(line)
        return bool(m) and len(m.group(2)) <= MAX_HEADING_CHARS
    return sum(len(l.strip()) for l in lines
               if l.strip() and not heading(l) and not HR_RE.match(l))


def split_markdown(md: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Pack heading-aware blocks into chunks of at most ~max_chars.

    Returns [{"sections": [...], "anchor": str | None, "text": str}]:
    `sections` are the chunk's sections as labels of the two deepest heading
    levels (so an image label like "After" reads as "Control speed › After",
    and a manual's parameter as "Effect Controls › Radius"); `anchor` is the
    top-level heading the chunk starts in, for the deep link. A chunk that
    continues a section mid-way repeats the heading path on its first line, so
    every chunk reads on its own. A tiny last chunk joins the one before.
    """
    chunks: list[dict] = []
    cur_text, cur_secs, cur_anchor, cur_page = "", [], None, None

    def close():
        nonlocal cur_text, cur_secs, cur_anchor, cur_page
        if _text_chars(cur_text):  # headings or rules alone carry nothing
            if chunks and _text_chars(chunks[-1]["text"]) < MIN_CHUNK_TEXT:
                prev = chunks.pop()  # a scrap (a heading and a line) joins the next chunk
                cur_text = prev["text"] + "\n\n" + cur_text.strip()
                cur_secs = prev["sections"] + [s for s in cur_secs if s not in prev["sections"]]
                cur_anchor, cur_page = prev["anchor"], prev["page"]
            chunks.append({"sections": cur_secs, "anchor": cur_anchor, "page": cur_page,
                           "text": cur_text.strip()})
        cur_text, cur_secs, cur_anchor, cur_page = "", [], None, None

    for path, text, page in _blocks(md):
        pieces = [text] if len(text) <= max_chars else _split_block(text, max_chars)
        label = " › ".join(path[-2:])  # the two deepest levels: most specific
        for piece in pieces:
            if cur_text and len(cur_text) + 2 + len(piece) > max_chars:
                close()
            if not cur_text and path and not piece.lstrip().startswith("#"):
                piece = "(" + " › ".join(path) + ")\n" + piece
            if not cur_text:
                cur_page = page
                if path:
                    cur_anchor = path[0]
            cur_text = f"{cur_text}\n\n{piece}" if cur_text else piece
            if label and label not in cur_secs:
                cur_secs.append(label)
    close()
    if len(chunks) > 1 and len(chunks[-1]["text"]) < MIN_TAIL_CHARS:
        tail = chunks.pop()
        chunks[-1]["text"] += "\n\n" + tail["text"]
        chunks[-1]["sections"] += [x for x in tail["sections"] if x not in chunks[-1]["sections"]]
    return chunks


def strip_repeated_paragraphs(pages: list[str], min_chars: int = 30) -> list[str]:
    """Drop paragraphs that appear on two or more of the given pages (pass pages
    from ONE site): footers, promos, cookie notes. Code fences are kept."""
    def paras(md):
        return [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
    seen: dict[str, int] = {}
    for md in pages:
        for para in set(paras(md)):
            if len(para) >= min_chars and not para.startswith("```"):
                seen[para] = seen.get(para, 0) + 1
    repeated = {p for p, n in seen.items() if n >= 2}
    return ["\n\n".join(p for p in paras(md) if p not in repeated) for md in pages]


# ---------- items ----------

def page_doc_id(url: str) -> str:
    return "web:" + hashlib.sha1(url.strip().rstrip("/").encode()).hexdigest()[:16]


def _deep_link(url: str, heading: str | None) -> str:
    """Link to the section via a text fragment (#:~:text=...), which browsers
    scroll to and highlight. Plain URL when the chunk has no heading."""
    if not heading:
        return url
    base = url.split("#", 1)[0]
    return f"{base}#:~:text={quote(heading, safe='')}"


def build_items_for_text(md: str, *, title: str, source_url: str, author: str,
                         published_at: str, type: str, origin: str,
                         doc_id: str | None = None, **extra) -> list[dict]:
    doc_id = doc_id or page_doc_id(source_url)
    items = []
    chunks = split_markdown(md)
    for i, c in enumerate(chunks):
        locator = " / ".join(c["sections"][:3])
        if not locator and not c.get("page") and len(chunks) > 1:
            locator = f"part {i + 1}/{len(chunks)}"  # a page without headings
        if c.get("page"):  # PDF: viewers jump to #page=N; text fragments don't work there
            locator = f"{locator} · p. {c['page']}" if locator else f"p. {c['page']}"
            link = f"{source_url.split('#', 1)[0]}#page={c['page']}"
        else:
            link = _deep_link(source_url, c["anchor"])
        meta = chunk_meta(
            type=type, origin=origin,
            source=link,
            doc_id=doc_id, title=title, author=author, published_at=published_at,
            locator=locator, chunk_index=i, url=source_url, **extra,
        )
        loc = f"§ {locator}" if locator else ""
        items.append({"content": header(author, title, published_at, loc) + "\n" + c["text"],
                      "meta": meta})
    return items


# ---------- fetching ----------

def _pdf_date(raw: str | None) -> str:
    """PDF dates look like D:20170612094532Z → 2017-06-12."""
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def pdf_to_markdown(data: bytes, name: str = "") -> dict:
    """PDF → Markdown with headings, a `<!-- page N -->` line before each page."""
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    md = "\n\n".join(f"<!-- page {i} -->\n{p['text']}" for i, p in enumerate(pages, 1))
    md = MD_ESCAPE_RE.sub(r"\1", md)
    meta = doc.metadata or {}
    first_h = next((l.lstrip("#* ").rstrip("* ").strip() for l in md.splitlines()
                    if l.startswith("# ")), "")
    return {"markdown": md, "title": meta.get("title") or first_h or name,
            "author": meta.get("author") or "", "site": "",
            "published_at": _pdf_date(meta.get("creationDate")), "pages": doc.page_count}


def _html_to_page(html: str, url: str) -> dict:
    import trafilatura

    # Reduce every <hN> to its bare text first: trafilatura drops headings
    # wrapped in anchors/spans (Adobe Help's <h2><span><a id=…></a>Title</span>),
    # which lost every section title on helpx.adobe.com.
    html = HTML_HEADING_RE.sub(
        lambda m: f"<{m.group(1)}>{re.sub(r'<[^>]+>', '', m.group(2)).strip()}</{m.group(1)}>",
        html)
    md = trafilatura.extract(
        html, url=url, output_format="markdown", include_formatting=True,
        include_tables=True, include_links=False, include_comments=False,
        favor_recall=True,
    ) or ""
    meta = trafilatura.extract_metadata(html, default_url=url)
    return {
        "markdown": MD_ESCAPE_RE.sub(r"\1", md),
        "title": (meta.title if meta else "") or "",
        "author": (meta.author if meta else "") or "",
        "site": (meta.sitename if meta else "") or "",
        "published_at": (meta.date if meta else "") or "",
    }


def fetch_page(url: str) -> dict:
    """Fetch a source and reduce it to main-text Markdown + metadata.

    Handles web pages (HTML), PDFs (web or local — manuals, papers), and local
    files (file:///path or /abs/path: .pdf, .md, .txt, .html).
    Raises ValueError for HTTP errors and sources with no real text."""
    local = url.startswith("file://") or url.startswith("/")
    if local:
        path = Path(url[7:] if url.startswith("file://") else url)
        data, final_url = path.read_bytes(), url
        is_pdf = path.suffix.lower() == ".pdf"
    else:
        from curl_cffi import requests
        r = requests.get(url, impersonate="chrome", timeout=60, allow_redirects=True)
        if r.status_code >= 400:
            raise ValueError(f"HTTP {r.status_code}")
        data, final_url = r.content, str(r.url)
        is_pdf = ("pdf" in (r.headers.get("content-type") or "").lower()
                  or urlparse(final_url).path.lower().endswith(".pdf"))

    name = Path(urlparse(final_url).path).name
    if is_pdf:
        page = pdf_to_markdown(data, name)
    elif local and Path(name).suffix.lower() in (".md", ".markdown", ".txt"):
        md = data.decode("utf-8", errors="replace")
        first_h = next((l.lstrip("# ").strip() for l in md.splitlines() if l.startswith("# ")), "")
        page = {"markdown": md, "title": first_h or name, "author": "", "site": "",
                "published_at": ""}
    else:
        page = _html_to_page(data.decode("utf-8", errors="replace"), final_url)
    if len(page["markdown"]) < MIN_PAGE_CHARS:
        raise ValueError(f"only {len(page['markdown'])} chars of main text "
                         "(error page, stub, scanned PDF, or JS-only)")
    return {**page, "url": final_url}


# ---------- entry points ----------

def ingest_pages(urls: list[str], collection: str, type: str | dict[str, str] = "article",
                 tags: dict | None = None, tags_by_url: dict[str, dict] | None = None) -> dict:
    """Fetch web pages into an existing collection. Idempotent (content hash).

    Pass pages from the same site together: paragraphs repeated across them
    (footers, promos) are recognised as boilerplate and dropped. `type` is one
    type for all pages, or {url: type} when a batch mixes docs and tutorials.
    `tags` apply to every page; `tags_by_url` gives each page its own."""
    tags = validate_tags(tags)
    tags_by_url = {u: validate_tags(t) for u, t in (tags_by_url or {}).items()}
    rag = _open_collection(collection)
    totals = {"pages": len(urls), "failed": 0, "chunks": 0, "created": 0, "skipped": 0}
    errors, pages = [], []
    for i, url in enumerate(urls, 1):
        _log(f"[{i}/{len(urls)}] fetch {url}")
        try:
            pages.append((url, fetch_page(url)))
        except Exception as e:
            _log(f"  ! {e}")
            totals["failed"] += 1
            errors.append({"url": url, "error": str(e)})

    # Boilerplate: menus, tables of contents, account lines; plus what pages of
    # one site repeat (footers, promos, shared menus). Local files are not a
    # site: two notes may share a paragraph on purpose.
    by_site: dict[str, list[int]] = {}
    for k, (url, _) in enumerate(pages):
        site = urlparse(url).netloc
        by_site.setdefault(site or f"local:{k}", []).append(k)
    for idx in by_site.values():
        cleaned = strip_site_boilerplate([pages[k][1]["markdown"] for k in idx])
        if len(idx) > 1:
            cleaned = strip_repeated_paragraphs(cleaned)
        for k, md in zip(idx, cleaned):
            pages[k][1]["markdown"] = md

    for url, page in pages:
        author = page["author"] or page["site"]
        items = build_items_for_text(
            page["markdown"], title=page["title"], source_url=url, author=author,
            published_at=page["published_at"],
            type=type.get(url, "article") if isinstance(type, dict) else type,
            origin="web", site=page["site"], **tags_by_url.get(url, tags),
        )
        r = _batch_ingest_with_meta(rag, items)
        totals["chunks"] += len(items)
        totals["created"] += r["created"]
        totals["skipped"] += r["skipped"]
        _log(f"  - {page['title'][:60]!r} chunks={len(items)} created={r['created']} skipped={r['skipped']}")
    return {**totals, "errors": errors, "collection": collection}


def ingest_urls(urls: list[str], collection: str, type: str = "article",
                sub_langs: list[str] | None = None, cookiefile: str | None = None,
                tags: dict | None = None) -> dict:
    """Any mix of links: YouTube ones go through captions, the rest are fetched as pages."""
    yt = [u for u in urls if YOUTUBE_RE.search(u)]
    web = [u for u in urls if not YOUTUBE_RE.search(u)]
    out: dict = {"collection": collection}
    if yt:
        out["videos"] = ingest_videos(yt, collection, sub_langs=sub_langs, cookiefile=cookiefile,
                                      tags=tags)
    if web:
        out["pages"] = ingest_pages(web, collection, type=type, tags=tags)
    return out


def ingest_text(collection: str, text: str, title: str, source_url: str,
                author: str = "", published_at: str = "", type: str = "note",
                tags: dict | None = None) -> dict:
    """Store text you already have (an agent's extract, a transcript from elsewhere,
    a local Markdown file). `source_url` is required so every chunk leads back."""
    if not source_url.strip():
        raise ValueError("source_url is required — every chunk must lead back to its source.")
    tags = validate_tags(tags)
    rag = _open_collection(collection)
    items = build_items_for_text(strip_boilerplate(text), title=title, source_url=source_url,
                                 author=author, published_at=published_at, type=type,
                                 origin="text", **tags)
    r = _batch_ingest_with_meta(rag, items)
    return {"collection": collection, "chunks": len(items), **r}
