"""One text layer: pages and captions become chunks with the same envelope."""
import pytest

from yt_rag.chunks import chunk_meta, header
from yt_rag.ingest import build_items_for_video
from yt_rag.text_ingest import (
    YOUTUBE_RE,
    build_items_for_text,
    page_doc_id,
    split_markdown,
)

CORE = {"type", "origin", "source", "doc_id", "title", "author",
        "published_at", "locator", "chunk_index"}

PAGE = """# Expression examples

Intro paragraph about expressions.

## Wiggle randomly

wiggle(2,30)

Makes the layer wiggle 2 times per second up to 30 pixels.

## Overshoot

```
amp = .06; freq = 3; decay = 5;
n = 0;
```

Inertial bounce after the last keyframe.
"""


# ---- splitting ----

def test_sections_pack_into_one_chunk_when_small():
    chunks = split_markdown(PAGE)
    assert len(chunks) == 1
    # the single H1 is the page title, so it is not a section
    assert chunks[0]["sections"] == ["Wiggle randomly", "Overshoot"]
    assert "## Overshoot" in chunks[0]["text"]


def test_section_labels_keep_two_levels():
    md = ("# Speed\n\n## Control speed\n\nIntro text here.\n\n### Before\n\n### After\n\n"
          + "Body " * 60)
    (chunk,) = split_markdown(md)
    assert chunk["sections"] == ["Control speed", "Control speed › After"]
    assert chunk["anchor"] == "Control speed"


def test_figure_caption_headings_are_text():
    cap = ("On the left is the original DV footage (courtesy Creative License); on the right "
           "is the same shot with 720 degrees of motion blur added")
    md = f"## Motion blur\n\nText.\n\n### {cap}\n\nMore text."
    (chunk,) = split_markdown(md)
    assert chunk["sections"] == ["Motion blur"]
    assert cap in chunk["text"]


def test_tiny_tail_joins_previous_chunk():
    md = "## A\n\n" + "x" * 1400 + "\n\n## B\n\nshort tail."
    chunks = split_markdown(md)
    assert len(chunks) == 1 and chunks[0]["text"].endswith("short tail.")


def test_chunks_respect_max_chars_and_carry_context():
    md = "## Graph editor\n\n" + "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(12))
    chunks = split_markdown(md, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 + 40 for c in chunks)  # + context line
    # a chunk that starts mid-section names its section on the first line
    assert chunks[1]["text"].startswith("(Graph editor)")


def test_code_fence_is_never_split():
    md = "## Loop\n\n```\nline1\n\nline2\n```\n"
    (chunk,) = split_markdown(md)
    assert "line1\n\nline2" in chunk["text"]


def test_oversized_paragraph_is_split_at_sentences():
    md = "## Long\n\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_markdown(md, max_chars=500)
    assert len(chunks) > 3
    assert all(len(c["text"]) <= 500 + 40 for c in chunks)


# ---- envelope ----

def test_page_items_share_the_envelope():
    items = build_items_for_text(PAGE, title="Expression examples",
                                 source_url="https://helpx.adobe.com/x.html",
                                 author="helpx.adobe.com", published_at="2024-08-29",
                                 type="doc", origin="web")
    meta = items[0]["meta"]
    assert CORE <= set(meta)
    assert meta["doc_id"] == page_doc_id("https://helpx.adobe.com/x.html")
    # the first chunk starts in the page intro (no section), so it links to the page top
    assert meta["source"] == "https://helpx.adobe.com/x.html"
    assert items[0]["content"].startswith('[helpx.adobe.com — "Expression examples" — 2024-08-29 — § ')


def test_video_items_share_the_envelope():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nhello graph editor\n\n"
           "00:00:04.000 --> 00:00:08.000\nspeed graph now\n")
    info = {"id": "abcdefghijk", "title": "Easing", "channel": "Ben", "upload_date": "20250101"}
    (item,) = build_items_for_video(info, vtt, "@ben")
    meta = item["meta"]
    assert CORE <= set(meta)
    assert meta["type"] == "transcript" and meta["origin"] == "youtube"
    assert meta["doc_id"] == "abcdefghijk" and meta["author"] == "Ben"
    assert meta["source"] == "https://youtu.be/abcdefghijk?t=1"
    assert meta["locator"] == "00:01–00:08"
    assert item["content"].startswith('[Ben — "Easing" — 2025-01-01 — 00:01]\n')


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        chunk_meta(type="video", origin="x", source="s", doc_id="d", title="t",
                   author="a", published_at="", locator="", chunk_index=0)


def test_header_skips_empty_parts():
    assert header("", "T", "", "") == '["T"]'


@pytest.mark.parametrize("url,yt", [
    ("https://www.youtube.com/watch?v=zjkBMFhNj_g", True),
    ("https://youtu.be/zjkBMFhNj_g?t=3", True),
    ("https://www.youtube.com/shorts/abcdefghijk", True),
    ("https://helpx.adobe.com/after-effects/using/expression-examples.html", False),
    ("https://www.schoolofmotion.com/blog/youtube-tips", False),
])
def test_url_routing(url, yt):
    assert bool(YOUTUBE_RE.search(url)) is yt


# ---- caption track choice ----

from yt_rag.ingest import pick_caption_track  # noqa: E402


def test_manual_track_wins():
    info = {"subtitles": {"en-US": [], "live_chat": []}, "automatic_captions": {"en": [], "en-orig": []}}
    assert pick_caption_track(info, ["en.*"]) == ("en-US", "manual")


def test_auto_prefers_original_over_translation_endpoint():
    info = {"subtitles": {}, "automatic_captions": {"en": [], "en-orig": [], "de": []}}
    assert pick_caption_track(info, ["en.*"]) == ("en-orig", "auto")


def test_no_matching_track():
    info = {"subtitles": {}, "automatic_captions": {"de-orig": [], "de": []}}
    assert pick_caption_track(info, ["en.*"]) is None


# ---- boilerplate + sponsors ----

from yt_rag.ingest import chunk_cues, drop_segments  # noqa: E402
from yt_rag.text_ingest import strip_repeated_paragraphs  # noqa: E402


def test_repeated_site_paragraphs_are_dropped():
    footer = "**Create stunning 3D animations in After Effects**"
    a, b = strip_repeated_paragraphs([f"Speed text one.\n\n{footer}", f"Mattes text two.\n\n{footer}"])
    assert footer not in a and footer not in b
    assert a == "Speed text one." and b == "Mattes text two."


def test_sponsor_cues_are_dropped():
    cues = [(0, 4, "intro"), (60, 64, "this video is sponsored by"), (170, 174, "graph editor")]
    assert [c[2] for c in drop_segments(cues, [(53, 161)])] == ["intro", "graph editor"]


def test_transcript_tail_merges():
    cues = [(i * 10, i * 10 + 10, "word " * 40) for i in range(10)] + [(100, 101, "time.")]
    chunks = chunk_cues(cues)
    assert chunks[-1]["text"].endswith("time.") and len(chunks[-1]["text"]) > 300


# ---- tags + search shaping ----

from yt_rag.mcp_server import cap_per_source  # noqa: E402
from yt_rag.taxonomy import TagError, validate_tags  # noqa: E402


def test_tags_normalise():
    t = validate_tags({"topics": "easing", "scenarios": ["ui-promo", "ui-promo"],
                       "level": "advanced", "impact": "high", "tools": ["Deep-Glow-2 "]})
    assert t == {"topics": ["easing"], "scenarios": ["ui-promo"], "level": "advanced",
                 "impact": "high", "tools": ["deep-glow-2"]}


@pytest.mark.parametrize("bad", [
    {"topics": ["glow"]}, {"scenarios": ["promo"]}, {"level": "pro"},
    {"impact": "huge"}, {"tools": ["Deep Glow"]}, {"mood": "calm"},
])
def test_tags_reject_off_vocabulary(bad):
    with pytest.raises(TagError):
        validate_tags(bad)


def test_stock_only_is_kept_as_empty_list():
    assert validate_tags({"tools": []}) == {"tools": []}


def test_cap_per_source_keeps_order():
    rows = [{"id": i, "metadata": {"doc_id": d}} for i, d in enumerate("aaabca")]
    assert [r["id"] for r in cap_per_source(rows, 2)] == [0, 1, 3, 4]
    assert len(cap_per_source(rows, 0)) == 6


# ---- PDFs ----

def test_pdf_page_markers_give_page_links():
    md = ("<!-- page 1 -->\n# **Manual**\n\n## **Emitter**\n\n" + "Emitter text. " * 80 +
          "\n\n<!-- page 2 -->\n## **Particle**\n\n" + "Particle text. " * 80)
    items = build_items_for_text(md, title="Manual", source_url="https://x.com/m.pdf", author="RG",
                                 published_at="", type="doc", origin="web")
    metas = [i["meta"] for i in items]
    assert metas[0]["source"] == "https://x.com/m.pdf#page=1"
    assert metas[0]["locator"] == "Emitter · p. 1"
    assert metas[-1]["source"] == "https://x.com/m.pdf#page=2"
    assert "<!-- page" not in items[-1]["content"]


# ---- embedding model formats ----

from yt_rag.gemini_embeddings import GeminiEmbeddings, format_doc_v2, format_query_v2  # noqa: E402


class _FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        n = len(contents) if isinstance(contents, list) else 1
        return type("R", (), {"embeddings": [type("E", (), {"values": [1.0] + [0.0] * 767})()] * n})()


def _fake(model):
    e = GeminiEmbeddings.__new__(GeminiEmbeddings)
    e.model, e.output_dimensionality, e.task_type = model, 768, "RETRIEVAL_DOCUMENT"
    e.client = type("C", (), {"models": _FakeModels()})()
    return e


def test_v2_formats():
    assert format_query_v2("glow on white") == "task: search result | query: glow on white"
    assert format_doc_v2("body", None) == "title: none | text: body"


def test_v2_sends_one_content_per_document_and_no_task_type():
    e = _fake("gemini-embedding-2")
    out = e.embed_documents(["a", "b", "c"], titles=["T", None, "U"])
    assert len(out) == 3
    _, contents, config = e.client.models.calls[0]
    assert [c.parts[0].text for c in contents] == [
        "title: T | text: a", "title: none | text: b", "title: U | text: c"]
    assert config.task_type is None and config.output_dimensionality == 768
    e.embed_query("q")
    assert e.client.models.calls[-1][1] == "task: search result | query: q"


def test_v1_keeps_task_types():
    e = _fake("gemini-embedding-001")
    e.embed_query("q")
    assert e.client.models.calls[-1][2].task_type == "RETRIEVAL_QUERY"
    e.embed_documents(["a"])
    assert e.client.models.calls[-1][1] == ["a"]


# ---- caption styles ----

from yt_rag.ingest import parse_vtt  # noqa: E402

AUTO_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
so today we're
going<00:00:02.000><c> to</c><00:00:02.500><c> animate</c>

00:00:03.000 --> 00:00:03.010
going to animate

00:00:03.010 --> 00:00:05.000
going to animate
the graph editor
"""

MANUAL_VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
The text animator gives you
per-character control.

00:00:04.000 --> 00:00:07.000
Add a range selector
and offset it.
"""


def test_auto_captions_rolling_lines_once():
    words = " ".join(c[2] for c in parse_vtt(AUTO_VTT)).split()
    assert words == "so today we're going to animate the graph editor".split()


def test_manual_two_line_captions_keep_both_lines():
    text = " ".join(c[2] for c in parse_vtt(MANUAL_VTT))
    assert text == ("The text animator gives you per-character control. "
                    "Add a range selector and offset it.")


# ---- boilerplate: menus, tables of contents, account lines ----

from yt_rag.text_ingest import strip_boilerplate, strip_site_boilerplate  # noqa: E402

TOC = "\n".join(f"- {t}" for t in (
    "What's new", "Get started", "Technical requirements", "After Effects FAQ",
    "Get familiar with the interface", "Planning and setup", "Keyboard shortcuts",
    "Supported file formats", "Arrange and manage panels", "Language support",
    "Work with projects", "Composition settings", "Precomposing and nesting",
    "Import and interpret footage", "Manage footage items", "Scene Edit Detection",
    "Create layers", "Layer properties", "Blending modes", "Work with masks",
    "Export H.264"))


def test_long_site_menu_is_dropped():
    md = f"{TOC}\n\n## Export\n\nSelect File > Export > Add to Render Queue to add your composition."
    out = strip_boilerplate(md)
    assert "Keyboard shortcuts" not in out and "Add to Render Queue" in out


def test_menu_broken_by_longer_labels_is_still_one_menu():
    seg = lambda k: [f"  - Topic {k}{i}" for i in range(12)]
    md = "\n".join(["Body sentence before the menu."] + seg("a") +
                   ["  - Work with Track Mattes and Traveling Mattes"] + seg("b") + ["  - Frame.io"] + seg("c") +
                   ["", "## Export", "", "Real content here."])
    out = strip_boilerplate(md)
    assert "Topic" not in out and "Frame.io" not in out and "Traveling Mattes" not in out
    assert "Body sentence before the menu." in out and "Real content here." in out


def test_menu_that_repeats_its_own_entries_is_dropped():
    nav = "\n\n".join(["Education for Filmmakers", "Language", "Language", "Education for Filmmakers",
                       "The CineD Channels", "Info", "Language", "Education for Filmmakers",
                       "The CineD Channels", "Info"])
    md = (f"{nav}\n\nYou are logged in as\n\n"
          "We will send you notifications in your browser, every time a new article is published.\n\n"
          "Why isn't the color of my video matching between my export and my timeline?")
    out = strip_boilerplate(md)
    assert "Language" not in out and "logged in" not in out and "notifications" not in out
    assert out.strip().startswith("Why isn't the color")


def test_short_content_list_stays():
    md = ("Blend modes that brighten:\n\n" +
          "\n".join(f"- {m}" for m in ("Screen", "Add", "Lighten", "Color Dodge", "Linear Dodge",
                                        "Lighter Color", "Classic Color Dodge", "Overlay")) +
          "\n\nUse Add for glows.")
    assert strip_boilerplate(md) == md


def test_numbered_steps_and_code_are_never_menus():
    steps = "\n".join(f"{i}. Step {i}" for i in range(1, 25))
    code = "```\n" + "\n".join(f"x{i}" for i in range(30)) + "\n```"
    md = f"{steps}\n\n{code}"
    assert strip_boilerplate(md) == md


def test_menu_shared_by_pages_of_one_site_is_dropped():
    toc = "\n".join(["The 7.0 Workflow", "The Timeline", "Optimizing Your Projects", "Color Correction",
                     "Color Keying", "Rotoscoping and Paint", "Effective Motion Tracking",
                     "Virtual Cinematography", "Expressions", "Working with Light"])
    a, b = strip_site_boilerplate([f"Backlight text one here.\n\n{toc}", f"Color matching text.\n\n{toc}"])
    assert "Color Keying" not in a and "Color Keying" not in b
    assert a.strip() == "Backlight text one here." and b.strip() == "Color matching text."


def test_link_lists_and_footers_are_dropped():
    links = "\n".join(f"*   [Resource {i}](https://example.com/{i})" for i in range(5))
    md = f"Body text that matters.\n\n{links}\n\nISBN: 0321385527\n\nFlylib.com © 2008-2020."
    assert strip_boilerplate(md).strip() == "Body text that matters."


# ---- tables: chunked by row, header repeated ----

def _table(n_rows: int, cell: str = "x" * 300) -> str:
    rows = "\n".join(f"| B{i} | **Technique {i}** | {cell} |" for i in range(n_rows))
    return f"## Catalog\n\n| # | Technique | Build route |\n|---|---|---|\n{rows}\n"


def test_table_splits_between_rows_with_header_repeated():
    chunks = split_markdown(_table(12), max_chars=1000)
    assert len(chunks) > 2
    for c in chunks:
        lines = [l for l in c["text"].splitlines() if l.count("|") >= 3]
        assert lines[0] == "| # | Technique | Build route |" and lines[1] == "|---|---|---|"
        # every row is whole: it starts with its id and ends with a pipe
        assert all(l.startswith("| ") and l.endswith("|") for l in lines[2:])
    ids = [l.split("|")[1].strip() for c in chunks for l in c["text"].splitlines() if l.startswith("| B")]
    assert ids == [f"B{i}" for i in range(12)]  # no row lost or duplicated


def test_oversized_row_becomes_a_named_record():
    md = _table(1, cell="Long recipe sentence. " * 80)
    chunks = split_markdown(md, max_chars=600)
    assert len(chunks) > 1
    assert all("#: B0" in c["text"] for c in chunks)  # every piece names its row
    assert "Build route: Long recipe" in chunks[0]["text"]


def test_table_without_outer_pipes():
    rows = "\n".join(f"**Row {i}** | `attr{i}` | {'y' * 200}" for i in range(10))
    md = f"## Attributes\n\nElement | Attribute | Default\n------- | --------- | -------\n{rows}\n"
    chunks = split_markdown(md, max_chars=800)
    assert len(chunks) > 1
    assert all("Element | Attribute | Default" in c["text"] for c in chunks)


# ---- empty chunks ----

def test_rules_alone_never_make_a_chunk():
    md = "## A\n\n" + "a" * 1400 + "\n\n### C3\n\n---\n\n### D1\n\n" + "d" * 1400
    chunks = split_markdown(md)
    assert all(c["text"].strip("()# \n").replace("-", "").strip() for c in chunks)
    assert not any(c["text"].rstrip().endswith("---") for c in chunks)
    assert "### D1" in chunks[-1]["text"]


def test_overlong_heading_line_counts_as_text():
    md = "## " + "Before You Start, EXTREMELY IMPORTANT. Common problems discussed here! " * 3
    (chunk,) = split_markdown(md + "\n\nMore.")
    assert "EXTREMELY IMPORTANT" in chunk["text"]
